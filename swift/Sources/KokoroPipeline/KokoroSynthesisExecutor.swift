import CoreML
import Foundation

/// The two Core ML packages of the rate-boundary split generator (task T7).
///
/// T6 split the ANE-admissible generator at the 2,400-frame rate boundary into
/// two packages so each stays below the A14's fvmlib object cap (the monolithic
/// `kokoro_decoder_har_ane_ln_3s` compiles on the Mac but the A14 rejects it —
/// see `README/Plans/ane-generator-a14-v1.md`, T6 gate + SPLIT DEVICE GATE RUN).
/// This struct carries the pair through the generator-stage boundary so the
/// executor can chain them; `predictGeneratorSplit` does the chaining.
///
/// - `trunk`: `x_pre`/`ref_s`/`har` → the `(1, 256, 2,400)` seam tensor.
/// - `body`: seam/`ref_s`/`har` → `spec`/`phase` `(1, 11, 14,401)`.
public struct GeneratorSplitModels {
    public let trunk: MLModel
    public let body: MLModel

    public init(trunk: MLModel, body: MLModel) {
        self.trunk = trunk
        self.body = body
    }
}

/// Supplies Core ML models to the shared synthesis executor.
///
/// The runtime pipeline uses already-loaded model dictionaries. The benchmark
/// uses a lazy cache that can evict bucket models before loading a new bucket.
public protocol KokoroModelProvider {
    func durationModelChoices() -> [DurationModelChoice]
    func availableBucketSeconds() -> [Int]
    func durationModel(choice: DurationModelChoice) throws -> MLModel
    func f0ntrainModel(tFrames: Int) throws -> MLModel
    func decoderPreModel(bucketSec: Int) throws -> MLModel
    func generatorModel(bucketSec: Int) throws -> MLModel
    /// The two-package rate-boundary split generator (task T7), or `nil` when
    /// this provider runs the single-package generator (the default — every
    /// existing provider, incl. the runtime `KokoroPipeline`, is unaffected).
    /// When non-nil, the executor runs `trunk` → `body` instead of
    /// `generatorModel(bucketSec:)`; see `predictGeneratorSplit`.
    func generatorSplitModels(bucketSec: Int) throws -> GeneratorSplitModels?
    func prepareForBucket(bucketSec: Int, tFrames: Int) throws
}

public extension KokoroModelProvider {
    func generatorSplitModels(bucketSec: Int) throws -> GeneratorSplitModels? { nil }
    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {}
}

/// Chain the rate-boundary split generator's two packages (task T7).
///
/// This is the ONE place the seam contract lives: the trunk package emits a
/// tensor named `trunk` `(1, 256, 2,400)`, and the body package consumes it
/// under the same name alongside the same `ref_s`/`har` the trunk saw
/// (`noise_convs[1]` is k=1, so the body reads `har` at body length too — see
/// `export_synth/wrappers.py` `GeneratorBodyANE`). Both `executeKokoroSynthesis`
/// and the split-parity test call this so the wiring is asserted in exactly one
/// spot.
///
/// - Returns: the body's feature provider (`spec`/`phase`) plus the trunk seam
///   tensor, so the caller can probe finiteness of BOTH halves and localize a
///   non-finite result to a stage.
public func predictGeneratorSplit(
    trunk: MLModel,
    body: MLModel,
    xPre: MLMultiArray,
    refS: MLMultiArray,
    har: MLMultiArray
) throws -> (bodyOutput: MLFeatureProvider, trunkSeam: MLMultiArray) {
    let trunkInput = try MLDictionaryFeatureProvider(dictionary: [
        "x_pre": MLFeatureValue(multiArray: xPre),
        "ref_s": MLFeatureValue(multiArray: refS),
        "har": MLFeatureValue(multiArray: har),
    ])
    let trunkOutput = try trunk.prediction(from: trunkInput)
    guard let trunkSeam = trunkOutput.featureValue(for: "trunk")?.multiArrayValue else {
        throw PipelineError.modelNotLoaded("generator split trunk output 'trunk'")
    }
    let bodyInput = try MLDictionaryFeatureProvider(dictionary: [
        "trunk": MLFeatureValue(multiArray: trunkSeam),
        "ref_s": MLFeatureValue(multiArray: refS),
        "har": MLFeatureValue(multiArray: har),
    ])
    let bodyOutput = try body.prediction(from: bodyInput)
    return (bodyOutput, trunkSeam)
}

/// Non-finite census over a flat tensor: `(finiteFraction, nonFinite, total)`.
///
/// Shared by the single-package and split finiteness gates (task T5/T7). A
/// `finiteFraction` below 1.0 means the ANE miscomputed the admitted graph —
/// see `PipelineError.nonFiniteGeneratorOutput`.
func finiteCensus(_ values: [Float]) -> (fraction: Double, nonFinite: Int, total: Int) {
    let nonFinite = values.reduce(0) { $0 + ($1.isFinite ? 0 : 1) }
    let total = values.count
    let fraction = total > 0 ? 1.0 - Double(nonFinite) / Double(total) : 1.0
    return (fraction, nonFinite, total)
}

/// Pre-tokenized synthesis request for the shared Swift/Core ML pipeline.
public struct KokoroSynthesisRequest {
    public let inputIds: [Int32]
    public let attentionMask: [Int32]
    public let refS: [Float]
    public let speed: Float
    public let seed: UInt64
    public let warmModelsBeforeTiming: Bool
    public let bucketDurationOverrideSeconds: Double?

    public init(
        inputIds: [Int32],
        attentionMask: [Int32],
        refS: [Float],
        speed: Float = 1.0,
        seed: UInt64 = 42,
        warmModelsBeforeTiming: Bool = false,
        bucketDurationOverrideSeconds: Double? = nil
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.refS = refS
        self.speed = speed
        self.seed = seed
        self.warmModelsBeforeTiming = warmModelsBeforeTiming
        self.bucketDurationOverrideSeconds = bucketDurationOverrideSeconds
    }
}

private struct DurationInputBundle {
    let provider: MLDictionaryFeatureProvider
    let idsArray: MLMultiArray
    let maskArray: MLMultiArray?
    let refSArray: MLMultiArray
    let speedArray: MLMultiArray
}

private struct DurationProbe {
    let bucketSec: Int
    let tFrames: Int
    let fullF0Len: Int
}

/// Run the Core ML Kokoro pipeline once.
///
/// This is the single orchestration path shared by `KokoroPipeline.synthesize`
/// and the `kokoro-bench` executable. Benchmark-only behavior is injected via
/// `KokoroModelProvider.prepareForBucket(...)`, `warmModelsBeforeTiming`, and
/// the optional tensor dump writer.
public func executeKokoroSynthesis(
    request: KokoroSynthesisRequest,
    modelProvider: KokoroModelProvider,
    linearWeights: [Float],
    linearBias: Float,
    tensorDump: inout TensorDumpWriter?
) throws -> SynthesisResult {
    let durationChoices = modelProvider.durationModelChoices()
    let durationChoice = try KokoroPipeline.selectDurationChoice(
        durationChoices,
        actualTokens: requestedTokenCount(
            inputIds: request.inputIds,
            attentionMask: request.attentionMask
        )
    )
    let durationInput = try buildDurationInput(
        inputIds: request.inputIds,
        attentionMask: request.attentionMask,
        refS: request.refS,
        speed: request.speed,
        choice: durationChoice
    )
    let durationModel = try modelProvider.durationModel(choice: durationChoice)

    try writeDurationInputs(durationInput, tensorDump: &tensorDump)

    if request.warmModelsBeforeTiming {
        let probe = try probeDurationAndBucket(
            input: durationInput,
            durationModel: durationModel,
            modelProvider: modelProvider,
            validTokenLimit: validTokenCount(
                predDurTokenCount: durationChoice.tokenLength,
                attentionMask: request.attentionMask
            ),
            bucketDurationOverrideSeconds: request.bucketDurationOverrideSeconds
        )
        try warmModels(
            probe: probe,
            durationModel: durationModel,
            durationInput: durationInput.provider,
            modelProvider: modelProvider
        )
    }

    var timings = StageTimings()

    // Stage 1: Duration Core ML.
    let t0 = CFAbsoluteTimeGetCurrent()
    let durOutput = try durationModel.prediction(from: durationInput.provider)
    let t1 = CFAbsoluteTimeGetCurrent()
    timings.durationCoreML = t1 - t0

    let predDurArray = durOutput.featureValue(for: "pred_dur")!.multiArrayValue!
    let dArray = durOutput.featureValue(for: "d")!.multiArrayValue!
    let tEnArray = durOutput.featureValue(for: "t_en")!.multiArrayValue!
    let tokenCount = predDurArray.shape.last!.intValue
    let validTokens = validTokenCount(
        predDurTokenCount: tokenCount,
        attentionMask: request.attentionMask
    )
    let predDur = try readDurationFrames(from: predDurArray, validCount: validTokens)
    let frames = predDur.reduce(0, +)
    let totalSeconds = Double(frames * 2) / PipelineConstants.f0FrameRate
    let bucketSelectionSeconds = request.bucketDurationOverrideSeconds ?? totalSeconds

    try writeDurationOutputs(
        predDurArray: predDurArray,
        predDur: predDur,
        dArray: dArray,
        tEnArray: tEnArray,
        tensorDump: &tensorDump
    )

    // Stage 2: alignment metadata. Tensor dumps keep the old sparse matrix as
    // debug data; the hot path expands token vectors directly in Stage 3.
    let t2 = CFAbsoluteTimeGetCurrent()
    let alignment: [Float]? = tensorDump == nil
        ? nil
        : buildAlignmentMatrix(predDur: predDur, traceLength: tokenCount, frameCount: frames)
    let t3 = CFAbsoluteTimeGetCurrent()
    timings.alignment = t3 - t2

    if let alignment {
        try tensorDump?.writeFloatArray(
            name: "alignment",
            values: alignment,
            shape: [1, tokenCount, frames]
        )
    }

    // Stage 3: direct token-vector expansion.
    let t4 = CFAbsoluteTimeGetCurrent()
    let en = try alignTokenMajorToFrames(
        source: dArray,
        predDur: predDur,
        channels: PipelineConstants.hiddenDim,
        frameCount: frames
    )
    let asr = try alignChannelMajorToFrames(
        source: tEnArray,
        predDur: predDur,
        channels: PipelineConstants.textEncoderDim,
        frameCount: frames
    )
    let t5 = CFAbsoluteTimeGetCurrent()
    timings.matrixOps = t5 - t4

    try tensorDump?.writeMLMultiArray(name: "en", array: en)
    try tensorDump?.writeMLMultiArray(name: "asr", array: asr)

    // Stage 4: F0Ntrain Core ML.
    let t6 = CFAbsoluteTimeGetCurrent()
    guard let bucketSec = selectBucket(
        totalSeconds: bucketSelectionSeconds,
        availableBuckets: modelProvider.availableBucketSeconds()
    ) else {
        throw PipelineError.noBucketAvailable
    }
    guard let tFrames = PipelineConstants.tFramesForBucket[bucketSec] else {
        throw PipelineError.modelNotLoaded("f0ntrain bucket \(bucketSec)")
    }
    try modelProvider.prepareForBucket(bucketSec: bucketSec, tFrames: tFrames)
    let f0nModel = try modelProvider.f0ntrainModel(tFrames: tFrames)
    let enPadded = try zeroPad3D(
        source: en,
        channels: PipelineConstants.hiddenDim,
        targetTime: tFrames
    )
    let sArray = try makeZeroArray2D(dim: PipelineConstants.styleDim)
    let sPtr = sArray.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<PipelineConstants.styleDim {
        sPtr[i] = request.refS[PipelineConstants.baselineDim + i]
    }

    try tensorDump?.writeMLMultiArray(name: "en_padded", array: enPadded)
    try tensorDump?.writeMLMultiArray(name: "s", array: sArray)

    let f0nInput = try MLDictionaryFeatureProvider(dictionary: [
        "en": MLFeatureValue(multiArray: enPadded),
        "s": MLFeatureValue(multiArray: sArray),
    ])
    let f0nOutput = try f0nModel.prediction(from: f0nInput)
    let f0PredArray = f0nOutput.featureValue(for: "F0_pred")!.multiArrayValue!
    let nPredArray = f0nOutput.featureValue(for: "N_pred")!.multiArrayValue!
    let t7 = CFAbsoluteTimeGetCurrent()
    timings.f0ntrainCoreML = t7 - t6

    let f0Curve = floatValues(from: f0PredArray)
    let nCurve = floatValues(from: nPredArray)

    try tensorDump?.writeFloatArray(name: "f0", values: f0Curve, shape: [1, f0Curve.count])
    try tensorDump?.writeFloatArray(name: "n", values: nCurve, shape: [1, nCurve.count])

    // Stage 5: pad to bucket geometry.
    let t8 = CFAbsoluteTimeGetCurrent()
    let bucketSamples = bucketSec * PipelineConstants.sampleRate
    let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
    let f0Padded = zeroPad1D(source: f0Curve, targetLength: fullF0Len)
    let nPadded = zeroPad1D(source: nCurve, targetLength: fullF0Len)
    let frameCount = decoderPreFrameCount(fullF0Len: fullF0Len)
    let asrPadded = try zeroPad3D(
        source: asr,
        channels: PipelineConstants.textEncoderDim,
        targetTime: frameCount
    )
    let t9 = CFAbsoluteTimeGetCurrent()
    timings.padding = t9 - t8

    try tensorDump?.writeFloatArray(name: "f0_padded", values: f0Padded, shape: [1, fullF0Len])
    try tensorDump?.writeFloatArray(name: "n_padded", values: nPadded, shape: [1, fullF0Len])
    try tensorDump?.writeMLMultiArray(name: "asr_padded", array: asrPadded)

    // Stage 6: DecoderPre Core ML.
    let t10 = CFAbsoluteTimeGetCurrent()
    let decPreModel = try modelProvider.decoderPreModel(bucketSec: bucketSec)
    let f0Array3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: f0Array3D, from: f0Padded)
    let nArray3D = try makeZeroArray3D(channels: 1, time: fullF0Len)
    copyInto(array: nArray3D, from: nPadded)
    let decRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    copyInto(array: decRefS, from: request.refS)

    let decPreInput = try MLDictionaryFeatureProvider(dictionary: [
        "asr": MLFeatureValue(multiArray: asrPadded),
        "f0": MLFeatureValue(multiArray: f0Array3D),
        "n_input": MLFeatureValue(multiArray: nArray3D),
        "ref_s": MLFeatureValue(multiArray: decRefS),
    ])
    let decPreOutput = try decPreModel.prediction(from: decPreInput)
    let xPre = decPreOutput.featureValue(for: "x_pre")!.multiArrayValue!
    let t11 = CFAbsoluteTimeGetCurrent()
    timings.decoderPre = t11 - t10

    try tensorDump?.writeMLMultiArray(name: "x_pre", array: xPre)

    // Stage 7: hn-nsf Swift DSP.
    let t12 = CFAbsoluteTimeGetCurrent()
    let harFlat: [Float]
    let harFrames: Int
    let harDebug: HarDebugComponents?
    if tensorDump != nil {
        let components = buildHarComponents(
            f0Padded: f0Padded,
            linearWeights: linearWeights,
            linearBias: linearBias,
            seed: request.seed
        )
        harFlat = components.har
        harFrames = components.nFrames
        harDebug = components
    } else {
        let built = buildHar(
            f0Padded: f0Padded,
            linearWeights: linearWeights,
            linearBias: linearBias,
            seed: request.seed
        )
        harFlat = built.har
        harFrames = built.nFrames
        harDebug = nil
    }
    let t13 = CFAbsoluteTimeGetCurrent()
    timings.hnsfSwift = t13 - t12

    if let harDebug {
        try tensorDump?.writeFloatArray(
            name: "har_source",
            values: harDebug.harSource,
            shape: [1, harDebug.harSource.count]
        )
        try tensorDump?.writeFloatArray(
            name: "har_magnitude",
            values: harDebug.magnitude,
            shape: [1, 11, harDebug.nFrames]
        )
        try tensorDump?.writeFloatArray(
            name: "har_phase",
            values: harDebug.phase,
            shape: [1, 11, harDebug.nFrames]
        )
    }
    try tensorDump?.writeFloatArray(name: "har", values: harFlat, shape: [1, 22, harFrames])

    // Stage 8/9: GeneratorFromHar Core ML (+ host iSTFT for ANE packages).
    //
    // Three shapes of generator flow through here, all entirely behind this
    // stage boundary (decoder-pre, F0, the host iSTFT, and the
    // SynthesisResult shape are untouched by the choice —
    // README/Plans/ane-generator-a14-v1.md T7/T9):
    //   - single  : one package, one predict (legacy `..._post_*s` →
    //     waveform, or the ANE `..._ane[_ln]_3s` → spec/phase).
    //   - split   : the T6 rate-boundary pair (task T7), when the provider
    //     vends a `GeneratorSplitModels`. trunk (x_pre/ref_s/har → the
    //     2,400-frame seam) then body (seam/ref_s/har → spec/phase), each
    //     half small enough to clear the A14's fvmlib cap the monolithic
    //     package hit.
    //   - windowed (task T9): when `fullF0Len` (x_pre's time axis, planned by
    //     stages 1-7 over the WHOLE frontend chunk — up to whatever bucket
    //     `selectBucket` chose, 7/15/30 s buckets exist precisely so
    //     decoder-pre/F0Ntrain can plan a whole sentence) exceeds one 3 s
    //     window, no single predict is even possible: every exported
    //     generator package is FIXED-shape at the 3 s bucket's geometry. Loop
    //     the T7 split pair over fixed 240-ASR-frame windows instead
    //     (`vocodeWindowed`, WindowedGeneratorExecutor.swift) — "plan prosody
    //     globally, vocode in 3 s windows", the design T8's owner ear-check
    //     green-lit (see this plan's Status log).
    let t14 = CFAbsoluteTimeGetCurrent()
    let genRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    copyInto(array: genRefS, from: request.refS)

    var aneGeneratorFiniteFraction: Double? = nil
    let fullWaveform: [Float]
    let t16: Double
    // Static x_pre/har time dimensions the generator model actually consumed
    // — reported in SynthesisResult below. Under windowing (task T9) this is
    // the FIXED per-window 3 s package shape (every window, not the
    // whole-chunk `fullF0Len`/`harFrames`); under single-predict it's
    // whatever `inputShapes(from:)` reads off the vended model (unchanged).
    let xPreExpectedTime: Int
    let harExpectedTime: Int

    // Windowing requires BOTH a chunk that overflows one window AND a
    // provider that actually vends the 3 s split pair. Providers that don't
    // (the runtime `KokoroPipeline` today, and every bench policy except
    // `aneGeneratorSplit` — `generatorSplitModels` defaults to `nil`) MUST
    // fall through to the unchanged single-predict path below, which already
    // handles buckets > 3 s correctly via the LEGACY per-bucket package
    // (`kokoro_decoder_har_post_<bucket>s`, one predict, no 3 s cap) — that
    // is how every non-3s bucket has always worked and T9 must not regress
    // it. Only `aneGeneratorSplit`-style providers, which guard
    // `generatorModel`/`generatorSplitModels` to the 3 s bucket ALONE, need
    // windowing to reach buckets > 3 s at all.
    if fullF0Len > WindowedVocodeConstants.winASR,
       let splitModels = try modelProvider.generatorSplitModels(
           bucketSec: windowedSplitBucket(fullF0Len: fullF0Len, planningBucketSec: bucketSec)) {
        let windowed = try vocodeWindowed(
            xPre: xPre,
            refS: genRefS,
            harFlat: harFlat,
            harFrames: harFrames,
            asrLen: fullF0Len,
            trunk: splitModels.trunk,
            body: splitModels.body
        )
        let t15 = CFAbsoluteTimeGetCurrent()
        // Covers the whole windowed loop (every window's split predict AND
        // host iSTFT, interleaved) — coarser-grained than the single-predict
        // path's generatorCoreML/trim split below, but StageTimings' FIELDS
        // are unchanged (T7's "SynthesisResult/StageTimings shape...
        // untouched" — only the final punctuation-suppression trim remains
        // for `timings.trim` to measure here).
        timings.generatorCoreML = t15 - t14
        t16 = t15
        aneGeneratorFiniteFraction = windowed.finiteFraction
        print(
            "ANEGEN: windowed windows=\(windowed.windowCount) " +
            "finite-fraction=\(String(format: "%.4f", windowed.finiteFraction)) " +
            "nonFinite=\(windowed.nonFinite) total=\(windowed.total)"
        )
        guard windowed.nonFinite == 0 else {
            throw PipelineError.nonFiniteGeneratorOutput(finiteFraction: windowed.finiteFraction)
        }
        fullWaveform = windowed.waveform
        xPreExpectedTime = WindowedVocodeConstants.winASR
        harExpectedTime = WindowedVocodeConstants.bodyPerASR * WindowedVocodeConstants.winASR + 1
    } else {
        let splitModels = try modelProvider.generatorSplitModels(bucketSec: bucketSec)
        // The model whose input geometry drives x_pre/har shaping: the trunk under
        // a split (it takes the SAME x_pre/ref_s/har as the monolithic package),
        // else the single generator.
        let genInputModel = try splitModels?.trunk ?? modelProvider.generatorModel(bucketSec: bucketSec)
        // The model that PRODUCES the outputs, keyed on to detect the ANE spec/phase
        // contract: the body under a split, else the single generator.
        let genOutputModel = splitModels?.body ?? genInputModel

        let genShapes = inputShapes(from: genInputModel)
        xPreExpectedTime = genShapes["x_pre"]?.last ?? xPre.shape.last!.intValue
        harExpectedTime = genShapes["har"]?.last ?? harFrames
        let xPrePadded = try zeroPad3D(
            source: xPre,
            channels: xPre.shape[1].intValue,
            targetTime: xPreExpectedTime
        )
        let harPadded = try zeroPad3D(
            sourceValues: harFlat,
            channels: HarmonicConstants.harChannels,
            sourceTime: harFrames,
            targetTime: harExpectedTime
        )

        // ANE package detection (T5): via the output model's own description, not
        // a policy flag threaded through the pipeline (README/Plans/ane-generator-a14-v1.md
        // T5). The ANE packages (kokoro_decoder_har_ane[_ln]_3s, or the split's
        // body) stop at spec/phase; the legacy package (kokoro_decoder_har_post_*s)
        // outputs waveform.
        let isAneGeneratorPackage = Set(genOutputModel.modelDescription.outputDescriptionsByName.keys)
            .isSuperset(of: ["spec", "phase"])
        if isAneGeneratorPackage {
            // T5's geometry finding: unlike the baseline package's har axis
            // (28,801, sized 2x the bucket's native frame count — see
            // README/Notes/ane-pretrim-equivalence-2026-07-17.md's "Flagged for
            // the owner" section), the ANE package's har axis (14,401) equals
            // exactly what buildHar naturally produces for a 3 s bucket, and
            // x_pre's 240 frames are always fully computed by decoder-pre (never
            // padded — the "120 of 240" figure in that note is asr's own INPUT
            // frame count, a different axis than x_pre's OUTPUT). Both axes are
            // therefore fully real content here with no zero-padding, verified
            // empirically against the Python reference
            // (build_decoder_har_post_inputs_np) before this branch was wired.
            print("ANEGEN: geometry x_pre_real=\(xPre.shape.last!.intValue)/\(xPreExpectedTime) har_real=\(harFrames)/\(harExpectedTime)")
            #if DEBUG
            assert(
                xPre.shape.last!.intValue >= xPreExpectedTime,
                "ANE generator x_pre input underfilled: decoder-pre produced \(xPre.shape.last!.intValue) " +
                "real frames, package expects \(xPreExpectedTime) — geometry regression, see " +
                "README/Plans/ane-generator-a14-v1.md T5"
            )
            assert(
                harFrames >= harExpectedTime,
                "ANE generator har input underfilled: buildHar produced \(harFrames) real frames, " +
                "package expects \(harExpectedTime) — geometry regression, see " +
                "README/Plans/ane-generator-a14-v1.md T5"
            )
            #endif
        }

        try tensorDump?.writeMLMultiArray(name: "x_pre_padded", array: xPrePadded)
        try tensorDump?.writeMLMultiArray(name: "har_padded", array: harPadded)

        // Split-trunk finiteness census, carried into Stage 9 so both halves are
        // reported on a single localizing line. `nil` ⇒ single-package path.
        var splitTrunkCensus: (fraction: Double, nonFinite: Int, total: Int)? = nil
        let genOutput: MLFeatureProvider
        if let splitModels {
            let (bodyOutput, trunkSeam) = try predictGeneratorSplit(
                trunk: splitModels.trunk,
                body: splitModels.body,
                xPre: xPrePadded,
                refS: genRefS,
                har: harPadded
            )
            splitTrunkCensus = finiteCensus(floatValues(from: trunkSeam))
            try tensorDump?.writeMLMultiArray(name: "generator_trunk", array: trunkSeam)
            genOutput = bodyOutput
        } else {
            let genInput = try MLDictionaryFeatureProvider(dictionary: [
                "x_pre": MLFeatureValue(multiArray: xPrePadded),
                "ref_s": MLFeatureValue(multiArray: genRefS),
                "har": MLFeatureValue(multiArray: harPadded),
            ])
            genOutput = try genInputModel.prediction(from: genInput)
        }
        let t15 = CFAbsoluteTimeGetCurrent()
        timings.generatorCoreML = t15 - t14

        // Stage 9: waveform reconstruction, then trim.
        //
        // The legacy package (kokoro_decoder_har_post_*s) carries the iSTFT
        // in-graph and returns `waveform` directly. The ANE package
        // (kokoro_decoder_har_ane_3s) stops at spec/phase — every axis past that
        // point exceeds the ANE's 16,384-elements-per-axis limit (see
        // README/Plans/ane-generator-a14-v1.md Ground Truth) — so T3's host-side
        // iSTFT (HostISTFT.swift) reconstructs the waveform here instead.
        t16 = CFAbsoluteTimeGetCurrent()
        if isAneGeneratorPackage {
            let specArray = genOutput.featureValue(for: "spec")!.multiArrayValue!
            let phaseArray = genOutput.featureValue(for: "phase")!.multiArrayValue!
            let specValues = floatValues(from: specArray)
            let phaseValues = floatValues(from: phaseArray)

            // T5's finiteness gate (T4's inheritance): on this Mac the ANE admits
            // the decoder-har-ane graph and then miscomputes it into non-finite
            // output (README/Notes/ane-generator-coreml-export-2026-07-17.md).
            // The phone's ANE is a different generation and may or may not
            // reproduce this — checked on EVERY pass so the device run makes
            // both admittance and correctness observable, not just the former.
            // Under the split (task T7) the trunk seam is censused too, so a
            // non-finite result localizes to a stage (trunk vs body) on one line.
            let bodyCensus = finiteCensus(specValues + phaseValues)
            if let trunkCensus = splitTrunkCensus {
                let combinedNonFinite = trunkCensus.nonFinite + bodyCensus.nonFinite
                let combinedTotal = trunkCensus.total + bodyCensus.total
                let combinedFraction = combinedTotal > 0
                    ? 1.0 - Double(combinedNonFinite) / Double(combinedTotal) : 1.0
                aneGeneratorFiniteFraction = combinedFraction
                print(
                    "ANEGEN: split trunk finite-fraction=\(String(format: "%.4f", trunkCensus.fraction)) " +
                    "body finite-fraction=\(String(format: "%.4f", bodyCensus.fraction)) " +
                    "trunkNonFinite=\(trunkCensus.nonFinite) bodyNonFinite=\(bodyCensus.nonFinite) total=\(combinedTotal)"
                )
                guard combinedNonFinite == 0 else {
                    throw PipelineError.nonFiniteGeneratorOutput(finiteFraction: combinedFraction)
                }
            } else {
                aneGeneratorFiniteFraction = bodyCensus.fraction
                print(
                    "ANEGEN: non-finite output check finite-fraction=\(String(format: "%.4f", bodyCensus.fraction)) " +
                    "nonFinite=\(bodyCensus.nonFinite) total=\(bodyCensus.total)"
                )
                guard bodyCensus.nonFinite == 0 else {
                    throw PipelineError.nonFiniteGeneratorOutput(finiteFraction: bodyCensus.fraction)
                }
            }

            let frameCount = specArray.shape.last!.intValue
            fullWaveform = hostISTFTInverse(spec: specValues, phase: phaseValues, frameCount: frameCount)
        } else {
            let waveformKey = genOutput.featureNames.contains("waveform") ? "waveform" : genOutput.featureNames.first!
            fullWaveform = floatValues(from: genOutput.featureValue(for: waveformKey)!.multiArrayValue!)
        }
    }

    let originalF0Len = frames * 2
    let targetLen = Int(
        round(Double(originalF0Len) / PipelineConstants.f0FrameRate * Double(PipelineConstants.sampleRate))
    )
    let trimLen = min(fullWaveform.count, targetLen)
    let rawAudio = Array(fullWaveform.prefix(trimLen))
    let expectedAudioSamples = predDur.reduce(0, +) * PipelineConstants.samplesPerDurationFrame
    #if DEBUG
    if trimLen < expectedAudioSamples {
        assertionFailure(
            "Trimmed waveform (\(trimLen) samples) is shorter than pred_dur span " +
            "(\(expectedAudioSamples) samples); punctuation suppression may be partial"
        )
    }
    #endif
    let audio = suppressPunctuationTokenAudio(
        rawAudio,
        inputIds: Array(request.inputIds.prefix(predDur.count)),
        predDur: predDur
    )
    let t17 = CFAbsoluteTimeGetCurrent()
    timings.trim = t17 - t16

    if tensorDump != nil {
        try tensorDump?.writeFloatArray(
            name: "waveform_full",
            values: fullWaveform,
            shape: [fullWaveform.count]
        )
        try tensorDump?.writeFloatArray(name: "waveform_raw_trimmed", values: rawAudio, shape: [trimLen])
        try tensorDump?.writeFloatArray(name: "waveform", values: audio, shape: [trimLen])
    }

    return SynthesisResult(
        audio: audio,
        timings: timings,
        bucketSeconds: bucketSec,
        audioDurationSeconds: Double(originalF0Len) / PipelineConstants.f0FrameRate,
        wallTimeSeconds: t17 - t0,
        predictedDurationFrames: frames,
        predictedDurationTokens: predDur.count,
        durationModelCacheKey: durationChoice.cacheKey,
        durationModelAllowsPadding: durationChoice.allowsPadding,
        durationTokenLength: durationChoice.tokenLength,
        tFrames: tFrames,
        fullF0Length: fullF0Len,
        decoderFrameCount: frameCount,
        xPreExpectedTime: xPreExpectedTime,
        harExpectedTime: harExpectedTime,
        trimSampleCount: trimLen,
        tokenDurationFrames: predDur,
        aneGeneratorFiniteFraction: aneGeneratorFiniteFraction
    )
}

private func requestedTokenCount(inputIds: [Int32], attentionMask: [Int32]) -> Int {
    let maskedTokenCount = attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
    return maskedTokenCount > 0 ? maskedTokenCount : inputIds.count
}

private func validTokenCount(predDurTokenCount: Int, attentionMask: [Int32]) -> Int {
    min(predDurTokenCount, attentionMask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) })
}

private func buildDurationInput(
    inputIds: [Int32],
    attentionMask: [Int32],
    refS: [Float],
    speed: Float,
    choice: DurationModelChoice
) throws -> DurationInputBundle {
    let tokenLength = choice.tokenLength
    let idsArray = try MLMultiArray(shape: [1, NSNumber(value: tokenLength)], dataType: .int32)
    let maskArray = choice.requiresAttentionMask
        ? try MLMultiArray(shape: [1, NSNumber(value: tokenLength)], dataType: .int32)
        : nil
    let refSArray = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    let speedArray = try MLMultiArray(shape: [1], dataType: .float32)

    let idsPtr = idsArray.dataPointer.assumingMemoryBound(to: Int32.self)
    let refSPtr = refSArray.dataPointer.assumingMemoryBound(to: Float.self)
    for i in 0..<min(inputIds.count, tokenLength) {
        idsPtr[i] = inputIds[i]
    }
    if let maskArray {
        let maskPtr = maskArray.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<min(attentionMask.count, tokenLength) {
            maskPtr[i] = attentionMask[i]
        }
    }
    for i in 0..<min(refS.count, PipelineConstants.voiceEmbeddingDim) {
        refSPtr[i] = refS[i]
    }
    speedArray[0] = NSNumber(value: speed)

    var features: [String: MLFeatureValue] = [
        "input_ids": MLFeatureValue(multiArray: idsArray),
        "ref_s": MLFeatureValue(multiArray: refSArray),
        "speed": MLFeatureValue(multiArray: speedArray),
    ]
    if let maskArray {
        features["attention_mask"] = MLFeatureValue(multiArray: maskArray)
    }

    return DurationInputBundle(
        provider: try MLDictionaryFeatureProvider(dictionary: features),
        idsArray: idsArray,
        maskArray: maskArray,
        refSArray: refSArray,
        speedArray: speedArray
    )
}

private func writeDurationInputs(
    _ input: DurationInputBundle,
    tensorDump: inout TensorDumpWriter?
) throws {
    try tensorDump?.writeMLMultiArray(name: "tokens", array: input.idsArray)
    if let maskArray = input.maskArray {
        try tensorDump?.writeMLMultiArray(name: "attention_mask", array: maskArray)
    }
    try tensorDump?.writeMLMultiArray(name: "ref_s", array: input.refSArray)
    try tensorDump?.writeMLMultiArray(name: "speed", array: input.speedArray)
}

private func writeDurationOutputs(
    predDurArray: MLMultiArray,
    predDur: [Int],
    dArray: MLMultiArray,
    tEnArray: MLMultiArray,
    tensorDump: inout TensorDumpWriter?
) throws {
    try tensorDump?.writeMLMultiArray(name: "pred_dur", array: predDurArray)
    try tensorDump?.writeInt32Array(
        name: "pred_dur_valid",
        values: predDur.map { Int32($0) },
        shape: [1, predDur.count]
    )
    try tensorDump?.writeMLMultiArray(name: "duration_d", array: dArray)
    try tensorDump?.writeMLMultiArray(name: "duration_t_en", array: tEnArray)
}

private func probeDurationAndBucket(
    input: DurationInputBundle,
    durationModel: MLModel,
    modelProvider: KokoroModelProvider,
    validTokenLimit: Int,
    bucketDurationOverrideSeconds: Double?
) throws -> DurationProbe {
    let output = try durationModel.prediction(from: input.provider)
    let predDurArray = output.featureValue(for: "pred_dur")!.multiArrayValue!
    let predDur = try readDurationFrames(from: predDurArray, validCount: validTokenLimit)
    let totalFrames = predDur.reduce(0, +)
    let totalSeconds = Double(totalFrames * 2) / PipelineConstants.f0FrameRate
    guard let bucketSec = selectBucket(
        totalSeconds: bucketDurationOverrideSeconds ?? totalSeconds,
        availableBuckets: modelProvider.availableBucketSeconds()
    ) else {
        throw PipelineError.noBucketAvailable
    }
    guard let tFrames = PipelineConstants.tFramesForBucket[bucketSec] else {
        throw PipelineError.modelNotLoaded("f0ntrain bucket \(bucketSec)")
    }
    try modelProvider.prepareForBucket(bucketSec: bucketSec, tFrames: tFrames)
    let bucketSamples = bucketSec * PipelineConstants.sampleRate
    let fullF0Len = Int(round(Double(bucketSamples) / Double(HarmonicConstants.upsampleScale)))
    return DurationProbe(
        bucketSec: bucketSec,
        tFrames: tFrames,
        fullF0Len: fullF0Len
    )
}

private func warmModels(
    probe: DurationProbe,
    durationModel: MLModel,
    durationInput: MLDictionaryFeatureProvider,
    modelProvider: KokoroModelProvider
) throws {
    _ = try durationModel.prediction(from: durationInput)

    let f0nModel = try modelProvider.f0ntrainModel(tFrames: probe.tFrames)
    let warmEnArr = try makeZeroArray3D(
        channels: PipelineConstants.hiddenDim,
        time: probe.tFrames
    )
    let warmSArr = try makeZeroArray2D(dim: PipelineConstants.styleDim)
    let warmF0nIn = try MLDictionaryFeatureProvider(dictionary: [
        "en": MLFeatureValue(multiArray: warmEnArr),
        "s": MLFeatureValue(multiArray: warmSArr),
    ])
    _ = try f0nModel.prediction(from: warmF0nIn)

    let decPreModel = try modelProvider.decoderPreModel(bucketSec: probe.bucketSec)
    let warmFrameCount = decoderPreFrameCount(fullF0Len: probe.fullF0Len)
    let warmAsr = try makeZeroArray3D(
        channels: PipelineConstants.textEncoderDim,
        time: warmFrameCount
    )
    let warmF0 = try makeZeroArray3D(channels: 1, time: probe.fullF0Len)
    let warmN = try makeZeroArray3D(channels: 1, time: probe.fullF0Len)
    let warmRefS = try makeZeroArray2D(dim: PipelineConstants.voiceEmbeddingDim)
    let warmDecIn = try MLDictionaryFeatureProvider(dictionary: [
        "asr": MLFeatureValue(multiArray: warmAsr),
        "f0": MLFeatureValue(multiArray: warmF0),
        "n_input": MLFeatureValue(multiArray: warmN),
        "ref_s": MLFeatureValue(multiArray: warmRefS),
    ])
    _ = try decPreModel.prediction(from: warmDecIn)

    // Warm zeros through every generator input axis. Under a split (task T7)
    // that means both halves — the body's `trunk` input warms exactly like any
    // other 3D axis, so one helper covers trunk, body, and the single package.
    func warmGenerator(_ model: MLModel) throws {
        var warmInputs: [String: MLFeatureValue] = [:]
        for (name, shape) in inputShapes(from: model) {
            if shape.count == 3 {
                warmInputs[name] = MLFeatureValue(
                    multiArray: try makeZeroArray3D(channels: shape[1], time: shape[2])
                )
            } else if shape.count == 2 {
                warmInputs[name] = MLFeatureValue(
                    multiArray: try makeZeroArray2D(dim: shape[1])
                )
            }
        }
        _ = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: warmInputs))
    }
    // Warm exactly the generator package(s) Stage 8/9 will run — mirror its
    // three-way branch. A chunk over one window runs the windowed 3 s split,
    // requested at `windowedSplitBucket` (NOT `probe.bucketSec`, or an
    // aneGeneratorSplit warm-up at a > 3 s bucket throws the provider's 3 s-only
    // split guard before synthesis, which asks for the split at 3 s, ever runs);
    // otherwise the T7 single-predict split at the chunk's own bucket, else the
    // single package.
    if probe.fullF0Len > WindowedVocodeConstants.winASR,
       let splitModels = try modelProvider.generatorSplitModels(
           bucketSec: windowedSplitBucket(fullF0Len: probe.fullF0Len, planningBucketSec: probe.bucketSec)) {
        try warmGenerator(splitModels.trunk)
        try warmGenerator(splitModels.body)
    } else if let splitModels = try modelProvider.generatorSplitModels(bucketSec: probe.bucketSec) {
        try warmGenerator(splitModels.trunk)
        try warmGenerator(splitModels.body)
    } else {
        try warmGenerator(try modelProvider.generatorModel(bucketSec: probe.bucketSec))
    }
}

private func decoderPreFrameCount(fullF0Len: Int) -> Int {
    (fullF0Len - 1) / 2 + 1
}
