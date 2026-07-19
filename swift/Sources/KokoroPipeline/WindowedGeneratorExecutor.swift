import CoreML
import Foundation

/// Windowed vocoding of a global prosody plan (task T9): the 3 s ANE split
/// generator, looped over fixed 240-ASR-frame windows, applied to an
/// `x_pre`/`har` pair planned over the WHOLE frontend chunk. Duration,
/// F0Ntrain, and decoder-pre are not 3 s-limited (their `.mlpackage`s exist at
/// 7/15/30 s bucket sizes too) — only the generator's exported axes are fixed
/// at the 3 s package's shape, so a chunk longer than one window can't run a
/// single generator predict at all. `executeKokoroSynthesis`
/// (KokoroSynthesisExecutor.swift) calls `vocodeWindowed` for Stage 8/9
/// instead of a single `predictGeneratorSplit` once the chunk exceeds one
/// window.
///
/// Ground truth (README/Plans/ane-generator-a14-v1.md T9, validated by the
/// owner ear-check on `scripts/probe_windowed_vocode.py`, T8 — "plan globally,
/// vocode in windows" is perceptually transparent with the CURRENT split
/// generator): 1 "ASR frame" = 1 unit of the decoder-pre `f0`/`n_input`/
/// `x_pre` time axis = 300 output samples = 60 har/body frames. This axis is
/// what the executor calls `fullF0Len` (Stage 5) — empirically confirmed
/// against the compiled `.mlpackage`s to equal `x_pre`'s own output length
/// (240 for the 3 s bucket, 1200 for 15 s, etc. — verified via
/// `coremltools` spec inspection, not assumed), so it is what this file calls
/// `asrLen` throughout.
///
/// Window = 240 ASR frames (the 3 s split package's fixed `x_pre`/`har` input
/// geometry: 14,401 body/har frames, 72,000 samples). Halo = 20 ASR frames
/// each side. Core/stride = 200 ASR frames. For window k: `coreLo = k*200`,
/// `coreHi = min(asrLen, coreLo+200)`, `lo = max(0, coreLo-20)`,
/// `hi = min(asrLen, coreHi+20)`; keep only the core-region output,
/// hard-concat, no crossfade.
///
/// ## Fixed-shape edge windows (the one deviation from the validated probe)
///
/// `scripts/probe_windowed_vocode.py` runs PyTorch, which accepts any window
/// width, so its first/last windows are naturally SHORTER than 240 frames (no
/// halo on the missing side). The exported `.mlpackage` trunk/body pair is
/// FIXED-shape 240, so a short window cannot be fed to it, and zero-padding
/// would pollute the AdaIN time-statistics T8's ear-check validated (the
/// whole point of windowing is to keep every window's AdaIN context REAL
/// content, never synthetic zeros). `windowedVocodePlan` instead shifts a
/// short edge window INWARD until it spans exactly 240 frames — overlapping
/// into the neighboring window's territory for CONTEXT only; the KEPT core
/// region (`coreLo..<coreHi`) is unchanged, so the concat output is identical
/// to the un-shifted geometry everywhere both are defined.
/// `scripts/dump_windowed_vocode_golden.py` implements the PyTorch mirror of
/// this exact policy (the validated probe itself does NOT — see that script's
/// docstring) so the golden fixture exercises the SAME geometry this file
/// does, not the probe's dynamic-shape-only version.
public enum WindowedVocodeConstants {
    /// Window width in ASR frames — exactly the 3 s ANE split package's fixed
    /// `x_pre` input length (14,401 body/har frames / 72,000 samples).
    public static let winASR = 240
    /// Halo (context, discarded from the kept output) on each side, in ASR frames.
    public static let haloASR = 20
    /// New content per window / stride between window starts, in ASR frames.
    public static let coreASR = winASR - 2 * haloASR // 200
    /// Output samples per ASR frame (generator: 60·T_asr+1 body frames -> iSTFT -> 300·T_asr samples).
    public static let samplesPerASR = 300
    /// har/body frames per ASR frame.
    public static let bodyPerASR = 60
    /// The generator bucket the fixed-shape window geometry matches. Stages
    /// 1-7 may plan at any larger bucket (7/15/30 s); the generator ALWAYS
    /// runs this bucket's split pair, looped — never the outer bucket.
    public static let windowBucketSec = 3
}

/// The generator bucket the split trunk/body packages are requested at, for a
/// chunk whose planned `x_pre` axis is `fullF0Len` frames long and was planned
/// at `planningBucketSec`.
///
/// When the chunk exceeds one 3 s window the generator ALWAYS runs the fixed
/// 3 s split, looped (`vocodeWindowed`) — the larger 7/15/30 s bucket exists
/// only so stages 1-7 can plan prosody over the whole sentence; no generator
/// package is exported at those sizes. Below one window it runs the split at the
/// chunk's own bucket (the T7 single-predict path).
///
/// Both the synthesis path (`executeKokoroSynthesis` Stage 8/9) and the warm
/// path (`warmModels`) request the windowed split at THIS bucket, so warm-up
/// loads exactly the packages synthesis will run. Requesting it at the raw
/// `planningBucketSec` instead makes an `aneGeneratorSplit` warm-up at a > 3 s
/// bucket throw that provider's 3 s-only split guard BEFORE synthesis (which
/// asks for the split at 3 s) ever runs.
public func windowedSplitBucket(fullF0Len: Int, planningBucketSec: Int) -> Int {
    fullF0Len > WindowedVocodeConstants.winASR
        ? WindowedVocodeConstants.windowBucketSec
        : planningBucketSec
}

/// One window's slice boundaries, in ASR frames (the `fullF0Len`/`x_pre` time axis).
///
/// `lo`/`hi`: the INPUT span fed to the generator (always
/// `WindowedVocodeConstants.winASR` wide once windowing is active — see the
/// edge-overlap policy in this file's header). `coreLo`/`coreHi`: the OUTPUT
/// span actually kept from this window's waveform. The halo,
/// `[lo, coreLo)` and `[coreHi, hi)`, is discarded: it lacks full context on
/// one side and duplicates content the neighboring window's kept core already
/// covers.
public struct GeneratorWindow: Equatable {
    public let lo: Int
    public let hi: Int
    public let coreLo: Int
    public let coreHi: Int

    public init(lo: Int, hi: Int, coreLo: Int, coreHi: Int) {
        self.lo = lo
        self.hi = hi
        self.coreLo = coreLo
        self.coreHi = coreHi
    }
}

/// Build the window plan for an utterance `asrLen` ASR frames long.
///
/// Pure geometry, no model/tensor dependency, so it is tested in isolation
/// (`WindowedVocodePlanTests`) against hand-derived edge cases and the
/// dumped probe geometry, independent of any CoreML numerics.
///
/// - Precondition: `asrLen > WindowedVocodeConstants.winASR`. The executor
///   only takes the windowed path once the chunk exceeds one window; below
///   that the existing single-predict path already covers it exactly (and at
///   `asrLen == winASR` a "window" would BE the whole utterance, needing no
///   plan at all). The edge-overlap shift below also assumes at least two
///   windows, so any inward-shifted edge always has a real neighbor to
///   overlap into — guaranteed once `asrLen > winASR` because
///   `coreASR < winASR`.
public func windowedVocodePlan(asrLen: Int) -> [GeneratorWindow] {
    let core = WindowedVocodeConstants.coreASR
    let halo = WindowedVocodeConstants.haloASR
    let win = WindowedVocodeConstants.winASR
    precondition(asrLen > win, "windowedVocodePlan requires asrLen > \(win); got \(asrLen)")

    let windowCount = (asrLen + core - 1) / core // ceil(asrLen / core)
    var windows: [GeneratorWindow] = []
    windows.reserveCapacity(windowCount)
    for k in 0..<windowCount {
        let coreLo = k * core
        let coreHi = min(asrLen, coreLo + core)
        var lo = max(0, coreLo - halo)
        var hi = min(asrLen, coreHi + halo)
        // Edge-overlap: a window clipped by an utterance boundary is shifted
        // INWARD (never zero-padded) until it spans exactly `win` frames.
        // Only one side can be clipped at a time here (see the precondition
        // above), so `lo == 0` (first window) and `hi == asrLen` (last
        // window) are mutually exclusive branches in practice.
        if hi - lo < win {
            if lo == 0 {
                hi = min(asrLen, win)
            } else if hi == asrLen {
                lo = max(0, asrLen - win)
            }
        }
        windows.append(GeneratorWindow(lo: lo, hi: hi, coreLo: coreLo, coreHi: coreHi))
    }
    return windows
}

/// Result of a windowed vocode pass: the concatenated waveform plus an
/// aggregated finiteness census across every window's trunk seam + spec/phase
/// — mirrors the single-predict split's per-pass census (task T7's
/// `finiteCensus`), just summed over windows instead of one call, so a
/// windowed pass throws under the identical discipline: never schedule
/// NaN/Inf audio.
public struct WindowedVocodeResult {
    public let waveform: [Float]
    public let windowCount: Int
    public let finiteFraction: Double
    public let nonFinite: Int
    public let total: Int
}

/// Vocode a globally-planned `xPre`/`har` pair in fixed 3 s windows (task T9).
///
/// `xPre` and `harFlat` are the FULL-CHUNK outputs of stages 1-7 (decoder-pre,
/// hn-nsf) — not 3 s-limited. This loops the T7 split generator
/// (`predictGeneratorSplit`) over `windowedVocodePlan(asrLen:)`'s fixed
/// 240-frame windows, runs T3's host iSTFT per window, keeps only each
/// window's core output samples, and writes them directly into their final
/// position in the output buffer — the trim to `samplesPerASR * asrLen`
/// happens for free because the last window's `coreHi` is always `asrLen`.
///
/// - Parameters:
///   - xPre: `(1, C, asrLen)` — decoder-pre's FULL-CHUNK output.
///   - refS: `(1, 256)` — voice embedding, shared by every window unsliced
///     (AdaIN style input is per-utterance, not per-window; only the
///     activations it modulates are windowed).
///   - harFlat: flat channel-major `(HarmonicConstants.harChannels,
///     harFrames)` — `buildHar`'s FULL-CHUNK output.
///   - harFrames: `harFlat`'s time-axis length. Exactly
///     `WindowedVocodeConstants.bodyPerASR * asrLen + 1` at native Swift
///     geometry (`buildHar` is fed `f0Padded` of length `asrLen`, and its
///     STFT hop-5/pad-10 math is an exact `60x+1` map — see
///     `HarmonicSource.swift`), so every window's har slice, including the
///     last window's `[.., bodyPerASR*asrLen+1)`, stays exactly in bounds.
///   - asrLen: `xPre`'s time-axis length — the executor's `fullF0Len` for the
///     selected (possibly >3 s) bucket, i.e. what the plan calls `asr_len`.
///   - trunk: the 3 s split generator's trunk package (task T7).
///   - body: the 3 s split generator's body package (task T7).
public func vocodeWindowed(
    xPre: MLMultiArray,
    refS: MLMultiArray,
    harFlat: [Float],
    harFrames: Int,
    asrLen: Int,
    trunk: MLModel,
    body: MLModel
) throws -> WindowedVocodeResult {
    let xPreChannels = xPre.shape[1].intValue
    let windows = windowedVocodePlan(asrLen: asrLen)
    let bodyPerASR = WindowedVocodeConstants.bodyPerASR

    var waveforms: [[Float]] = []
    waveforms.reserveCapacity(windows.count)
    var totalNonFinite = 0
    var totalCount = 0

    for w in windows {
        let xWin = try sliceTime3D(source: xPre, channels: xPreChannels, lo: w.lo, hi: w.hi)
        let harWin = try sliceTime3D(
            sourceValues: harFlat,
            channels: HarmonicConstants.harChannels,
            sourceTime: harFrames,
            lo: bodyPerASR * w.lo,
            hi: bodyPerASR * w.hi + 1
        )
        let (bodyOutput, trunkSeam) = try predictGeneratorSplit(
            trunk: trunk, body: body, xPre: xWin, refS: refS, har: harWin
        )
        guard let specArray = bodyOutput.featureValue(for: "spec")?.multiArrayValue,
              let phaseArray = bodyOutput.featureValue(for: "phase")?.multiArrayValue else {
            throw PipelineError.modelNotLoaded("windowed generator body output 'spec'/'phase'")
        }
        let specValues = floatValues(from: specArray)
        let phaseValues = floatValues(from: phaseArray)

        let trunkCensus = finiteCensus(floatValues(from: trunkSeam))
        let bodyCensus = finiteCensus(specValues + phaseValues)
        totalNonFinite += trunkCensus.nonFinite + bodyCensus.nonFinite
        totalCount += trunkCensus.total + bodyCensus.total

        let windowFrameCount = specArray.shape.last!.intValue
        waveforms.append(hostISTFTInverse(spec: specValues, phase: phaseValues, frameCount: windowFrameCount))
    }

    let samples = stitchWindowedWaveforms(windows: windows, waveforms: waveforms, asrLen: asrLen)
    let fraction = totalCount > 0 ? 1.0 - Double(totalNonFinite) / Double(totalCount) : 1.0
    return WindowedVocodeResult(
        waveform: samples,
        windowCount: windows.count,
        finiteFraction: fraction,
        nonFinite: totalNonFinite,
        total: totalCount
    )
}

/// Crop each window's waveform to its core region and write it into its
/// final position, trimming to `samplesPerASR * asrLen` — the trim falls out
/// for free because the last window's `coreHi` is always `asrLen`. Pure
/// arithmetic, no model/tensor dependency, factored out of `vocodeWindowed`
/// so it is independently testable (`WindowedVocodeGoldenTests`) against
/// per-window waveforms dumped by `scripts/dump_windowed_vocode_golden.py`:
/// this crop/concat/trim, together with `windowedVocodePlan`'s geometry, is
/// T9's actual new contribution — the generator model's own numerics are
/// already SNR-gated elsewhere (T4/T6/T7) and re-testing them here at a tight
/// tolerance would just re-measure fp16-CoreML-vs-fp32-PyTorch noise.
///
/// - Precondition: `windows.count == waveforms.count`, and each
///   `waveforms[i]` is at least `samplesPerASR * (windows[i].hi -
///   windows[i].lo)` samples long (`hostISTFTInverse`'s full per-window
///   output, uncropped).
public func stitchWindowedWaveforms(
    windows: [GeneratorWindow],
    waveforms: [[Float]],
    asrLen: Int
) -> [Float] {
    precondition(windows.count == waveforms.count, "stitchWindowedWaveforms requires one waveform per window")
    let samplesPerASR = WindowedVocodeConstants.samplesPerASR
    var samples = [Float](repeating: 0, count: samplesPerASR * asrLen)
    for (w, wave) in zip(windows, waveforms) {
        let locLo = samplesPerASR * (w.coreLo - w.lo)
        let locHi = samplesPerASR * (w.coreHi - w.lo)
        let destLo = samplesPerASR * w.coreLo
        samples.withUnsafeMutableBufferPointer { dst in
            wave.withUnsafeBufferPointer { src in
                for i in 0..<(locHi - locLo) {
                    dst[destLo + i] = src[locLo + i]
                }
            }
        }
    }
    return samples
}
