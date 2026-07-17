/// Headless iPhone benchmark app: Core ML/ANE pipeline vs MLX Swift.
///
/// On launch this app runs the bundled bakeoff inputs (3s/7s/15s/30s,
/// voice af_heart, speed 1.0 — identical token IDs and ref_s to the Mac
/// bakeoff, produced by ``scripts/prepare_swift_bench_inputs.py``) through:
///
///   - Arm "coreml": ``executeKokoroSynthesis`` from this repo's
///     KokoroPipeline package, models precompiled to .mlmodelc by Xcode.
///     Timing boundary: token IDs in → 24 kHz PCM out (same as Mac bakeoff).
///   - Arm "mlx": ``KokoroTTS.generateAudio`` from mlalma/kokoro-ios
///     (MLX Swift). Its public API takes raw text, so this arm ALSO includes
///     Misaki G2P + tokenization. Disclosed wherever results are published.
///
/// Launch arguments (all optional):
///   --arms coreml,mlx       which arms to run (default: both; ladder mode only)
///   --keys 7s,15s,30s       which buckets (default: 3s,7s,15s,30s)
///   --out results.json      output filename in Documents
///   --mode ladder|matrix|g2p|soak|computeplan  ladder (default): walk the
///                           compute-policy fallback ladder per bucket.
///                           matrix: single-stage compute-unit flips for
///                           ANE-rejection attribution (coreml arm only; see
///                           ``BenchRunner/matrixCells``). g2p: time ONLY the
///                           Misaki G2P pass per input (via
///                           ``KokoroTTS/phonemizeOnlyForBench``) to bound the
///                           raw-text-vs-pretokenized boundary asymmetry
///                           between the two arms numerically. computeplan:
///                           dump MLComputePlan per-op preferred-device
///                           counts for one bundled model (--model <name>,
///                           no extension) — see
///                           ``BenchRunner/runComputePlanMode()``.
///   --policy aneGenerator     T5's ANE-generator viability probe: decoder-pre
///                           and the new kokoro_decoder_har_ane_3s package
///                           both on CPU+ANE (3s bucket only). See
///                           ``StagePolicy/aneGenerator``.
///   --generator-package <n>   override the aneGenerator generator package
///                           (default kokoro_decoder_har_ane_3s) — walk T6's
///                           candidates (kokoro_decoder_har_ane_ln_3s, ...) in
///                           one session. See BenchRunner.generatorPackageOverride.
///   --policy aneGeneratorSplit  T7's rate-boundary split probe: same compute
///                           units as aneGenerator, but the generator stage
///                           chains T6's two packages (trunk → body), each half
///                           small enough to clear the A14's fvmlib cap the
///                           monolithic package hit. 3s only. See
///                           ``StagePolicy/aneGeneratorSplit``.
///   --generator-split-packages <trunk>,<body>  override the aneGeneratorSplit
///                           pair (default kokoro_decoder_har_ane_ln_trunk_3s,
///                           kokoro_decoder_har_ane_ln_body_3s). See
///                           BenchRunner.generatorSplitPackagesOverride.
///   --exact-duration 1      use exact-native-LSTM duration packages
///                           (kokoro_duration_exact_tN, 780 ops) instead of
///                           the padded unrolled ones (17k-134k ops); mirrors
///                           the Mac frontier rows' exact-duration path.
///
/// Untethered runs: a home-screen launch has no launch arguments, so flags
/// fall back to Documents/launch_args.txt (seeded by the first soak run,
/// editable in the Files app — UIFileSharingEnabled). Soak mode also writes
/// per-pass telemetry (RTF, thermal, battery, phys_footprint, lifecycle
/// transitions) to Documents/soak-<timestamp>.csv, fsync'd per line, so a
/// debugger-free run still leaves a full record.
///
/// Results are appended to Documents/<out> after every (arm, key) pair so a
/// jetsam kill mid-run still leaves partial data. Per-stage timings come from
/// ``SynthesisResult/timings`` (StageTimings in KokoroPipeline.swift), which
/// the executor populates unconditionally on every call. Console lines
/// prefixed "BENCH:" mirror progress; "BENCHDONE" marks completion.
import SwiftUI
import AVFoundation
import CoreML
import KokoroPipeline
import KokoroSwift
import MLX
import MLXUtilsLibrary

// MARK: - Bench input (same JSON schema as swift/Sources/KokoroBenchmark)

struct BenchInput: Decodable {
    let key: String
    let text: String
    let voice: String
    let speed: Float
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let ref_s: [Float]
    let num_tokens: Int
    let canonical_duration_s: Double?
}

struct HnsfWeights: Decodable {
    let linear_weights: [Float]
    let linear_bias: Float
}

// MARK: - Compute policy

/// Per-stage compute-unit policy. The published Mac Config F rows run the
/// staged policy (StageComputeUnitPolicy.staged in
/// swift/Sources/KokoroBenchmark/main.swift); `.all` is the maximal policy
/// Macs accept but both test iPhones (A14 and A17 Pro) reject at first
/// predict with ANECCompile error -9, so the ladder runner walks a fallback
/// ladder per bucket and records which policy actually produced the timings.
/// Matrix mode instead flips one stage at a time to attribute the rejection
/// (see README/Plans/kokoro-iphone-performance-v1.md, Phase 1-2).
struct StagePolicy {
    let name: String
    let duration: MLComputeUnits
    let f0n: MLComputeUnits
    let decoderPre: MLComputeUnits
    let generator: MLComputeUnits

    /// Production-shaped staged policy (decoder-pre on the ANE, everything
    /// else CPU+GPU) — the policy behind the published Mac Config F rows.
    static let staged = StagePolicy(
        name: "staged",
        duration: .cpuAndGPU, f0n: .cpuAndGPU,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuAndGPU
    )

    /// Everything on CPU+ANE, GPU excluded. Not on the ladder: this is the
    /// background-synthesis viability probe (iOS bans Metal in the background
    /// but permits ANE and CPU), selected explicitly via --policy. A failure
    /// or a slow RTF here IS the data point — no fallback may mask it.
    static let cpuAndNeuralEngine = StagePolicy(
        name: "cpuAndNeuralEngine",
        duration: .cpuAndNeuralEngine, f0n: .cpuAndNeuralEngine,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuAndNeuralEngine
    )

    /// Ladder order: maximal `.all` first, then production-staged (the
    /// policy the published Mac Config F rows use), then no-ANE, then
    /// CPU-only as the last resort.
    static let ladder: [StagePolicy] = [
        StagePolicy(name: "all", duration: .all, f0n: .all, decoderPre: .all, generator: .all),
        staged,
        StagePolicy(name: "cpuAndGPU", duration: .cpuAndGPU, f0n: .cpuAndGPU, decoderPre: .cpuAndGPU, generator: .cpuAndGPU),
        StagePolicy(name: "cpuOnly", duration: .cpuOnly, f0n: .cpuOnly, decoderPre: .cpuOnly, generator: .cpuOnly),
    ]

    /// GPU-free like cpuAndNeuralEngine, but only decoder-pre — the one
    /// stage whose ANE compile is proven to succeed on the test iPhones —
    /// asks for the ANE. Everything else goes straight to CPU, skipping the
    /// doomed ANE compile attempts that jetsammed a 4 GB iPhone 12 Pro
    /// under all-stage cpuAndNeuralEngine (the padded duration models are
    /// 17k-134k-op unrolled LSTMs — a documented compile-memory hazard, see
    /// README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md).
    static let backgroundSafe = StagePolicy(
        name: "backgroundSafe",
        duration: .cpuOnly, f0n: .cpuOnly,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuOnly
    )

    /// T5's ANE-generator viability probe (README/Plans/ane-generator-a14-v1.md
    /// T5): decoder-pre AND the new `kokoro_decoder_har_ane_3s` generator
    /// package both requested on CPU+ANE. Duration/f0n are pinned `.cpuOnly`
    /// on purpose — they are not part of this probe and their unrolled-LSTM
    /// padded packages are a separate, already-documented compile-memory
    /// hazard (see `backgroundSafe`'s doc comment). 3s bucket only:
    /// `BundleModelCache.generatorModel` throws for any other bucket under
    /// this policy rather than silently falling back to the legacy package.
    static let aneGenerator = StagePolicy(
        name: "aneGenerator",
        duration: .cpuOnly, f0n: .cpuOnly,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuAndNeuralEngine
    )

    /// T7's rate-boundary split probe (README/Plans/ane-generator-a14-v1.md T7):
    /// identical compute units to `aneGenerator`, but the generator stage runs
    /// T6's two-package split (trunk → body) instead of one package. Exists
    /// because the monolithic ln package still hit the A14's fvmlib cap while
    /// both split halves compile clean (SPLIT DEVICE GATE RUN entry). Same 3 s
    /// guard: `BundleModelCache.generatorSplitModels` throws for any other
    /// bucket rather than silently rerouting to the legacy package.
    static let aneGeneratorSplit = StagePolicy(
        name: "aneGeneratorSplit",
        duration: .cpuOnly, f0n: .cpuOnly,
        decoderPre: .cpuAndNeuralEngine, generator: .cpuAndNeuralEngine
    )

    /// Policies addressable by --policy.
    static let named: [String: StagePolicy] = {
        var byName = [String: StagePolicy]()
        for p in ladder + [cpuAndNeuralEngine, backgroundSafe, aneGenerator, aneGeneratorSplit] { byName[p.name] = p }
        return byName
    }()
}

// MARK: - Bundle-backed model provider

/// Serves Xcode-precompiled .mlmodelc bundles to the synthesis executor.
///
/// Mirrors ModelCache in swift/Sources/KokoroBenchmark/main.swift, except
/// models are already compiled (Xcode runs coremlc at build time), so
/// loading is a plain `MLModel(contentsOf:)`. Loaded instances for other
/// buckets are evicted on bucket switch to keep the footprint small on a
/// 4 GB phone.
final class BundleModelCache: KokoroModelProvider {
    /// Buckets bundled in this app (10s omitted — no 10s bakeoff input).
    static let buckets = [3, 7, 15, 30]
    /// Padded duration token sizes bundled (cover 44/105/219/476-token inputs).
    static let durationSizes = [64, 128, 256, 512]
    /// Exact-native-LSTM duration sizes bundled, one per bench input's true
    /// token count (3s/7s/15s/30s inputs are 44/105/219/476 tokens). Opt-in
    /// via --exact-duration; semantics match
    /// KokoroPipeline.discoverDurationChoices (no attention mask, no padding).
    static let exactDurationSizes = [44, 105, 219, 476]
    /// Default generator package for the `aneGenerator` policy. Overridable per
    /// launch via `--generator-package <name>` (BenchRunner.generatorPackageOverride)
    /// so the owner can walk T6's candidate packages — `kokoro_decoder_har_ane_ln_3s`
    /// (layer_norm-lowered), etc. — in one device session without re-installing.
    static let defaultANEGeneratorPackage = "kokoro_decoder_har_ane_3s"
    /// Default trunk/body pair for the `aneGeneratorSplit` policy (T6's
    /// rate-boundary split of the ln-lowered generator). Overridable per launch
    /// via `--generator-split-packages <trunk>,<body>`
    /// (BenchRunner.generatorSplitPackagesOverride).
    static let defaultSplitTrunkPackage = "kokoro_decoder_har_ane_ln_trunk_3s"
    static let defaultSplitBodyPackage = "kokoro_decoder_har_ane_ln_body_3s"

    let policy: StagePolicy
    /// Bundled .mlmodelc name (no extension) the `aneGenerator` policy loads for
    /// the generator stage. Defaults to `defaultANEGeneratorPackage`.
    private let aneGeneratorPackage: String
    /// Bundled .mlmodelc names for the `aneGeneratorSplit` policy's trunk/body
    /// pair. Default to `defaultSplitTrunkPackage`/`defaultSplitBodyPackage`.
    private let splitTrunkPackage: String
    private let splitBodyPackage: String
    /// Stage family of the most recently vended model. When an ANEF compile
    /// fails at first predict, this is the best in-process hint for which
    /// stage threw; definitive attribution comes from --mode matrix.
    private(set) var lastVendedStage: String?
    private let durationConfig: MLModelConfiguration
    private let f0nConfig: MLModelConfiguration
    private let decPreConfig: MLModelConfiguration
    private let genConfig: MLModelConfiguration
    private let choices: [DurationModelChoice]
    private var durationModels: [String: MLModel] = [:]
    private var f0nModels: [Int: MLModel] = [:]
    private var decPreModels: [Int: MLModel] = [:]
    private var genModels: [Int: MLModel] = [:]
    private var genTrunkModels: [Int: MLModel] = [:]
    private var genBodyModels: [Int: MLModel] = [:]

    init(
        policy: StagePolicy,
        useExactDuration: Bool,
        generatorPackage: String? = nil,
        splitPackages: (trunk: String, body: String)? = nil
    ) {
        self.policy = policy
        self.aneGeneratorPackage = generatorPackage ?? Self.defaultANEGeneratorPackage
        self.splitTrunkPackage = splitPackages?.trunk ?? Self.defaultSplitTrunkPackage
        self.splitBodyPackage = splitPackages?.body ?? Self.defaultSplitBodyPackage
        func cfg(_ u: MLComputeUnits) -> MLModelConfiguration {
            let c = MLModelConfiguration(); c.computeUnits = u; return c
        }
        durationConfig = cfg(policy.duration)
        f0nConfig = cfg(policy.f0n)
        decPreConfig = cfg(policy.decoderPre)
        genConfig = cfg(policy.generator)
        if useExactDuration {
            choices = Self.exactDurationSizes.map { n in
                DurationModelChoice(
                    cacheKey: "exact_t\(n)",
                    tokenLength: n,
                    packageURL: Self.compiledURL("kokoro_duration_exact_t\(n)"),
                    requiresAttentionMask: false,
                    allowsPadding: false
                )
            }
        } else {
            choices = Self.durationSizes.map { n in
                DurationModelChoice(
                    cacheKey: "padded_t\(n)",
                    tokenLength: n,
                    packageURL: Self.compiledURL("kokoro_duration_t\(n)"),
                    requiresAttentionMask: true,
                    allowsPadding: true
                )
            }
        }
    }

    private static func compiledURL(_ name: String) -> URL {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            fatalError("Missing compiled model in bundle: \(name).mlmodelc — run ios-bench/prepare_resources.sh and re-run xcodegen generate")
        }
        return url
    }

    func durationModelChoices() -> [DurationModelChoice] { choices }
    func availableBucketSeconds() -> [Int] { Self.buckets }

    func durationModel(choice: DurationModelChoice) throws -> MLModel {
        lastVendedStage = "duration"
        if let m = durationModels[choice.cacheKey] { return m }
        let m = try MLModel(contentsOf: choice.packageURL, configuration: durationConfig)
        durationModels[choice.cacheKey] = m
        return m
    }

    func f0ntrainModel(tFrames: Int) throws -> MLModel {
        lastVendedStage = "f0n"
        if let m = f0nModels[tFrames] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_f0ntrain_t\(tFrames)"), configuration: f0nConfig)
        f0nModels[tFrames] = m
        return m
    }

    func decoderPreModel(bucketSec: Int) throws -> MLModel {
        lastVendedStage = "decoderPre"
        if let m = decPreModels[bucketSec] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_decoder_pre_\(bucketSec)s"), configuration: decPreConfig)
        decPreModels[bucketSec] = m
        return m
    }

    func generatorModel(bucketSec: Int) throws -> MLModel {
        lastVendedStage = "generator"
        if policy.name == "aneGeneratorSplit" {
            // The split runs through generatorSplitModels(bucketSec:); a call
            // here means a caller took the single-package path by mistake.
            throw PipelineError.modelNotLoaded(
                "aneGeneratorSplit uses the two-package split (\(splitTrunkPackage)/\(splitBodyPackage)); " +
                "generatorModel(bucketSec:) is not its entry point"
            )
        }
        if policy.name == "aneGenerator" {
            // Fail loudly rather than silently falling back to the legacy
            // package: aneGenerator's whole point is exercising the
            // ANE-admissible graph, and 3s is the only bucket small enough
            // to fit under the ANE's 16,384-elements-per-axis limit (see
            // README/Plans/ane-generator-a14-v1.md Ground Truth).
            guard bucketSec == 3 else {
                throw PipelineError.modelNotLoaded(
                    "aneGenerator policy only supports the 3s bucket (\(aneGeneratorPackage)); got \(bucketSec)s"
                )
            }
            if let m = genModels[bucketSec] { return m }
            let m = try MLModel(contentsOf: Self.compiledURL(aneGeneratorPackage), configuration: genConfig)
            genModels[bucketSec] = m
            return m
        }
        if let m = genModels[bucketSec] { return m }
        let m = try MLModel(contentsOf: Self.compiledURL("kokoro_decoder_har_post_\(bucketSec)s"), configuration: genConfig)
        genModels[bucketSec] = m
        return m
    }

    func generatorSplitModels(bucketSec: Int) throws -> GeneratorSplitModels? {
        // Only the split policy vends a pair; every other policy returns nil so
        // the executor takes the single-package path unchanged (T7).
        guard policy.name == "aneGeneratorSplit" else { return nil }
        lastVendedStage = "generator"
        // Same 3 s guard as aneGenerator: 3 s is the only bucket whose body fits
        // under the ANE's 16,384-elements-per-axis limit, and the split packages
        // are exported for it alone (README/Plans/ane-generator-a14-v1.md T6).
        // Fail loudly rather than reroute to the legacy package.
        guard bucketSec == 3 else {
            throw PipelineError.modelNotLoaded(
                "aneGeneratorSplit only supports the 3s bucket " +
                "(\(splitTrunkPackage)/\(splitBodyPackage)); got \(bucketSec)s"
            )
        }
        let trunk = try genTrunkModels[bucketSec]
            ?? MLModel(contentsOf: Self.compiledURL(splitTrunkPackage), configuration: genConfig)
        genTrunkModels[bucketSec] = trunk
        let body = try genBodyModels[bucketSec]
            ?? MLModel(contentsOf: Self.compiledURL(splitBodyPackage), configuration: genConfig)
        genBodyModels[bucketSec] = body
        return GeneratorSplitModels(trunk: trunk, body: body)
    }

    func prepareForBucket(bucketSec: Int, tFrames: Int) throws {
        f0nModels = f0nModels.filter { $0.key == tFrames }
        decPreModels = decPreModels.filter { $0.key == bucketSec }
        genModels = genModels.filter { $0.key == bucketSec }
        genTrunkModels = genTrunkModels.filter { $0.key == bucketSec }
        genBodyModels = genBodyModels.filter { $0.key == bucketSec }
    }
}

// MARK: - Stage timing serialization

/// One warm series over a (policy, bucket) pair: everything the results JSON
/// needs beyond raw wall times.
private struct WarmSeries {
    var wallTimes: [Double] = []
    var stageRows: [StageTimings] = []
    /// Thermal state name per call (including warmups) — fanless phones
    /// throttle, and a `serious`/`critical` row must not be promoted.
    var thermalStates: [String] = []
    var bucketSeconds: Int = 0
    var durationCacheKey: String = ""
}

/// StageTimings → JSON dict (seconds). Keys mirror the StageTimings field
/// names so iPhone rows diff directly against Mac kokoro-bench output.
private func stageDict(_ t: StageTimings) -> [String: Double] {
    [
        "duration_coreml": t.durationCoreML,
        "alignment": t.alignment,
        "matrix_ops": t.matrixOps,
        "f0ntrain_coreml": t.f0ntrainCoreML,
        "padding": t.padding,
        "decoder_pre": t.decoderPre,
        "hnsf_swift": t.hnsfSwift,
        "decoder_pre_hnsf_overlap": t.decoderPreHnsfOverlap,
        "generator_coreml": t.generatorCoreML,
        "trim": t.trim,
        "total": t.total,
    ]
}

private func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    return s.count % 2 == 0 ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2 : s[s.count / 2]
}

/// Per-stage arrays and medians across the warm rows of a series.
private func stageSummaries(_ rows: [StageTimings]) -> (arrays: [String: [Double]], medians: [String: Double]) {
    var arrays: [String: [Double]] = [:]
    for row in rows {
        for (k, v) in stageDict(row) {
            arrays[k, default: []].append(v)
        }
    }
    return (arrays, arrays.mapValues { median($0) })
}

private func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

/// App phys_footprint in MB — the number jetsam decisions are made on
/// (matches Xcode's memory gauge, unlike resident_size). 0 on failure.
private func physFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
    )
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return Double(info.phys_footprint) / 1_048_576
}

/// Battery percent (0–100; -1 if unavailable) and charge state. The state
/// column is the confound check for untethered soaks: it must read
/// "unplugged" for the run to say anything about real-world heat/battery.
/// Requires UIDevice.isBatteryMonitoringEnabled (set in runSoak).
@MainActor
private func batterySnapshot() -> (pct: Double, state: String) {
    let level = UIDevice.current.batteryLevel
    let state: String
    switch UIDevice.current.batteryState {
    case .unplugged: state = "unplugged"
    case .charging: state = "charging"
    case .full: state = "full"
    case .unknown: state = "unknown"
    @unknown default: state = "unknown"
    }
    return (level >= 0 ? Double(level) * 100 : -1, state)
}

/// Appends soak telemetry to Documents/soak-<timestamp>.csv, one line per
/// pass or lifecycle event, fsync'd per line so a crash/jetsam keeps the
/// tail. This is the artifact an untethered run (no Xcode console) leaves
/// behind; UIFileSharingEnabled makes it visible in the Files app mid-run.
final class SoakCSVLogger: @unchecked Sendable {
    let url: URL
    private let handle: FileHandle
    private let lock = NSLock()
    private let start = Date()
    private let iso = ISO8601DateFormatter()

    init?() {
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("soak-\(df.string(from: Date())).csv")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        guard let h = try? FileHandle(forWritingTo: url) else { return nil }
        handle = h
        write("time,elapsed_s,event,pass,wall_s,audio_s,x_realtime,app_state,thermal,buffered_s,battery_pct,battery_state,footprint_mb")
    }

    func pass(_ n: Int, wall: Double, audio: Double, appState: String,
              buffered: Double, batteryPct: Double, batteryState: String) {
        row(event: "pass", pass: "\(n)",
            wall: String(format: "%.3f", wall),
            audio: String(format: "%.2f", audio),
            x: String(format: "%.2f", wall > 0 ? audio / wall : 0),
            appState: appState,
            buffered: String(format: "%.1f", buffered),
            batteryPct: String(format: "%.1f", batteryPct),
            batteryState: batteryState)
    }

    func event(_ label: String, appState: String = "",
               batteryPct: Double? = nil, batteryState: String = "") {
        row(event: label.replacingOccurrences(of: ",", with: ";"),
            pass: "", wall: "", audio: "", x: "", appState: appState, buffered: "",
            batteryPct: batteryPct.map { String(format: "%.1f", $0) } ?? "",
            batteryState: batteryState)
    }

    private func row(event: String, pass: String, wall: String, audio: String,
                     x: String, appState: String, buffered: String,
                     batteryPct: String, batteryState: String) {
        let elapsed = String(format: "%.1f", Date().timeIntervalSince(start))
        write("\(iso.string(from: Date())),\(elapsed),\(event),\(pass),\(wall),\(audio),\(x),\(appState),\(thermalStateName()),\(buffered),\(batteryPct),\(batteryState),\(String(format: "%.1f", physFootprintMB()))")
    }

    private func write(_ line: String) {
        lock.lock()
        defer { lock.unlock() }
        handle.write(Data((line + "\n").utf8))
        try? handle.synchronize()
    }
}

// MARK: - Runner

@MainActor
final class BenchRunner: ObservableObject {
    @Published var status = "starting…"

    static let warmups = 2
    static let iterations = 5

    /// Launch-argument access. Arms were split into separate processes after
    /// the iPhone 12 Pro (4 GB) jetsammed with both pipelines resident
    /// (signal 9 during the MLX 7s generation).
    ///
    /// Untethered fallback: a Springboard (home-screen) launch carries no
    /// custom arguments, so when the process has none at all, flags are read
    /// from Documents/launch_args.txt instead (whitespace-separated; written
    /// once by the first soak run, editable in the Files app). All-or-nothing
    /// on purpose: any Xcode-supplied argument disables the file entirely, so
    /// a stale soak file can't hijack a tethered ladder/matrix run.
    static let effectiveArgs: [String] = {
        let procArgs = Array(ProcessInfo.processInfo.arguments.dropFirst())
        if !procArgs.isEmpty { return procArgs }
        guard let s = try? String(contentsOf: launchArgsFileURL, encoding: .utf8) else { return [] }
        return s.split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }()

    static let launchArgsFileURL = FileManager.default
        .urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("launch_args.txt")

    static func argValue(_ flag: String) -> String? {
        guard let i = effectiveArgs.firstIndex(of: flag), i + 1 < effectiveArgs.count else { return nil }
        return effectiveArgs[i + 1]
    }
    static let arms = (argValue("--arms") ?? "coreml,mlx").split(separator: ",").map(String.init)
    static let keys = (argValue("--keys") ?? "3s,7s,15s,30s").split(separator: ",").map(String.init)
    static let outName = argValue("--out") ?? "results.json"
    static let mode = argValue("--mode") ?? "ladder"
    static let exactDuration = (argValue("--exact-duration") ?? "0") == "1"
    /// --policy <name>: pin the coreml arm to one StagePolicy (see
    /// StagePolicy.named) with NO ladder fallback — a failure is the data
    /// point. Soak mode defaults to cpuAndNeuralEngine.
    static let policyOverride = argValue("--policy")
    /// --soak-seconds N: how long soak mode keeps synthesizing (default 900).
    static let soakSeconds = Double(argValue("--soak-seconds") ?? "900") ?? 900
    /// --model <name>: bundled .mlmodelc name for --mode computeplan (no
    /// extension, e.g. "kokoro_decoder_har_ane_3s" or "kokoro_decoder_pre_3s").
    static let modelOverride = argValue("--model")
    /// --generator-package <name>: override which bundled package the
    /// `aneGenerator` policy loads for the generator stage (default:
    /// `kokoro_decoder_har_ane_3s`). Lets the owner walk T6's candidate packages
    /// — e.g. `kokoro_decoder_har_ane_ln_3s` (layer_norm-lowered, ~35% fewer ops)
    /// — in one device session. See README/Plans/ane-generator-a14-v1.md T6.
    static let generatorPackageOverride = argValue("--generator-package")
    /// --generator-split-packages <trunk>,<body>: override the two packages the
    /// `aneGeneratorSplit` policy chains for the generator stage (default:
    /// `kokoro_decoder_har_ane_ln_trunk_3s`,`kokoro_decoder_har_ane_ln_body_3s`).
    /// Comma-separated, exactly two names, no extensions — a malformed value is
    /// a fatal launch error, never a silent fallback. See
    /// README/Plans/ane-generator-a14-v1.md T7.
    static let generatorSplitPackagesOverride: (trunk: String, body: String)? = {
        guard let raw = argValue("--generator-split-packages") else { return nil }
        let parts = raw.split(separator: ",", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            fatalError("--generator-split-packages expects <trunk>,<body> (two non-empty names); got \(raw)")
        }
        return (trunk: parts[0], body: parts[1])
    }()
    /// Soak mode synthesizes one bucket per run (fresh models per bucket
    /// would re-trigger AOT compiles mid-soak): the first explicit --keys
    /// entry, else 15s (nearest bucket to FreeReader's chunk size).
    static let soakKey = (argValue("--keys")?.split(separator: ",").first).map(String.init) ?? "15s"

    /// Ladder-mode policies: the full fallback ladder, or just the pinned
    /// policy when --policy is given.
    static let activeLadder: [StagePolicy] = {
        if let name = policyOverride {
            guard let p = StagePolicy.named[name] else {
                fatalError("--policy \(name) unknown; known: \(StagePolicy.named.keys.sorted().joined(separator: ","))")
            }
            return [p]
        }
        return StagePolicy.ladder
    }()

    private var records: [[String: Any]] = []

    func log(_ s: String) {
        print("BENCH: \(s)")
        status = s
    }

    private var resultsURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(Self.outName)
    }

    /// Persist all records so far — called after every record so a mid-run
    /// jetsam still leaves usable partial data on disk.
    private func flush() {
        var uts = utsname(); uname(&uts)
        let hw = withUnsafeBytes(of: &uts.machine) { raw in
            String(cString: raw.bindMemory(to: CChar.self).baseAddress!)
        }
        let payload: [String: Any] = [
            "device": UIDevice.current.model,
            "hardware": hw,  // e.g. iPhone13,3 = iPhone 12 Pro, iPhone16,2 = 15 Pro Max
            "system": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "warmups": Self.warmups,
            "iterations": Self.iterations,
            "mode": Self.mode,
            "exact_duration": Self.exactDuration,
            "records": records,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .prettyPrinted]) {
            try? data.write(to: resultsURL)
        }
    }

    private func loadInput(_ key: String) throws -> BenchInput {
        let url = Bundle.main.url(forResource: key, withExtension: "json")!
        return try JSONDecoder().decode(BenchInput.self, from: Data(contentsOf: url))
    }

    func run() {
        Task.detached(priority: .userInitiated) {
            if Self.mode == "matrix" {
                do {
                    try await self.runCoreMLMatrix()
                } catch {
                    await self.log("matrix run failed: \(error)")
                    await self.record(["arm": "coreml", "key": "matrix-level", "error": String(describing: error)])
                }
            } else if Self.mode == "soak" {
                do {
                    try await self.runSoak()
                } catch {
                    await self.log("soak run failed: \(error)")
                    await self.record(["arm": "soak", "key": "mode-level", "error": String(describing: error)])
                }
            } else if Self.mode == "g2p" {
                do {
                    try await self.runG2PMode()
                } catch {
                    await self.log("g2p run failed: \(error)")
                    await self.record(["arm": "g2p", "key": "mode-level", "error": String(describing: error)])
                }
            } else if Self.mode == "computeplan" {
                do {
                    try await self.runComputePlanMode()
                } catch {
                    await self.log("computeplan run failed: \(error)")
                    await self.record(["arm": "computeplan", "key": "mode-level", "error": String(describing: error)])
                }
            } else {
                // Arms are isolated: a Core ML failure must not block the MLX
                // arm, and vice versa. Per-bucket failures are recorded inside
                // each arm and the run continues.
                if Self.arms.contains("coreml") {
                    do {
                        try await self.runCoreMLArm()
                    } catch {
                        await self.log("coreml arm failed: \(error)")
                        await self.record(["arm": "coreml", "key": "arm-level", "error": String(describing: error)])
                    }
                }
                if Self.arms.contains("mlx") {
                    do {
                        try await self.runMLXArm()
                    } catch {
                        await self.log("mlx arm failed: \(error)")
                        await self.record(["arm": "mlx", "key": "arm-level", "error": String(describing: error)])
                    }
                }
            }
            await self.log("BENCHDONE")
            print("BENCHDONE")
        }
    }

    nonisolated private func record(_ rec: [String: Any]) async {
        await MainActor.run {
            self.records.append(rec)
            self.flush()
        }
    }

    nonisolated private func loadHnsfWeights() throws -> HnsfWeights {
        let wURL = Bundle.main.url(forResource: "hnsf_weights", withExtension: "json")!
        return try JSONDecoder().decode(HnsfWeights.self, from: Data(contentsOf: wURL))
    }

    /// One synthesis call against the given cache. Throws on Core ML failure.
    nonisolated private func synthesizeOnce(
        _ input: BenchInput, weights: HnsfWeights, cache: BundleModelCache
    ) throws -> SynthesisResult {
        var dump: TensorDumpWriter? = nil
        return try executeKokoroSynthesis(
            request: KokoroSynthesisRequest(
                inputIds: input.input_ids,
                attentionMask: input.attention_mask,
                refS: input.ref_s,
                speed: input.speed,
                seed: 42,
                warmModelsBeforeTiming: true,
                bucketDurationOverrideSeconds: input.canonical_duration_s
            ),
            modelProvider: cache,
            linearWeights: weights.linear_weights,
            linearBias: weights.linear_bias,
            tensorDump: &dump
        )
    }

    /// Warmups + recorded iterations for one (policy, bucket) pair. Throws on
    /// the first failed call; the caller decides whether to ladder-step
    /// (ladder mode) or record the cell as failed (matrix mode).
    nonisolated private func runWarmSeries(
        _ input: BenchInput, weights: HnsfWeights, cache: BundleModelCache,
        key: String, policyName: String
    ) async throws -> WarmSeries {
        var series = WarmSeries()
        for i in 0..<(Self.warmups + Self.iterations) {
            let result = try synthesizeOnce(input, weights: weights, cache: cache)
            series.thermalStates.append(thermalStateName())
            series.bucketSeconds = result.bucketSeconds
            series.durationCacheKey = result.durationModelCacheKey
            if i >= Self.warmups {
                series.wallTimes.append(result.wallTimeSeconds)
                series.stageRows.append(result.timings)
            }
            await log("coreml \(key) policy=\(policyName) iter \(i): \(String(format: "%.3f", result.wallTimeSeconds))s bucket=\(result.bucketSeconds)s gen=\(String(format: "%.3f", result.timings.generatorCoreML))s")
        }
        return series
    }

    /// Failure record shared by ladder and matrix paths: NSError fields plus
    /// the last vended stage as the in-process attribution hint.
    nonisolated private func recordFailure(
        key: String, policyName: String, cache: BundleModelCache, error: Error
    ) async {
        let ns = error as NSError
        await record([
            "arm": "coreml",
            "key": key,
            "event": "policy_failure",
            "policy": policyName,
            "last_vended_stage": cache.lastVendedStage ?? "unknown",
            "error_domain": ns.domain,
            "error_code": ns.code,
            "error_description": ns.localizedDescription,
            "error_full": String(describing: error),
            "thermal_state": thermalStateName(),
        ] as [String: Any])
    }

    /// Success record shared by ladder and matrix paths.
    nonisolated private func recordSuccess(
        key: String, policyName: String, series: WarmSeries, canonicalDuration: Double
    ) async {
        let med = median(series.wallTimes)
        let stages = stageSummaries(series.stageRows)
        await record([
            "arm": "coreml",
            "key": key,
            "compute_policy": policyName,
            "warm_times_s": series.wallTimes,
            "median_s": med,
            "stage_seconds": stages.arrays,
            "stage_medians_s": stages.medians,
            "thermal_states": series.thermalStates,
            "bucket_seconds": series.bucketSeconds,
            "duration_cache_key": series.durationCacheKey,
            "canonical_duration_s": canonicalDuration,
            "rtf": canonicalDuration > 0 && med > 0 ? med / canonicalDuration : 0,
            "error": NSNull(),
        ] as [String: Any])
    }

    // MARK: Ladder mode (default)

    nonisolated private func runCoreMLArm() async throws {
        await log("coreml: loading hnsf weights")
        let weights = try loadHnsfWeights()
        // Cache persists across buckets while a policy works; a bucket that
        // fails restarts lower on the ladder with a fresh cache.
        let ladder = Self.activeLadder
        var ladderIndex = 0
        var cache = BundleModelCache(policy: ladder[ladderIndex], useExactDuration: Self.exactDuration, generatorPackage: Self.generatorPackageOverride, splitPackages: Self.generatorSplitPackagesOverride)

        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var series: WarmSeries? = nil
            var failure: String? = nil

            while series == nil {
                let policy = ladder[ladderIndex]
                // First iteration triggers Core ML's on-device E5/ANE AOT
                // specialization. On the Mac bakeoff the 30s bucket spent
                // ~20 min here (README/Notes/external-bakeoff-phase2-run-log.md);
                // expect longer on A14. Silence after this line = compiler.
                await log("coreml \(key) policy=\(policy.name): loading (first load can take many minutes — ANE AOT compile)")
                do {
                    series = try await runWarmSeries(input, weights: weights, cache: cache, key: key, policyName: policy.name)
                } catch {
                    await log("coreml \(key) policy=\(policy.name) FAILED: \(error)")
                    await recordFailure(key: key, policyName: policy.name, cache: cache, error: error)
                    if ladderIndex + 1 < ladder.count {
                        ladderIndex += 1
                        cache = BundleModelCache(policy: ladder[ladderIndex], useExactDuration: Self.exactDuration, generatorPackage: Self.generatorPackageOverride, splitPackages: Self.generatorSplitPackagesOverride)
                    } else {
                        failure = String(describing: error)
                        break
                    }
                }
            }

            let dur = input.canonical_duration_s ?? 0
            if let series {
                await recordSuccess(key: key, policyName: ladder[ladderIndex].name, series: series, canonicalDuration: dur)
            } else {
                await record([
                    "arm": "coreml",
                    "key": key,
                    "compute_policy": ladder[ladderIndex].name,
                    "canonical_duration_s": dur,
                    "error": failure ?? "unknown",
                ] as [String: Any])
            }
        }
    }

    // MARK: Matrix mode (--mode matrix)

    /// Single-stage compute-unit flips against the staged baseline, used to
    /// attribute the iOS `.all` ANEF rejection to a specific stage and to
    /// test whether decoder-pre's ANE pin is real on the phone:
    ///   staged            — baseline
    ///   duration=ne       — duration → CPU+NE (padded or exact per flag)
    ///   f0n=ne            — F0Ntrain → CPU+NE
    ///   decoderPre=gpu    — decoder-pre ANE pin REMOVED (inverse probe)
    ///   generator=ne      — generator → CPU+NE (3s body axis 14,401 fits the
    ///                       16,384 ANE cap; 30s does not — 3s vs 30s failure
    ///                       parity tests enforcement granularity)
    ///   cpuOnly           — floor reference
    /// No ladder fallback: a failed cell IS the data point.
    nonisolated static func matrixCells(allowedKeys: [String]) -> [(policy: StagePolicy, keys: [String])] {
        let base = allowedKeys.contains("3s") ? ["3s"] : Array(allowedKeys.prefix(1))
        let generatorKeys = base + (allowedKeys.contains("30s") ? ["30s"] : [])
        func flip(_ name: String, d: MLComputeUnits = .cpuAndGPU, f: MLComputeUnits = .cpuAndGPU,
                  p: MLComputeUnits = .cpuAndNeuralEngine, g: MLComputeUnits = .cpuAndGPU) -> StagePolicy {
            StagePolicy(name: name, duration: d, f0n: f, decoderPre: p, generator: g)
        }
        return [
            (StagePolicy.staged, base),
            (flip("duration=ne", d: .cpuAndNeuralEngine), base),
            (flip("f0n=ne", f: .cpuAndNeuralEngine), base),
            (flip("decoderPre=gpu", p: .cpuAndGPU), base),
            (flip("generator=ne", g: .cpuAndNeuralEngine), generatorKeys),
            (flip("cpuOnly", d: .cpuOnly, f: .cpuOnly, p: .cpuOnly, g: .cpuOnly), base),
        ]
    }

    nonisolated private func runCoreMLMatrix() async throws {
        await log("matrix: loading hnsf weights")
        let weights = try loadHnsfWeights()
        for cell in Self.matrixCells(allowedKeys: Self.keys) {
            for key in cell.keys {
                let input = try await MainActor.run { try self.loadInput(key) }
                // Fresh cache per cell+bucket: no state leaks across cells,
                // and only one bucket's models are resident on the 4 GB phone.
                let cache = BundleModelCache(policy: cell.policy, useExactDuration: Self.exactDuration, generatorPackage: Self.generatorPackageOverride, splitPackages: Self.generatorSplitPackagesOverride)
                await log("matrix \(key) policy=\(cell.policy.name): loading (first load can take many minutes — ANE AOT compile)")
                do {
                    let series = try await runWarmSeries(input, weights: weights, cache: cache, key: key, policyName: cell.policy.name)
                    await recordSuccess(key: key, policyName: cell.policy.name, series: series, canonicalDuration: input.canonical_duration_s ?? 0)
                } catch {
                    await log("matrix \(key) policy=\(cell.policy.name) FAILED: \(error)")
                    await recordFailure(key: key, policyName: cell.policy.name, cache: cache, error: error)
                }
            }
        }
    }

    // MARK: Soak mode (--mode soak)

    /// Background-synthesis viability probe (the FreeReader spike's decisive
    /// test): synthesize the same input in a loop for --soak-seconds, PLAYING
    /// the audio through an AVAudioSession so the `audio` background mode
    /// keeps the process alive when the screen locks. Without playback the
    /// app would just be suspended on lock and the run would prove nothing
    /// about the ANE. Synthesis is paced against the playhead (~20 s bounded
    /// look-ahead), mirroring the shape of a real TTS reader, so passes keep
    /// executing while backgrounded instead of racing ahead and going idle.
    ///
    /// Policy defaults to backgroundSafe (GPU excluded — a GPU pass in the
    /// background aborts the process, which is the very thing being probed —
    /// and ANE requested only where its compile is known to succeed). Every pass logs a "SOAK:" line: wall time, x-realtime
    /// (audio/wall — higher is better, unlike the ladder records' rtf
    /// field), app state, thermal state, buffered seconds. Lifecycle and
    /// screen-lock transitions log as "SOAK: >>>" lines. Verdict: passes
    /// with app=background and no crash for 2+ locked minutes.
    nonisolated private func runSoak() async throws {
        let policyName = Self.policyOverride ?? StagePolicy.backgroundSafe.name
        guard let policy = StagePolicy.named[policyName] else {
            fatalError("--policy \(policyName) unknown; known: \(StagePolicy.named.keys.sorted().joined(separator: ","))")
        }
        // 15s (the --keys default's nearest bucket to FreeReader's chunk
        // size) unless --keys picks another bucket.
        let key = Self.soakKey
        let weights = try loadHnsfWeights()
        let input = try await MainActor.run { try self.loadInput(key) }
        let cache = BundleModelCache(policy: policy, useExactDuration: Self.exactDuration, generatorPackage: Self.generatorPackageOverride, splitPackages: Self.generatorSplitPackagesOverride)
        let player = SoakPlayer()
        let csv = SoakCSVLogger()
        Self.seedLaunchArgsFileIfMissing()
        try await MainActor.run {
            UIDevice.current.isBatteryMonitoringEnabled = true
            SoakPlayer.logLifecycleTransitions { label in
                let battery = batterySnapshot()
                csv?.event(label, batteryPct: battery.pct, batteryState: battery.state)
            }
            try player.start()
        }
        if let csv {
            csv.event("start policy=\(policyName) key=\(key) soak_seconds=\(Int(Self.soakSeconds))")
            await log("soak CSV: \(csv.url.lastPathComponent) (Files app → KokoroIPhoneBench)")
        } else {
            await log("soak CSV logger FAILED to open — console/JSON only")
        }

        await log("soak \(key) policy=\(policyName) for \(Int(Self.soakSeconds))s — first pass includes ANE AOT compile (can take minutes); lock the screen once passes are flowing")
        let start = Date()
        var pass = 0
        while Date().timeIntervalSince(start) < Self.soakSeconds {
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try synthesizeOnce(input, weights: weights, cache: cache)
            let wall = CFAbsoluteTimeGetCurrent() - t0
            pass += 1
            let (appState, batteryPct, batteryState) = await MainActor.run {
                () -> (String, Double, String) in
                let state: String
                switch UIApplication.shared.applicationState {
                case .active: state = "active"
                case .inactive: state = "inactive"
                case .background: state = "background"
                @unknown default: state = "unknown"
                }
                let battery = batterySnapshot()
                return (state, battery.pct, battery.state)
            }
            await MainActor.run { player.schedule(result.audio) }
            let buffered = await MainActor.run { player.bufferedAhead }
            print("SOAK: pass \(pass) wall=\(String(format: "%.2f", wall))s audio=\(String(format: "%.1f", result.audioDurationSeconds))s x-realtime=\(String(format: "%.2f", wall > 0 ? result.audioDurationSeconds / wall : 0)) app=\(appState) thermal=\(thermalStateName()) buffered=\(String(format: "%.1f", buffered))s battery=\(String(format: "%.0f", batteryPct))%/\(batteryState) footprint=\(String(format: "%.0f", physFootprintMB()))MB")
            csv?.pass(pass, wall: wall, audio: result.audioDurationSeconds,
                      appState: appState, buffered: buffered,
                      batteryPct: batteryPct, batteryState: batteryState)
            await record([
                "arm": "soak",
                "key": key,
                "compute_policy": policyName,
                "pass": pass,
                "wall_s": wall,
                "audio_s": result.audioDurationSeconds,
                "app_state": appState,
                "thermal_state": thermalStateName(),
                "buffered_s": buffered,
                "battery_pct": batteryPct,
                "battery_state": batteryState,
                "footprint_mb": physFootprintMB(),
                "elapsed_s": Date().timeIntervalSince(start),
            ] as [String: Any])
            // Bounded look-ahead: don't synthesize more than ~20 s past the
            // playhead. Deadline-capped so a stalled player can't hang the run.
            let waitStart = Date()
            while await MainActor.run(body: { player.bufferedAhead }) > 20,
                  Date().timeIntervalSince(waitStart) < 60,
                  Date().timeIntervalSince(start) < Self.soakSeconds {
                try await Task.sleep(nanoseconds: 500_000_000)
            }
        }
        let finalBattery = await MainActor.run { batterySnapshot() }
        csv?.event("done passes=\(pass)", batteryPct: finalBattery.pct,
                   batteryState: finalBattery.state)
        await log("soak complete: \(pass) passes in \(Int(Date().timeIntervalSince(start)))s — if the screen stayed locked 2+ min with app=background passes, background synthesis is VIABLE")
    }

    /// A Springboard launch carries no process arguments, so the first soak
    /// run records its configuration to Documents/launch_args.txt; a later
    /// home-screen launch (untethered soak — no debugger, no charging
    /// confound) reads it back via `effectiveArgs` and repeats the same
    /// soak. Never overwritten once present: edit or delete it in the Files
    /// app to change untethered behavior.
    nonisolated private static func seedLaunchArgsFileIfMissing() {
        guard !FileManager.default.fileExists(atPath: launchArgsFileURL.path) else { return }
        var line = "--mode soak --keys \(soakKey) --soak-seconds \(Int(soakSeconds)) --out \(outName)"
        if let policy = policyOverride { line += " --policy \(policy)" }
        if exactDuration { line += " --exact-duration 1" }
        try? line.write(to: launchArgsFileURL, atomically: true, encoding: .utf8)
    }

    // MARK: G2P-only mode (--mode g2p)

    /// Times ONLY the Misaki G2P pass of the MLX arm for each bench input.
    ///
    /// Purpose: the MLX arm's `generateAudio` timings include G2P (raw-text
    /// API) while the Core ML arm starts from pre-tokenized IDs. This mode
    /// measures that boundary asymmetry per input so published comparisons
    /// can subtract it numerically instead of calling it "small but nonzero".
    /// Uses the same warmup/iteration discipline as the arms; first warmup
    /// absorbs Misaki dictionary loading.
    nonisolated private func runG2PMode() async throws {
        // KokoroTTS init loads the full 327 MB weight set even though only
        // g2pProcessor is used — acceptable: this mode reuses the vendored
        // init as-is rather than forking it (SIMPLER IS BETTER).
        await log("g2p: constructing KokoroTTS (loads full weights; G2P init is what we time)")
        let modelURL = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
        let tts = KokoroTTS(modelPath: modelURL)
        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var times: [Double] = []
            var phonemeCount = 0
            var failure: String? = nil
            for i in 0..<(Self.warmups + Self.iterations) {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let phonemes = try tts.phonemizeOnlyForBench(language: .enUS, text: input.text)
                    let t1 = CFAbsoluteTimeGetCurrent()
                    phonemeCount = phonemes.count
                    if i >= Self.warmups { times.append(t1 - t0) }
                    await log("g2p \(key) iter \(i): \(String(format: "%.4f", t1 - t0))s phonemes=\(phonemes.count)")
                } catch {
                    failure = String(describing: error)
                    await log("g2p \(key) iter \(i) FAILED: \(error)")
                    break
                }
            }
            await record([
                "arm": "g2p",
                "key": key,
                "warm_times_s": times,
                "median_s": median(times),
                "phoneme_count": phonemeCount,
                "text_chars": input.text.count,
                "thermal_state": thermalStateName(),
                "error": failure ?? NSNull(),
            ] as [String: Any])
        }
    }

    // MARK: Compute-plan mode (--mode computeplan --model <name>)

    /// Dumps per-op preferred-device counts for one bundled `.mlmodelc` via
    /// `MLComputePlan` (iOS 17.4+; this app's deployment target is 18.0, see
    /// project.yml). Xcode's Performance tab only shows an *estimate* — this
    /// is how the owner verifies real on-device residency for decoder-pre
    /// (Phase 0) and, later, the ANE generator (see
    /// README/Plans/ane-generator-a14-v1.md T5 and
    /// README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md's
    /// Python equivalent).
    nonisolated private func runComputePlanMode() async throws {
        guard let name = Self.modelOverride else {
            await log("computeplan: --model <name> is required (bundled .mlmodelc name, no extension)")
            await record(["arm": "computeplan", "key": "mode-level", "error": "--model not provided"])
            return
        }
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            await log("computeplan: \(name).mlmodelc not found in bundle")
            await record(["arm": "computeplan", "key": name, "error": "\(name).mlmodelc not found in bundle"])
            return
        }
        await log("computeplan \(name): loading compute plan (.cpuAndNeuralEngine)")
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let plan = try await MLComputePlan.load(contentsOf: url, configuration: config)

        guard case .program(let program) = plan.modelStructure, let mainFunction = program.functions["main"] else {
            await log("computeplan \(name): not an ML Program — no per-op device usage available")
            await record(["arm": "computeplan", "key": name, "error": "model is not an ML Program"])
            return
        }

        var counts: [String: Int] = [:]
        var total = 0
        for op in mainFunction.block.operations {
            total += 1
            let label = Self.deviceLabel(plan.deviceUsage(for: op)?.preferred)
            counts[label, default: 0] += 1
        }
        for (device, n) in counts.sorted(by: { $0.key < $1.key }) {
            let line = "COMPUTEPLAN: \(device): \(n) ops"
            print(line)
            await log(line)
        }
        let aneCount = counts["neuralEngine"] ?? 0
        let anePercent = total > 0 ? Double(aneCount) / Double(total) * 100 : 0
        await log("computeplan \(name): \(aneCount)/\(total) ops on ANE (\(String(format: "%.1f", anePercent))%)")
        await record([
            "arm": "computeplan",
            "key": name,
            "total_ops": total,
            "device_counts": counts,
            "ane_ops": aneCount,
            "ane_percent": anePercent,
        ] as [String: Any])
    }

    /// `MLComputeDevice` is `CustomStringConvertible` but its description is
    /// the underlying device-object description (verbose, host-specific);
    /// this gives the stable short label the COMPUTEPLAN: lines and results
    /// JSON use instead.
    nonisolated private static func deviceLabel(_ device: MLComputeDevice?) -> String {
        guard let device else { return "unknown" }
        switch device {
        case .cpu: return "cpu"
        case .gpu: return "gpu"
        case .neuralEngine: return "neuralEngine"
        @unknown default: return "unknown"
        }
    }

    // MARK: MLX arm

    nonisolated private func runMLXArm() async throws {
        // Cap MLX's GPU buffer cache so warm iterations don't accumulate
        // freed buffers — on the 4 GB iPhone 12 Pro the uncapped cache plus
        // resident Core ML models jetsammed the process (signal 9).
        MLX.GPU.set(cacheLimit: 256 * 1024 * 1024)
        await log("mlx: loading kokoro-v1_0.safetensors")
        let modelURL = Bundle.main.url(forResource: "kokoro-v1_0", withExtension: "safetensors")!
        let voicesURL = Bundle.main.url(forResource: "voices", withExtension: "npz")!
        let tts = KokoroTTS(modelPath: modelURL)
        let voices = NpyzReader.read(fileFromPath: voicesURL) ?? [:]
        guard let voice = voices["af_heart.npy"] else {
            throw NSError(domain: "bench", code: 1, userInfo: [NSLocalizedDescriptionKey: "af_heart voice missing from voices.npz"])
        }

        for key in Self.keys {
            let input = try await MainActor.run { try self.loadInput(key) }
            var times: [Double] = []
            var audioSeconds = 0.0
            var failure: String? = nil
            for i in 0..<(Self.warmups + Self.iterations) {
                do {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let (audio, _) = try tts.generateAudio(
                        voice: voice, language: .enUS, text: input.text, speed: input.speed
                    )
                    let t1 = CFAbsoluteTimeGetCurrent()
                    audioSeconds = Double(audio.count) / 24000.0
                    if i >= Self.warmups { times.append(t1 - t0) }
                    await log("mlx \(key) iter \(i): \(String(format: "%.3f", t1 - t0))s audio=\(String(format: "%.1f", audioSeconds))s")
                } catch {
                    failure = String(describing: error)
                    await log("mlx \(key) iter \(i) FAILED: \(error)")
                    break
                }
            }
            let med = median(times)
            await record([
                "arm": "mlx",
                "key": key,
                "warm_times_s": times,
                "median_s": med,
                "observed_audio_s": audioSeconds,
                "rtf": audioSeconds > 0 && med > 0 ? med / audioSeconds : 0,
                "error": failure ?? NSNull(),
            ] as [String: Any])
        }
    }
}

// MARK: - Soak playback

/// Plays synthesized 24 kHz mono PCM through an AVAudioEngine. The active
/// .playback session plus the UIBackgroundModes=audio entry in Info.plist is
/// what keeps the process running (not suspended) while the screen is locked
/// — the precondition for soak mode to probe background synthesis at all.
final class SoakPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let format = AVAudioFormat(
        commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false
    )!
    private var scheduledSeconds: Double = 0

    func start() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .spokenAudio)
        try session.setActive(true)
        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()
    }

    func schedule(_ samples: [Float]) {
        guard !samples.isEmpty, let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)
        ) else { return }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        player.scheduleBuffer(buffer)
        scheduledSeconds += Double(samples.count) / format.sampleRate
    }

    /// Seconds of scheduled audio the playhead hasn't consumed yet.
    var bufferedAhead: Double {
        guard let nodeTime = player.lastRenderTime,
              let playerTime = player.playerTime(forNodeTime: nodeTime) else {
            return 0
        }
        return scheduledSeconds - Double(playerTime.sampleTime) / playerTime.sampleRate
    }

    /// Log app-lifecycle and screen-lock transitions so the SOAK pass lines
    /// can be read against when the lock actually happened.
    /// protectedDataWillBecomeUnavailable ≈ screen locked (passcode devices).
    /// `onEvent` additionally receives each label on the main actor — soak
    /// mode uses it to land the transitions in the CSV, where they are the
    /// only lock/unlock record an untethered run has.
    static func logLifecycleTransitions(onEvent: (@MainActor (String) -> Void)? = nil) {
        let transitions: [(Notification.Name, String)] = [
            (UIApplication.willResignActiveNotification, "willResignActive"),
            (UIApplication.didEnterBackgroundNotification, "didEnterBackground (Metal now banned)"),
            (UIApplication.willEnterForegroundNotification, "willEnterForeground"),
            (UIApplication.didBecomeActiveNotification, "didBecomeActive"),
            (UIApplication.protectedDataWillBecomeUnavailableNotification, "screen LOCKED"),
            (UIApplication.protectedDataDidBecomeAvailableNotification, "screen UNLOCKED"),
        ]
        for (name, label) in transitions {
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { _ in
                print("SOAK: >>> \(label)")
                if let onEvent {
                    Task { @MainActor in onEvent(label) }
                }
            }
        }
    }
}

// MARK: - App shell

@main
struct KokoroIPhoneBenchApp: App {
    @StateObject private var runner = BenchRunner()

    var body: some Scene {
        WindowGroup {
            VStack(spacing: 16) {
                Text("Kokoro iPhone Bench").font(.headline)
                Text(runner.status).font(.caption).multilineTextAlignment(.center)
            }
            .padding()
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                runner.run()
            }
        }
    }
}
