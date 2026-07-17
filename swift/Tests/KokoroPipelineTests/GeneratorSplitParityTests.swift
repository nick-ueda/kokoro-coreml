/// Parity test for the T7 rate-boundary split generator against the monolithic
/// ln-lowered generator, on identical inputs, predicting under `CPU_AND_GPU`.
///
/// This is the Swift analogue of `scripts/verify_decoder_har_ane_split.py`: it
/// proves the executor's split WIRING is numerically correct — that chaining
/// `kokoro_decoder_har_ane_ln_trunk_3s` → `..._body_3s` reproduces the
/// monolithic `kokoro_decoder_har_ane_ln_3s`'s `spec`/`phase` (and hence its
/// host-iSTFT waveform). It calls the same `predictGeneratorSplit` helper the
/// executor uses, so the seam contract (`x_pre`/`ref_s`/`har` → `trunk` → the
/// body) is asserted in exactly the spot the pipeline relies on.
///
/// ## Why CPU_AND_GPU, never CPU_AND_NE
/// This Mac's ANE ADMITS the decoder-har-ane graph and then miscomputes it into
/// non-finite output (README/Notes/ane-generator-coreml-export-2026-07-17.md,
/// README/Plans/ane-generator-a14-v1.md T4/T6). `CPU_AND_GPU` evaluates the
/// graph faithfully; `CPU_AND_NE` here would measure the Mac's ANE, not the
/// wiring. The A14's ANE is a different generation — whether it executes the
/// split correctly is strictly the owner's device test.
///
/// ## Tolerance
/// Split vs monolithic differ only by the extra fp32 round-trip at the
/// materialized `trunk` seam (internal fp16 in the monolithic graph); the math
/// is otherwise identical (PyTorch fp32 has them bit-identical at 1.71e-6, T6).
/// The gate is the established Core ML waveform bar — SNR ≥ 40 dB — which a
/// correct chaining clears by a wide margin while a mis-wired seam (wrong input
/// name, wrong order) fails catastrophically.
///
/// ## Real inputs, not random
/// Inputs are the REAL x_pre/ref_s/har fixture from
/// `scripts/dump_generator_split_inputs.py` (the same tensors the Python
/// verifiers validate). Random inputs overflow fp16 in `spec = exp(conv_post)`
/// — every AdaIN block layer-normalizes activations, so their magnitude tracks
/// the learned weights and voice, not the input scale — which would make the
/// finiteness check spurious. See that script's docstring.

import Accelerate
import CoreML
import XCTest
@testable import KokoroPipeline

final class GeneratorSplitParityTests: XCTestCase {

    private static let monolithicPackage = "kokoro_decoder_har_ane_ln_3s"
    private static let trunkPackage = "kokoro_decoder_har_ane_ln_trunk_3s"
    private static let bodyPackage = "kokoro_decoder_har_ane_ln_body_3s"

    /// Repo-root `coreml/` directory, located from this source file's path so
    /// the test is CWD-independent. (`swift test` runs on macOS only, so a
    /// filesystem path is fine — these packages are not bundled with the test
    /// target.)
    private static var coremlDir: URL {
        URL(fileURLWithPath: #filePath)          // …/swift/Tests/KokoroPipelineTests/<this>.swift
            .deletingLastPathComponent()          // …/swift/Tests/KokoroPipelineTests
            .deletingLastPathComponent()          // …/swift/Tests
            .deletingLastPathComponent()          // …/swift
            .deletingLastPathComponent()          // …/<repo root>
            .appendingPathComponent("coreml")
    }

    /// Compile + load a bundled `.mlpackage` under `CPU_AND_GPU`, or skip if the
    /// package is absent (e.g. a checkout without the T6 exports).
    private func loadModel(_ name: String) throws -> MLModel {
        let url = Self.coremlDir.appendingPathComponent("\(name).mlpackage")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Missing \(name).mlpackage under \(Self.coremlDir.path) — export via `--mode decoder-har-ane-split`")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU
        let compiled = try MLModel.compileModel(at: url)
        return try MLModel(contentsOf: compiled, configuration: config)
    }

    /// Load a real-input `.f32` fixture (flat little-endian float32, no header)
    /// into an `MLMultiArray` of the given shape.
    private func loadInput(_ name: String, shape: [Int]) throws -> MLMultiArray {
        guard let url = Bundle.module.url(
            forResource: name,
            withExtension: "f32",
            subdirectory: "Fixtures/generator_split"
        ) else {
            throw XCTSkip("Missing \(name).f32 — run scripts/dump_generator_split_inputs.py")
        }
        let data = try Data(contentsOf: url)
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw XCTSkip("\(name).f32 has \(values.count) floats, expected \(expected) for shape \(shape) — regenerate fixture")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<expected { ptr[i] = values[i] }
        return array
    }

    private func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var maxAbs: Float = 0
        vDSP_maxmgv(diff, 1, &maxAbs, vDSP_Length(a.count))
        return maxAbs
    }

    /// SNR in dB of `test` against `reference`: 20·log10(‖ref‖ / ‖ref − test‖).
    /// `+inf` when identical.
    private func snrDecibels(reference: [Float], test: [Float]) -> Double {
        precondition(reference.count == test.count)
        var refEnergy: Float = 0
        vDSP_svesq(reference, 1, &refEnergy, vDSP_Length(reference.count))
        var diff = [Float](repeating: 0, count: reference.count)
        vDSP_vsub(test, 1, reference, 1, &diff, 1, vDSP_Length(reference.count))
        var diffEnergy: Float = 0
        vDSP_svesq(diff, 1, &diffEnergy, vDSP_Length(diff.count))
        if diffEnergy == 0 { return .infinity }
        return 10.0 * log10(Double(refEnergy) / Double(diffEnergy))
    }

    private func allFinite(_ values: [Float]) -> Bool {
        values.allSatisfy { $0.isFinite }
    }

    /// The split chain reproduces the monolithic generator's `spec`/`phase` and
    /// waveform on identical inputs, under CPU_AND_GPU.
    func testSplitMatchesMonolithicUnderCPUAndGPU() throws {
        let monolithic = try loadModel(Self.monolithicPackage)
        let trunk = try loadModel(Self.trunkPackage)
        let body = try loadModel(Self.bodyPackage)

        // Real inputs at Ground Truth geometry (README/Plans/ane-generator-a14-v1.md T4/T6).
        let xPre = try loadInput("x_pre", shape: [1, 512, 240])
        let refS = try loadInput("ref_s", shape: [1, 256])
        let har = try loadInput("har", shape: [1, 22, 14401])

        // Monolithic path: one predict → spec/phase.
        let monoInput = try MLDictionaryFeatureProvider(dictionary: [
            "x_pre": MLFeatureValue(multiArray: xPre),
            "ref_s": MLFeatureValue(multiArray: refS),
            "har": MLFeatureValue(multiArray: har),
        ])
        let monoOutput = try monolithic.prediction(from: monoInput)
        let monoSpec = floatValues(from: monoOutput.featureValue(for: "spec")!.multiArrayValue!)
        let monoPhase = floatValues(from: monoOutput.featureValue(for: "phase")!.multiArrayValue!)

        // Split path: the exact helper the executor calls (trunk → body).
        let (splitOutput, trunkSeam) = try predictGeneratorSplit(
            trunk: trunk, body: body, xPre: xPre, refS: refS, har: har
        )
        let splitSpec = floatValues(from: splitOutput.featureValue(for: "spec")!.multiArrayValue!)
        let splitPhase = floatValues(from: splitOutput.featureValue(for: "phase")!.multiArrayValue!)
        let trunkValues = floatValues(from: trunkSeam)

        // Everything finite under CPU_AND_GPU (the graph is correct here — the
        // non-finite result is a CPU_AND_NE-only, this-Mac's-ANE artifact).
        XCTAssertEqual(trunkSeam.shape.map { $0.intValue }, [1, 256, 2400], "trunk seam geometry")
        XCTAssertTrue(allFinite(trunkValues), "trunk seam has non-finite values under CPU_AND_GPU")
        XCTAssertTrue(allFinite(monoSpec) && allFinite(monoPhase), "monolithic spec/phase non-finite under CPU_AND_GPU")
        XCTAssertTrue(allFinite(splitSpec) && allFinite(splitPhase), "split spec/phase non-finite under CPU_AND_GPU")

        // spec/phase parity (reported; exp() amplifies ulps, so the hard gate is
        // on the waveform below — mirrors the Python split verifier).
        let specMaxAbs = maxAbsDifference(splitSpec, monoSpec)
        let phaseMaxAbs = maxAbsDifference(splitPhase, monoPhase)

        // Waveform parity through the shared host iSTFT (the generator's product
        // and the quantity the 40 dB Core ML gate judges).
        let frameCount = 14401
        let monoWave = hostISTFTInverse(spec: monoSpec, phase: monoPhase, frameCount: frameCount)
        let splitWave = hostISTFTInverse(spec: splitSpec, phase: splitPhase, frameCount: frameCount)
        let waveMaxAbs = maxAbsDifference(splitWave, monoWave)
        let waveSNR = snrDecibels(reference: monoWave, test: splitWave)

        print("SPLITPARITY: spec max-abs=\(specMaxAbs) phase max-abs=\(phaseMaxAbs) " +
              "waveform max-abs=\(waveMaxAbs) waveform SNR=\(String(format: "%.2f", waveSNR)) dB")

        XCTAssertEqual(splitWave.count, monoWave.count, "split/monolithic waveform length mismatch")
        XCTAssertGreaterThanOrEqual(
            waveSNR, 40.0,
            "split-vs-monolithic waveform SNR \(String(format: "%.2f", waveSNR)) dB below the 40 dB gate — split wiring is wrong"
        )
    }
}
