/// End-to-end integration test for the windowed generator (task T9): runs
/// the REAL T6/T7 split `.mlpackage` pair (`kokoro_decoder_har_ane_ln_{trunk,
/// body}_3s`) through `vocodeWindowed` on the golden fixture's full-chunk
/// `x_pre`/`ref_s`/`har` (`scripts/dump_windowed_vocode_golden.py`, 600 ASR
/// frames / 3 windows) and compares the result to that script's `wav_w.f32`.
///
/// This is the Swift analogue of `GeneratorSplitParityTests` — the windowed
/// counterpart proving the FULL executor path (per-window slicing, split
/// predict, host iSTFT, crop/concat/trim) is wired correctly against real
/// models, not just the pure-arithmetic crop/concat logic
/// `WindowedVocodeGoldenTests` isolates.
///
/// ## Why SNR, not `WindowedVocodeGoldenTests`' tight max-abs
///
/// `wav_w.f32` is computed by pure fp32 PyTorch (`GeneratorFromHarANE`, no
/// CoreML); this test runs the ACTUAL exported fp16-precision `.mlpackage`s.
/// That is the same PyTorch-vs-CoreML boundary T4/T6/T7 already
/// characterized at ~40-46 dB SNR under `CPU_AND_GPU` (never `CPU_AND_NE` —
/// see `GeneratorSplitParityTests`' header for why: this Mac's ANE admits the
/// graph and miscomputes it into non-finite output, a separate,
/// already-documented axis). Windowing chains multiple such predicts, so the
/// gate here is deliberately looser than the single-predict 40 dB bar.
///
/// ## Real inputs, not random
///
/// Same discipline as `GeneratorSplitParityTests`: `spec = exp(conv_post)`
/// overflows fp16 on random input because every AdaIN block layer-normalizes
/// activations to the LEARNED weights' scale, not the input's. The fixture
/// is real speech-distributed `x_pre`/`har` from the golden dump script.

import Accelerate
import CoreML
import XCTest
@testable import KokoroPipeline

final class WindowedGeneratorSplitIntegrationTests: XCTestCase {

    private static let trunkPackage = "kokoro_decoder_har_ane_ln_trunk_3s"
    private static let bodyPackage = "kokoro_decoder_har_ane_ln_body_3s"

    /// Measured 46.31 dB on this fixture (3 windows, 600 ASR frames) —
    /// chaining 3 split predicts through independent windows lands almost
    /// exactly in T7's single-predict neighborhood (48.73 dB), i.e.
    /// windowing itself costs negligible extra SNR beyond the fp16-CoreML
    /// boundary T4/T6/T7 already characterized. Gate set with headroom below
    /// the measured value: a mis-wired seam (wrong har slice offset, wrong
    /// window order, wrong core crop) collapses this to near 0 dB, not a few
    /// dB below the gate — see this file's "## Why SNR" note for why the
    /// gate isn't tighter still.
    private static let snrGateDB = 35.0

    private static var coremlDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()          // …/swift/Tests/KokoroPipelineTests
            .deletingLastPathComponent()          // …/swift/Tests
            .deletingLastPathComponent()          // …/swift
            .deletingLastPathComponent()          // …/<repo root>
            .appendingPathComponent("coreml")
    }

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

    private func loadFloatArray(_ resource: String) throws -> [Float] {
        guard let url = Bundle.module.url(
            forResource: resource,
            withExtension: "f32",
            subdirectory: "Fixtures/windowed_vocode"
        ) else {
            throw XCTSkip("Missing fixture \(resource).f32 — run scripts/dump_windowed_vocode_golden.py")
        }
        let data = try Data(contentsOf: url)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadInput(_ name: String, shape: [Int]) throws -> MLMultiArray {
        let values = try loadFloatArray(name)
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw XCTSkip("\(name).f32 has \(values.count) floats, expected \(expected) for shape \(shape) — regenerate fixture")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<expected { ptr[i] = values[i] }
        return array
    }

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

    func testWindowedExecutorMatchesGoldenWaveformUnderCPUAndGPU() throws {
        let trunk = try loadModel(Self.trunkPackage)
        let body = try loadModel(Self.bodyPackage)

        let asrLen = 600 // must match scripts/dump_windowed_vocode_golden.py's fixture geometry
        let xPre = try loadInput("x_pre", shape: [1, 512, asrLen])
        let refS = try loadInput("ref_s", shape: [1, 256])
        let harChannels = HarmonicConstants.harChannels
        let harBodyFrames = WindowedVocodeConstants.bodyPerASR * asrLen + 1
        let harFlat = try loadFloatArray("har")
        XCTAssertEqual(harFlat.count, harChannels * harBodyFrames, "har fixture geometry drift")

        let expected = try loadFloatArray("wav_w")

        let result = try vocodeWindowed(
            xPre: xPre,
            refS: refS,
            harFlat: harFlat,
            harFrames: harBodyFrames,
            asrLen: asrLen,
            trunk: trunk,
            body: body
        )

        XCTAssertEqual(result.windowCount, 3)
        XCTAssertEqual(result.nonFinite, 0, "windowed generator produced non-finite output under CPU_AND_GPU")
        XCTAssertEqual(result.waveform.count, expected.count)

        let snr = snrDecibels(reference: expected, test: result.waveform)
        print("WINDOWEDPARITY: windows=\(result.windowCount) waveform SNR=\(String(format: "%.2f", snr)) dB")
        XCTAssertGreaterThanOrEqual(
            snr, Self.snrGateDB,
            "windowed executor vs golden fp32 waveform SNR \(String(format: "%.2f", snr)) dB below the \(Self.snrGateDB) dB gate"
        )
    }
}
