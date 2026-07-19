/// Golden test for `stitchWindowedWaveforms` (task T9) against
/// `scripts/dump_windowed_vocode_golden.py`'s `wav_w.f32` — the Python mirror
/// of the SAME deployable edge-overlap policy `windowedVocodePlan` computes
/// (NOT `scripts/probe_windowed_vocode.py`'s dynamic-shape-only geometry; see
/// `WindowedGeneratorExecutor.swift`'s header).
///
/// ## Why this can hit T3's tight 1e-4 bar, unlike a real-model comparison
///
/// Feeding fp32-PyTorch inputs through the ACTUAL exported `.mlpackage`
/// trunk/body pair and comparing to a PyTorch reference would re-measure the
/// fp16-CoreML-vs-fp32-PyTorch gap T4/T6/T7 already characterized at ~40-46
/// dB SNR (see `GeneratorSplitParityTests`) — not a tight max-abs bound, and
/// not anything new. This test instead takes the dump script's own
/// PER-WINDOW waveforms (`window{k}_wav.f32`, pure PyTorch fp32, no CoreML
/// involved) as GIVEN, and exercises ONLY `windowedVocodePlan` +
/// `stitchWindowedWaveforms` — the geometry/crop/concat/trim arithmetic that
/// is T9's actual new contribution, with the generator's own numerics held
/// fixed. Both sides are then fp32 array slicing and concatenation, so a
/// T3-style tight tolerance is a meaningful bar, not a nominal one.
///
/// `WindowedGeneratorSplitIntegrationTests` (this directory) separately runs
/// the real trunk/body packages end-to-end for the SNR-gated wiring check.

import Accelerate
import XCTest
@testable import KokoroPipeline

final class WindowedVocodeGoldenTests: XCTestCase {

    private struct GoldenMeta: Decodable {
        struct Window: Decodable {
            let lo: Int
            let hi: Int
            let coreLo: Int
            let coreHi: Int

            enum CodingKeys: String, CodingKey {
                case lo, hi
                case coreLo = "core_lo"
                case coreHi = "core_hi"
            }
        }

        let asrLen: Int
        let windowCount: Int
        let windows: [Window]

        enum CodingKeys: String, CodingKey {
            case asrLen = "asr_len"
            case windowCount = "window_count"
            case windows
        }
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
        precondition(data.count % MemoryLayout<Float>.size == 0, "\(resource).f32 is not float32-aligned")
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadGoldenMeta() throws -> GoldenMeta {
        guard let url = Bundle.module.url(
            forResource: "meta",
            withExtension: "json",
            subdirectory: "Fixtures/windowed_vocode"
        ) else {
            throw XCTSkip("Missing meta.json — run scripts/dump_windowed_vocode_golden.py")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(GoldenMeta.self, from: data)
    }

    private func maxAbsDifference(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var diff = [Float](repeating: 0, count: a.count)
        vDSP_vsub(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var maxAbs: Float = 0
        vDSP_maxmgv(diff, 1, &maxAbs, vDSP_Length(a.count))
        return maxAbs
    }

    /// `windowedVocodePlan` + `stitchWindowedWaveforms`, fed the dump
    /// script's own per-window waveforms, reproduces `wav_w.f32` exactly
    /// (mirrors `HostISTFTTests`' tolerance and rationale).
    func testStitchMatchesGoldenWaveform() throws {
        let meta = try loadGoldenMeta()
        let expected = try loadFloatArray("wav_w")
        XCTAssertEqual(expected.count, WindowedVocodeConstants.samplesPerASR * meta.asrLen)

        var perWindowWaveforms: [[Float]] = []
        for k in 0..<meta.windowCount {
            perWindowWaveforms.append(try loadFloatArray("window\(k)_wav"))
        }

        let windows = windowedVocodePlan(asrLen: meta.asrLen)
        XCTAssertEqual(windows.count, meta.windowCount)

        let stitched = stitchWindowedWaveforms(windows: windows, waveforms: perWindowWaveforms, asrLen: meta.asrLen)

        XCTAssertEqual(stitched.count, expected.count)
        let maxAbs = maxAbsDifference(stitched, expected)
        XCTAssertLessThan(maxAbs, 1e-4, "windowed stitch max-abs error \(maxAbs) exceeds golden tolerance")
    }
}
