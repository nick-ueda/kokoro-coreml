/// Tests for `windowedVocodePlan` (task T9) — pure geometry, no model or
/// tensor dependency. See `WindowedGeneratorExecutor.swift`'s header for the
/// edge-overlap policy rationale (the one deviation from
/// `scripts/probe_windowed_vocode.py`'s dynamic-shape-only window widths).

import XCTest
@testable import KokoroPipeline

final class WindowedVocodePlanTests: XCTestCase {

    /// Meta describing the golden fixture's window plan, dumped by
    /// `scripts/dump_windowed_vocode_golden.py` alongside the waveform
    /// fixtures `WindowedVocodeGoldenTests` loads.
    private struct GoldenMeta: Decodable {
        struct Window: Decodable, Equatable {
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

    /// Every window is fixed-width (`WindowedVocodeConstants.winASR`) once
    /// windowing is active — the whole point of the edge-overlap policy is
    /// that the exported `.mlpackage`'s x_pre/har axes never see a short
    /// window.
    private func assertFixedWidth(_ windows: [GeneratorWindow], file: StaticString = #filePath, line: UInt = #line) {
        for (i, w) in windows.enumerated() {
            XCTAssertEqual(
                w.hi - w.lo, WindowedVocodeConstants.winASR,
                "window \(i) width \(w.hi - w.lo) != \(WindowedVocodeConstants.winASR)",
                file: file, line: line
            )
            XCTAssertLessThanOrEqual(w.lo, w.coreLo, "window \(i): lo must not exceed coreLo", file: file, line: line)
            XCTAssertLessThanOrEqual(w.coreHi, w.hi, "window \(i): coreHi must not exceed hi", file: file, line: line)
        }
    }

    /// Consecutive windows' core regions tile the utterance exactly: no gap,
    /// no overlap, starting at 0 and ending at `asrLen`.
    private func assertCoresTile(_ windows: [GeneratorWindow], asrLen: Int, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(windows.first?.coreLo, 0, file: file, line: line)
        XCTAssertEqual(windows.last?.coreHi, asrLen, file: file, line: line)
        for i in 1..<windows.count {
            XCTAssertEqual(windows[i - 1].coreHi, windows[i].coreLo, "gap/overlap between windows \(i - 1) and \(i)", file: file, line: line)
        }
    }

    /// Minimal windowed case: `asrLen` just above one window. Two windows,
    /// both edge-shifted (first has no left halo to spare, last has no right
    /// halo to spare) — the tightest case where the edge-overlap policy's
    /// "at most one side clipped per window" assumption is exercised hardest.
    func testMinimalTwoWindowCase() {
        let asrLen = WindowedVocodeConstants.winASR + 1 // 241
        let windows = windowedVocodePlan(asrLen: asrLen)
        XCTAssertEqual(windows.count, 2)
        assertFixedWidth(windows)
        assertCoresTile(windows, asrLen: asrLen)
        // Hand-derived: k=0 core=[0,200) halo->[0,220), shifted to [0,240).
        XCTAssertEqual(windows[0], GeneratorWindow(lo: 0, hi: 240, coreLo: 0, coreHi: 200))
        // k=1 (last) core=[200,241) halo->[180,241), shifted to [1,241).
        XCTAssertEqual(windows[1], GeneratorWindow(lo: 1, hi: 241, coreLo: 200, coreHi: 241))
    }

    /// A partial (short) last core — `asrLen` not a multiple of
    /// `coreASR` — must still produce a full-width last window and a core
    /// span that ends exactly at `asrLen`.
    func testPartialLastCoreCase() {
        let asrLen = 850 // 5 windows; last core = [800, 850), only 50 frames
        let windows = windowedVocodePlan(asrLen: asrLen)
        XCTAssertEqual(windows.count, 5)
        assertFixedWidth(windows)
        assertCoresTile(windows, asrLen: asrLen)
        XCTAssertEqual(windows.last, GeneratorWindow(lo: 610, hi: 850, coreLo: 800, coreHi: 850))
    }

    /// A true interior window (both halos fully available) needs no
    /// edge-overlap shift — this is the majority case for any long utterance
    /// and must reproduce the validated probe's un-shifted geometry exactly.
    func testInteriorWindowNeedsNoShift() {
        let windows = windowedVocodePlan(asrLen: 1000)
        // k=2: core=[400,600), halo->[380,620) — already 240 wide, untouched.
        XCTAssertEqual(windows[2], GeneratorWindow(lo: 380, hi: 620, coreLo: 400, coreHi: 600))
    }

    /// Every window spans exactly `winASR`, and cores tile losslessly, across
    /// a spread of utterance lengths (multiples of `coreASR` and not).
    func testFixedWidthAndTilingAcrossLengths() {
        for asrLen in [241, 400, 401, 600, 799, 800, 1001, 2400] {
            let windows = windowedVocodePlan(asrLen: asrLen)
            assertFixedWidth(windows)
            assertCoresTile(windows, asrLen: asrLen)
        }
    }

    /// Cross-check against the golden fixture's independently-computed plan
    /// (`scripts/dump_windowed_vocode_golden.py`'s `_window_vocode_edge_overlap`,
    /// the PyTorch mirror of this exact policy) — not just hand-derived cases.
    func testMatchesGoldenFixturePlan() throws {
        let meta = try loadGoldenMeta()
        let windows = windowedVocodePlan(asrLen: meta.asrLen)
        XCTAssertEqual(windows.count, meta.windowCount)
        for (i, (got, want)) in zip(windows, meta.windows).enumerated() {
            XCTAssertEqual(got.lo, want.lo, "window \(i) lo")
            XCTAssertEqual(got.hi, want.hi, "window \(i) hi")
            XCTAssertEqual(got.coreLo, want.coreLo, "window \(i) coreLo")
            XCTAssertEqual(got.coreHi, want.coreHi, "window \(i) coreHi")
        }
    }
}
