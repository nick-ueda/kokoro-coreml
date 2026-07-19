/// Regression tests for `windowedSplitBucket` (post-T9 warm-path fix).
///
/// The bug: `warmModels` requested the split generator at the raw planning
/// bucket (`probe.bucketSec`), so warming an `aneGeneratorSplit` run at a > 3 s
/// bucket threw the provider's 3 s-only split guard (BenchApp
/// `generatorSplitModels`) BEFORE synthesis ran — even though the windowed
/// synthesis path (Stage 8/9) runs the split at a fixed 3 s. Both paths now
/// resolve the split bucket through `windowedSplitBucket`; these tests lock its
/// contract and the guard-clearing invariant.

import CoreML
import XCTest
@testable import KokoroPipeline

final class WindowedSplitBucketTests: XCTestCase {

    /// A chunk longer than one 3 s window always vocodes on the fixed 3 s split,
    /// regardless of the (larger) bucket stages 1-7 planned at.
    func testOverflowingChunkAlwaysUsesThreeSecondSplit() {
        // fullF0Len: 240 = 3 s, 560 = 7 s, 1200 = 15 s, 2400 = 30 s.
        XCTAssertEqual(windowedSplitBucket(fullF0Len: 560, planningBucketSec: 7), 3)
        XCTAssertEqual(windowedSplitBucket(fullF0Len: 1200, planningBucketSec: 15), 3)
        XCTAssertEqual(windowedSplitBucket(fullF0Len: 2400, planningBucketSec: 30), 3)
        // Just over the window boundary still windows.
        XCTAssertEqual(
            windowedSplitBucket(fullF0Len: WindowedVocodeConstants.winASR + 1, planningBucketSec: 7),
            3
        )
    }

    /// A chunk that fits in one window runs the single-predict split at its own
    /// bucket (the T7 path) — here that bucket is always the 3 s window itself.
    func testSingleWindowChunkUsesItsOwnBucket() {
        XCTAssertEqual(
            windowedSplitBucket(fullF0Len: WindowedVocodeConstants.winASR, planningBucketSec: 3),
            3
        )
        XCTAssertEqual(windowedSplitBucket(fullF0Len: 120, planningBucketSec: 3), 3)
    }

    /// The exact warm-up regression: the bucket the fix resolves for a > 3 s
    /// chunk clears an `aneGeneratorSplit`-style 3 s-only split guard, whereas
    /// the pre-fix raw planning bucket trips it (throwing before synthesis runs).
    func testChosenBucketClearsThreeSecondOnlySplitGuard() throws {
        let provider = ThreeSecondOnlySplitProvider()
        // Pre-fix behavior asked for the split at the planning bucket — throws.
        XCTAssertThrowsError(
            try provider.generatorSplitModels(bucketSec: 15),
            "a > 3 s split request must trip the 3 s-only guard"
        )
        // The fix asks at windowedSplitBucket(...) == 3, which the guard accepts.
        let chosen = windowedSplitBucket(fullF0Len: 1200, planningBucketSec: 15)
        XCTAssertEqual(chosen, 3)
        XCTAssertNoThrow(
            try provider.generatorSplitModels(bucketSec: chosen),
            "warm-up must request the windowed split at a bucket the provider vends"
        )
    }

    /// Mirrors BenchApp's `aneGeneratorSplit` provider: the split exists at the
    /// 3 s bucket alone and throws for any other. Only `generatorSplitModels` is
    /// exercised here, and the guard fires before any model would be
    /// constructed, so the 3 s case returns `nil` (no real `.mlmodelc` needed —
    /// this test asserts guard throw-vs-accept, not the returned pair).
    private struct ThreeSecondOnlySplitProvider: KokoroModelProvider {
        enum Failure: Error { case unsupportedBucket(Int), notExercised }
        func durationModelChoices() -> [DurationModelChoice] { [] }
        func availableBucketSeconds() -> [Int] { [3, 7, 15, 30] }
        func durationModel(choice: DurationModelChoice) throws -> MLModel { throw Failure.notExercised }
        func f0ntrainModel(tFrames: Int) throws -> MLModel { throw Failure.notExercised }
        func decoderPreModel(bucketSec: Int) throws -> MLModel { throw Failure.notExercised }
        func generatorModel(bucketSec: Int) throws -> MLModel { throw Failure.notExercised }
        func generatorSplitModels(bucketSec: Int) throws -> GeneratorSplitModels? {
            guard bucketSec == WindowedVocodeConstants.windowBucketSec else {
                throw Failure.unsupportedBucket(bucketSec)
            }
            return nil
        }
    }
}
