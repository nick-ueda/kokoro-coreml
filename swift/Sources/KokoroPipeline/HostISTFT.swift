/// Native Swift implementation of the Kokoro generator's inverse STFT tail.
///
/// Replaces ``kokoro/custom_stft.py`` ``CustomSTFT.inverse`` (called as
/// ``gen.stft.inverse(spec, phase)`` with no ``length`` argument — see
/// ``kokoro/istftnet.py`` ``Generator.forward`` and
/// ``export_synth/wrappers.py`` ``GeneratorFromHar.forward``). Moving this
/// step to the host lets an ANE-exported generator package stop at
/// spec/phase (see `README/Plans/ane-generator-a14-v1.md` T4) instead of
/// carrying the 72,000-sample iSTFT output past the ANE's per-axis element
/// limit.
///
/// ``HarmonicSource.swift`` ``stftTransform`` already implements the FORWARD
/// half of this same n_fft=20/hop=5 transform (analysis, via decimated
/// ``vDSP_desamp`` convolutions). This file is the synthesis half
/// (overlap-add reconstruction). The two are intentionally symmetric: the
/// forward pass loops over frequency bins (11) and decimates; the inverse
/// pass loops over kernel taps (20) and does a strided overlap-add — each
/// loop takes the axis that vDSP can vectorize over.
///
/// ## Math
///
/// PyTorch's ``inverse`` runs two ``F.conv_transpose1d`` calls (real/imag)
/// with stride=hop and subtracts them. For frame `i`, bin `c`, tap `k`
/// (0..<n_fft), conv_transpose1d scatters
/// ``real_part[c,i] * backward_real[c,k] - imag_part[c,i] * backward_imag[c,k]``
/// into output position `i*hop + k`, summed over `c` and over overlapping
/// frames. This is implemented here as one small matmul per frame-batch
/// (bins -> taps) followed by an overlap-add over taps, rather than a naive
/// O(T·n_fft) scalar loop, so the O(T·n_fft) work vDSP still does happens
/// inside vectorized calls instead of a Swift loop.
///
/// Called by:
/// - (T5, not yet wired) ``KokoroSynthesisExecutor`` for the
///   ``aneGenerator`` stage policy, after the ANE package returns spec/phase.
///
/// Reference implementation:
/// - ``kokoro/custom_stft.py:300-386`` (``CustomSTFT.inverse``)
/// - ``kokoro/istftnet.py:465-467`` (call site: ``spec = exp(...)``,
///   ``phase = sin(...)``, ``return self.stft.inverse(spec, phase)`` — no
///   `length`, so the center-pad trim is the only trim; any final
///   `[..., :bucket_samples]` slicing happens at THIS function's call site,
///   not inside it).

import Accelerate
import Foundation

/// Precomputed inverse-DFT basis for Kokoro's 20-point Hann-window iSTFT.
///
/// Mirrors the one-time buffer precomputation in ``CustomSTFT.__init__``
/// (`kokoro/custom_stft.py:166-227`). Built once and cached because
/// `hostISTFTInverse` may run once per synthesis call.
private enum HostISTFTBasis {
    /// Periodic Hann window, length n_fft. Same formula as
    /// ``HarmonicSTFTBasis.window`` in HarmonicSource.swift; duplicated
    /// (not shared) because that type is file-private and n_fft=20 makes
    /// recomputing it here a 3-line cost, not worth cross-file coupling.
    static let window: [Float] = {
        let nfft = HarmonicConstants.stftNfft
        var values = [Float](repeating: 0, count: nfft)
        for n in 0..<nfft {
            values[n] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(n) / Float(nfft)))
        }
        return values
    }()

    /// Combined backward-DFT weight, laid out as an (n_fft x 2*freqBins)
    /// row-major matrix so one `vDSP_mmul` against a stacked
    /// [real_part; imag_part] (2*freqBins x frames) input produces the
    /// per-tap overlap-add contribution (n_fft x frames) directly — the
    /// `imag` half is pre-negated so the matmul also folds in
    /// `real_rec - imag_rec` (see `CustomSTFT.inverse`), avoiding a
    /// separate vDSP_vsub pass.
    ///
    /// Row n (tap k=n, 0..<n_fft), columns 0..<freqBins = backward_real[·,n];
    /// columns freqBins..<2*freqBins = -backward_imag[·,n].
    static let combinedWeight: [Float] = {
        let nfft = HarmonicConstants.stftNfft       // 20
        let hop = HarmonicConstants.stftHop         // 5
        let freqBins = HarmonicConstants.stftFreqBins // 11

        // custom_stft.py's constant-scalar overlap-add fold is only exact
        // for a periodic Hann window when n_fft is a multiple of hop with
        // overlap factor n_fft/hop >= 3 (see the derivation in
        // kokoro/custom_stft.py:188-208). Kokoro's generator geometry
        // (20/5 = 4) satisfies this; assert it here so a future constant
        // change fails loudly instead of silently baking in ripple.
        precondition(
            nfft % hop == 0 && nfft / hop >= 3,
            "HostISTFT constant-scalar OLA fold requires n_fft a multiple of hop with overlap factor >= 3"
        )

        // One-sided spectrum scale: interior bins represent +/-k and carry
        // 2/n_fft; DC and Nyquist (n_fft even) appear once and carry
        // 1/n_fft. See custom_stft.py:169-186.
        var onesidedScale = [Float](repeating: 2.0 / Float(nfft), count: freqBins)
        onesidedScale[0] = 1.0 / Float(nfft)
        if nfft % 2 == 0 {
            onesidedScale[freqBins - 1] = 1.0 / Float(nfft)
        }

        // Overlap-add normalization: mean windowed-power envelope over one
        // hop period (custom_stft.py:209-213).
        var windowPowerSum = [Float](repeating: 0, count: hop)
        var offset = 0
        while offset < nfft {
            for j in 0..<hop {
                windowPowerSum[j] += window[offset + j] * window[offset + j]
            }
            offset += hop
        }
        let olaNorm = windowPowerSum.reduce(0, +) / Float(hop)

        var combined = [Float](repeating: 0, count: nfft * 2 * freqBins)
        let twoPiOverN = 2.0 * Float.pi / Float(nfft)
        for n in 0..<nfft {
            let rowBase = n * 2 * freqBins
            for k in 0..<freqBins {
                let invWindow = window[n] * onesidedScale[k] / olaNorm
                let angle = twoPiOverN * Float(k) * Float(n)
                let backwardReal = cos(angle) * invWindow
                let backwardImag = sin(angle) * invWindow
                combined[rowBase + k] = backwardReal
                combined[rowBase + freqBins + k] = -backwardImag
            }
        }
        return combined
    }()
}

/// Inverse STFT matching ``CustomSTFT.inverse(spec, phase)`` (no `length`).
///
/// - Parameters:
///   - spec: Magnitude spectrogram, flat (freqBins=11 x frameCount)
///     row-major (frequency-major, time-inner) — same layout as
///     ``stftTransform``'s output and the ``har`` tensor's magnitude half.
///   - phase: Phase spectrogram, same shape/layout as `spec`.
///   - frameCount: Number of STFT frames (T).
/// - Returns: Reconstructed waveform after the center-padding trim, length
///   `(frameCount - 1) * hop` when `frameCount` is large enough for the trim
///   to apply (matches `CustomSTFT.inverse`'s own guard — otherwise the
///   untrimmed `(frameCount - 1) * hop + n_fft` samples are returned
///   unchanged). No further length trim/pad is applied; the caller performs
///   any final `[..., :bucket_samples]` slice.
public func hostISTFTInverse(spec: [Float], phase: [Float], frameCount: Int) -> [Float] {
    let freqBins = HarmonicConstants.stftFreqBins // 11
    let nfft = HarmonicConstants.stftNfft         // 20
    let hop = HarmonicConstants.stftHop           // 5

    precondition(spec.count == freqBins * frameCount, "spec must be (freqBins x frameCount)")
    precondition(phase.count == freqBins * frameCount, "phase must be (freqBins x frameCount)")
    guard frameCount > 0 else { return [] }

    // real_part = spec * cos(phase); imag_part = spec * sin(phase), stacked
    // into one (2*freqBins x frameCount) buffer so the backward-DFT matmul
    // runs in a single vDSP_mmul call.
    let bins = freqBins * frameCount
    var cosPhase = [Float](repeating: 0, count: bins)
    var sinPhase = [Float](repeating: 0, count: bins)
    var trigCount = Int32(bins)
    vvcosf(&cosPhase, phase, &trigCount)
    vvsinf(&sinPhase, phase, &trigCount)

    var combinedInput = [Float](repeating: 0, count: 2 * bins)
    combinedInput.withUnsafeMutableBufferPointer { buf in
        vDSP_vmul(spec, 1, cosPhase, 1, buf.baseAddress!, 1, vDSP_Length(bins))
        vDSP_vmul(spec, 1, sinPhase, 1, buf.baseAddress!.advanced(by: bins), 1, vDSP_Length(bins))
    }

    // contribution (n_fft x frameCount) = combinedWeight (n_fft x 2*freqBins)
    // @ combinedInput (2*freqBins x frameCount) — one frame's worth of
    // overlap-add contribution per tap, for every frame at once.
    var contribution = [Float](repeating: 0, count: nfft * frameCount)
    HostISTFTBasis.combinedWeight.withUnsafeBufferPointer { weightPtr in
        combinedInput.withUnsafeBufferPointer { inputPtr in
            contribution.withUnsafeMutableBufferPointer { outPtr in
                vDSP_mmul(
                    weightPtr.baseAddress!, 1,
                    inputPtr.baseAddress!, 1,
                    outPtr.baseAddress!, 1,
                    vDSP_Length(nfft), vDSP_Length(frameCount), vDSP_Length(2 * freqBins)
                )
            }
        }
    }

    // Overlap-add: for each tap k, frame i's contribution lands at output
    // position i*hop + k. Fixing k turns this into one strided vector add
    // per tap (20 total) instead of a scalar double loop.
    let fullLength = (frameCount - 1) * hop + nfft
    var outputFull = [Float](repeating: 0, count: fullLength)
    contribution.withUnsafeBufferPointer { contribPtr in
        outputFull.withUnsafeMutableBufferPointer { outPtr in
            for k in 0..<nfft {
                let tapRow = contribPtr.baseAddress!.advanced(by: k * frameCount)
                let outAtTap = outPtr.baseAddress!.advanced(by: k)
                vDSP_vadd(tapRow, 1, outAtTap, vDSP_Stride(hop), outAtTap, vDSP_Stride(hop), vDSP_Length(frameCount))
            }
        }
    }

    // Center-padding removal, matching CustomSTFT.inverse's `if self.center`
    // block exactly (kokoro/custom_stft.py:368-374), including its guard
    // against a too-short waveform.
    let padLen = nfft / 2
    guard padLen > 0, fullLength > 2 * padLen else { return outputFull }
    return Array(outputFull[padLen..<(fullLength - padLen)])
}
