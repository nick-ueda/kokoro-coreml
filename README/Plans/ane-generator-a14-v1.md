# ANE Generator on A14 — Execution Plan v1

Written 2026-07-16 for the FreeReader spike fork. Tasks below are designed to
be executed one-per-session by a coding agent (Sonnet/Opus) with NO prior
context. Read this whole file before starting any task. Execute ONLY the task
you were asked to run. Device (iPhone) runs are done by the owner, never by
you.

## Why

Kokoro background synthesis on the iPhone 12 Pro works today (backgroundSafe
policy, CPU generator) but costs ~2.5–3.5 W — measured ~31–43%/h of an aged
battery (see `SPIKE_RUNBOOK.md` Test 3b and the FreeReader spike doc). The
generator is 72–83% of wall time and runs on CPU only because its Core ML
graph violates the ANE's 16,384-elements-per-axis limit at the graph
BOUNDARIES — not in its body. Goal: an ANE-admissible 3 s generator package.
Expected payoff: ~5–10x lower synthesis power.

## Ground truth — do NOT re-derive, do NOT "correct"

Verified against `export_synth/convert.py`, `export_synth/wrappers.py`
(`GeneratorFromHar`), and `kokoro/istftnet.py` on 2026-07-16:

- 3 s bucket: `bucket_samples=72,000` (24 kHz), traced with 2x internal
  geometry (`geometry_samples=144,000`), `full_f0_len=480`, ASR/x_pre frame
  count 240.
- Generator body time axes at 3 s: x_pre 240 → ups[0] (×10) 2,400 → ups[1]
  (×6) 14,400 → reflection_pad 14,401 → conv_post/spec/phase 14,401 → iSTFT
  (n_fft=20, hop=5) 72,00x. **Max body axis 14,401 < 16,384.** 3 s is the
  largest standard bucket that fits (ceiling ≈ 3.4 s).
- Over-limit tensors (the ONLY ones): `har` input (1, 22, 28,801); the noise
  branch intermediates at har length (28,801) because trimming happens after
  `noise_res`; the waveform output (72,00x).
- `noise_convs[0]`: Conv1d k=12, s=6, pad=3 over har → 4,800, trimmed to
  2,400. `noise_convs[1]`: k=1 over har → 28,801, trimmed to 14,401. A har
  input of length **14,401** is sufficient for both stages' kept outputs
  (out = floor((L+2·3−12)/6)+1 ≥ 2,400 needs L ≥ 14,400).
- A14 ANE is fp16-only. The 16,384 limit may be enforced per-segment or
  whole-graph — open question, probed by T0 (owner-run).
- `.all` and all-stage `.cpuAndNeuralEngine` are FORBIDDEN on the phone
  (hard-fail / jetsam — see `README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md`).
  Always pin per-stage compute units.
- decoder-pre already runs pinned `.cpuAndNeuralEngine` on the A14 with the
  same op family (manual mean/var AdaIN, weight-normed convs).

## Known subtlety — the noise-branch AdaIN statistics shift

`AdaIN1d` (kokoro/istftnet.py) computes mean/var over the CURRENT tensor's
time axis. In the original graph, `noise_res[0]`/`noise_res[1]` see har-length
tensors (4,800 / 28,801) and are trimmed AFTER. Pre-trimming har to 14,401
changes those statistics (computed over 2,400 / 14,401 instead). This is NOT
bit-equivalent to the original — plausibly it is *more* correct (stats over
the region actually used), but it must be quantified by ear and SNR in
PyTorch (T2) BEFORE any Core ML export builds on it. Fallback if audible:
keep the noise branch at full har length in a separate tiny CPU-pinned
package and feed its trimmed outputs to the ANE generator package as extra
inputs.

## Tasks

Recommended executor in [brackets]. Commit each task separately in this repo
(`~/Git/kokoro-coreml`). NEVER touch `~/Git/FreeReader`. If an acceptance
check fails after two focused attempts, STOP and write up what you found —
do not thrash.

### T1 — Export environment + checkpoint restore [Sonnet]

`checkpoints/config.json` and `checkpoints/kokoro-v1_0.pth` are broken
symlinks (they point into the upstream maintainer's machine). Replace them
with real files downloaded from Hugging Face `hexgrad/Kokoro-82M`
(`config.json`, `kokoro-v1_0.pth`). Stand up the Python env per
`requirements-export.txt` (repo uses uv; `uv.lock` present). Then reproduce
the EXISTING 3 s decoder-har export as a baseline: read
`export_synth/main.py` / `convert.py` for the CLI (mode `decoder-har`,
bucket 3s only) and run it.

Acceptance: `kokoro_decoder_har_post_3s.mlpackage` produced; the export
log prints the traced-waveform line for the 3 s bucket; note the printed
har/x_pre shapes and confirm they match Ground Truth above.

### T2 — Quantify the har pre-trim effect in PyTorch [Opus or Sonnet]

Pure PyTorch on the Mac, fp32, no Core ML. Build a modified
`GeneratorFromHar` variant that (a) takes har already trimmed to 14,401,
(b) trims `x_source` BEFORE `noise_res[i]` (i.e., stage-1 noise path runs at
2,400, stage-2 at 14,401). Run both variants on identical real inputs
(reuse the export trace-input plumbing in `convert.py` to get real x_pre /
ref_s / har from the checkpoint and a bench text, seed fixed) and compare
final waveforms: SNR, max-abs diff, and write both WAVs to disk for the
owner to A/B by ear.

Acceptance: a short markdown note in `README/Notes/` with the SNR/max-abs
numbers and WAV paths, plus a one-line verdict: "indistinguishable /
audible". Do NOT proceed to graph changes on your own if audible — stop and
report (the fallback design is a separate decision).

### T3 — Host-side iSTFT in Swift (vDSP) + golden test [Sonnet]

Implement the inverse STFT currently done in-graph, as a Swift function in
`swift/Sources/KokoroPipeline/` (new file `HostISTFT.swift`): input spec
(1, 11, T) and phase (1, 11, T) as fp32 arrays, output waveform. Contract:
replicate `kokoro/custom_stft.py` `CustomSTFT.inverse` EXACTLY — n_fft=20,
hop=5, periodic Hann, including the overlap-add window-sum normalization and
any edge trim; then the final `[..., :bucket_samples]` trim happens at the
call site. n_fft=20 is tiny — a direct O(T·n_fft) loop with vDSP/Accelerate
for the inner products is fine; do not over-engineer an FFT.

Golden test: a Python script (put it in `scripts/`) dumps spec/phase/waveform
for two cases — a synthetic random-seed case and a real 3 s bucket case
(from T1's baseline) — as .npy or raw float32; a Swift test in the package
loads them and asserts max-abs error < 1e-4 against the Python waveform.

Acceptance: `swift test` (or the package's existing test invocation) passes
the golden test on the Mac.

### T4 — `decoder-har-ane` export mode [Opus — gated on T2 verdict "indistinguishable"]

Add an export mode to `export_synth/` producing
`kokoro_decoder_har_ane_3s.mlpackage`:

- Inputs: x_pre (1, 512, 240), ref_s (1, 256), har (1, 22, **14,401**).
- Wrapper: T2's pre-trim variant of `GeneratorFromHar`.
- Outputs: spec (1, 11, 14,401) and phase (1, 11, 14,401) — STOP before
  `stft.inverse` (`torch.exp` / `torch.sin` of conv_post stay IN the graph).
- Keep the existing `rewrite_generator_ups_conv_transpose` (zero-insert)
  rewrite. fp16 compute precision (match whatever the existing decoder-har
  export passes to coremltools). 3 s bucket only.
- Verify with coremltools on the Mac: model loads, predicts under
  `.cpuAndNeuralEngine` (macOS will silently reroute — that's fine, this is
  a shape/parity check, not the admittance proof), and full-pipeline parity:
  spec/phase → T3's golden Python iSTFT → SNR vs the T1 baseline package's
  waveform ≥ 40 dB (fp16 tolerance).
- Also dump `MLComputePlan` op-level device usage for the new package under
  CPU_AND_NE (see the Python snippet in
  `README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md`)
  and record the ANE-mapped op percentage in a `README/Notes/` note.

Acceptance: package produced; parity SNR reported; compute-plan note
written. The on-device admittance proof itself is owner-run, not yours.

### T5 — Bench-app plumbing for the ANE generator [Sonnet]

In `ios-bench/`:

- `prepare_resources.sh`: stage `kokoro_decoder_har_ane_3s.mlpackage`
  (optional — skip gracefully if absent, like the exact-duration packages).
- `Sources/BenchApp.swift`: new `StagePolicy.aneGenerator` = duration/f0n
  `.cpuOnly`, decoderPre `.cpuAndNeuralEngine`, generator
  `.cpuAndNeuralEngine`, addressable via `--policy aneGenerator`; the model
  cache must vend `kokoro_decoder_har_ane_3s` for the generator stage under
  this policy (guard: 3 s bucket only — fail loudly for other buckets).
  NOTE: the pipeline executor must feed the new input shapes (har trimmed to
  14,401) and run T3's HostISTFT on the outputs — look at how
  `executeKokoroSynthesis` builds generator inputs in
  `swift/Sources/KokoroPipeline/` and add a branch keyed on the model's
  input description (query `MLModel.modelDescription` rather than hardcoding
  a policy check inside the pipeline).
- New launch mode `--mode computeplan --model <name>`: load the named
  bundled .mlmodelc with `.cpuAndNeuralEngine`, dump `MLComputePlan`
  per-op preferred-device counts (iOS 17.4+ API:
  `MLComputePlan.load(contentsOf:configuration:)`, `deviceUsage(for:)`),
  print a summary table (`COMPUTEPLAN: <device>: <n> ops`) and append it to
  the results JSON. This is how the owner verifies decoder-pre's real ANE
  residency (Phase 0) and, later, the new generator's.

Acceptance: `xcodebuild -project ios-bench/KokoroIPhoneBench.xcodeproj
-scheme KokoroIPhoneBench -destination 'generic/platform=iOS'
CODE_SIGNING_ALLOWED=NO build` succeeds. Simulator/device execution is
owner-run.

### T6 — AdaIN windowed-stats experiment (Phase 3 pre-test) [Sonnet — only when asked]

PyTorch script: synthesize a ~12 s utterance normally; then re-synthesize
with the generator run in 3 s windows over x_pre/har with a 0.25 s halo
each side (discard halos, hard-concat), AdaIN stats computed per-window.
Write both WAVs + SNR for owner ABX. Do not build any Core ML export for
this.

## Owner-run device steps (not for coding agents)

- T0a: `--mode matrix --keys 3s,30s` on the 12 Pro — the `generator=ne` cell
  probes per-segment vs whole-graph 16,384 enforcement on the CURRENT
  packages.
- T0b: `--mode computeplan --model kokoro_decoder_pre_7s` (after T5) —
  is decoder-pre genuinely ANE-resident?
- Phase 1 gate: `--mode matrix`-style single run of `aneGenerator` policy —
  does `kokoro_decoder_har_ane_3s` load and predict on the phone? Then the
  untethered battery soak (`SPIKE_RUNBOOK.md` Test 3b) under
  `--policy aneGenerator`.

## Status log

- 2026-07-16 **T1 COMPLETE** (commit a901019, review-validated). What later
  tasks inherit: run all Python via `uv run python` (torch pinned 2.5.0,
  coremltools 8.3.0); `checkpoints/{config.json,kokoro-v1_0.pth}` exist as
  local HF-cache symlinks (restore procedure in `.gitignore` if broken); the
  3 s baseline package is already exported and shape-verified at
  `coreml/kokoro_decoder_har_post_3s.mlpackage` — T3's real-case golden and
  T4's parity reference should use it, no re-export needed.

## Execution protocol

One task per fresh agent session, launched in `~/Git/kokoro-coreml`. Give
the agent: this file's path + the single task ID. Branch discipline:

- Sole session in the clone: work directly on `spike/freereader-a14`, no
  task branches, push to `origin` after your task's commit.
- Concurrent sessions: work in a `git worktree` on a task branch
  `ane-plan/t<n>-<slug>` cut from the CURRENT `spike/freereader-a14` tip
  (never from `main` — it predates this plan). Do not push or merge the
  task branch; report done and the owner fast-forwards it into
  `spike/freereader-a14`. The agent must read the
referenced source files before editing, run the task's acceptance check
itself, commit with a message `ane-plan T<n>: <summary>`, and STOP after two
failed acceptance attempts with a written finding instead of thrashing.
Escalate to the owner (who reviews with Fable): any ANECCompile/espresso
error, any parity number below gate, any need to deviate from Ground Truth.
