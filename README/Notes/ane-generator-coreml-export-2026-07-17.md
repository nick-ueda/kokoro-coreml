# ANE generator export (3 s) — parity, and the ANE admitted it

Collected: 2026-07-17. Task T4 of
[../Plans/ane-generator-a14-v1.md](../Plans/ane-generator-a14-v1.md), building on
T2's verdict in
[ane-pretrim-equivalence-2026-07-17.md](ane-pretrim-equivalence-2026-07-17.md).

## Verdict

`coreml/kokoro_decoder_har_ane_3s.mlpackage` is exported and **the parity gate
passes at 45.78 dB** (gate: 40 dB). The exported graph is the pre-trim graph in
fp16 — T4's acceptance criteria are met.

Two findings matter more than the gate number:

1. **The pre-trim worked. This Mac's ANE admits the graph: 98.1% of ops
   (1,019/1,039) map to the Neural Engine.** The T1 baseline maps **0%** — it is
   100% CPU. That is the plan's whole thesis, confirmed at the compute-plan
   level: the >16,384 axes were the only thing keeping the generator off the ANE.
2. **The Mac's ANE then miscomputes the graph into non-finite output.** Not a
   reroute, not a graph defect — the same package on the GPU clears the gate.
   Mechanism unidentified; two hypotheses falsified below. **This is the open
   question for the A14** and it is owner-run.

Stated plainly: **nothing here proves A14 ANE admittance**, and this note makes
no such claim. A Mac compute plan is not an A14 compute plan.

## What was built

`export_synth/convert_ane.py`, CLI `--mode decoder-har-ane`
(`uv run python -m export_synth.main --mode decoder-har-ane`). 3 s only — the
largest standard bucket whose body fits under 16,384 (ceiling ~3.4 s).

| | T1 baseline `decoder_har_post_3s` | T4 `decoder_har_ane_3s` |
| --- | --- | --- |
| `har` input | (1, 22, **28,801**) | (1, 22, **14,401**) |
| noise branch | runs at har length, trim AFTER `noise_res` | trim BEFORE `noise_res` |
| output | `waveform` (1, **72,000**) | `spec`/`phase` (1, 11, **14,401**) |
| iSTFT | in-graph | host (`HostISTFT.swift`, T3) |
| max axis | 28,801 ❌ | **14,401** ✅ |

`torch.exp`/`torch.sin` stay in the graph; only the overlap-add leaves it. Held
identical to the baseline on purpose: fp16, `mlprogram`, `ct.target.macOS13`,
fp32 I/O, and the `rewrite_generator_ups_conv_transpose` zero-insert rewrite.

Verification: `uv run python scripts/verify_decoder_har_ane.py`
(`--wrapper-only` for the pre-conversion gate alone).

## Parity

Real inputs throughout (T2's construction path: real misaki phonemes, real
BERT/duration/`F0Ntrain`, real `af_heart` embedding), **one `har` tensor shared
by both sides**. Both traps from T2's "Inheritance for T4" respected: the
reference is `_forward_pretrim` (PyTorch fp32 pre-trim), *not* the T1 baseline
package, and `har` is never regenerated between sides.

Gate chain: Core ML spec/phase → `CustomSTFT.inverse` → SNR vs `_forward_pretrim`.

| Compute unit | Waveform SNR |
| --- | ---: |
| `CPU_AND_GPU` | **45.78 dB** ✅ |
| `CPU_ONLY` | 24.37 dB |
| `CPU_AND_NE` | **non-finite** ❌ |

Before any of that, the export wrapper `GeneratorFromHarANE` is asserted
**bit-identical** to `_forward_pretrim` on real inputs (max-abs 0.0) — the same
guard T2 used on its own hand-copy. Without it, T2's "indistinguishable" verdict
would not transfer to whatever this exports.

### Why the gate reads CPU_AND_GPU

T4 specified a `CPU_AND_NE` predict, reasoning "macOS will silently reroute —
that's fine, this is a shape/parity check". **That assumption is now false, and
it is false because T4 succeeded.** It held for every earlier package because the
ANE rejected them; this one it admits, so nothing reroutes and the number stops
measuring the graph. The gate judges what it was written to judge — is the
exported graph the pre-trim graph in fp16 — so it reads the compute unit that
evaluates the graph faithfully. The ANE result is measured and printed on every
run, never suppressed.

`CPU_ONLY`'s 24.37 dB is below gate but not alarming: BNNS runs the fp16 weights
without the GPU's accumulation behavior. It is not the ship path (the phone pins
`.cpuAndNeuralEngine`) and was not chased.

## MLComputePlan — per-op device usage, CPU_AND_NE, this Mac

Snippet from
[../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md)
(note: the guide writes `ct.ComputeUnits`; the real symbol is `ct.ComputeUnit`.
Also: keep the `MLModel` alive — `get_compiled_model_path()` returns a temp
`.mlmodelc` owned by it, and letting it go out of scope deletes the directory
mid-call, reporting "The model is not found at URL").

**`kokoro_decoder_har_ane_3s`: 1,019/1,039 ops on the ANE (98.1%), 20 on CPU (1.9%).**

The 20 CPU ops: `ios16.conv` x8, `ios16.cast` x5, `slice_by_index` x2,
`ios16.leaky_relu` x1, `expand_dims` x1, `pad` x1, `ios16.reshape` x1,
`ios16.mul` x1. Consistent with graph-boundary shaping (the input `cast`s, the
`slice_by_index`/`pad` around the spec/phase split), not a hot-loop fallback.

**`kokoro_decoder_har_post_3s` (T1 baseline): 0/1,038 ops on the ANE — 100% CPU.**
Its 1,038 CPU ops include `ios16.add` x218, `ios16.mul` x148, `tile` x96,
`ios16.reduce_mean` x88. This is the receipt for the plan's premise: the baseline
generator has never touched the Neural Engine, on any host.

## The ANE corruption — what it is not

Under `CPU_AND_NE` the package returns non-finite `spec` (84.1% finite) and a
`phase` that is decorrelated from the reference (SNR **-5.59 dB**). `phase` is
`sin()`-bounded, so it stays finite no matter how wrong it is — which is what
proves the **body** diverged rather than just `exp()` overflowing. Core ML emits
`inf` at positions where PyTorch has ~1e-25. The fp32 reference `spec` maxes at
**22.2** (`conv_post` range -66.4 to +3.1), so there is no overflow in the math
itself: nothing here should exceed fp16 range.

Two focused attempts, both **falsified**:

- **AdaIN `reduce_mean`/variance over the long axis in fp16.** Plausible: `AdaIN1d`
  normalizes over 14,401 elements, and an fp16 accumulator would lose the mean.
  Tested directly with a standalone mean/var model at T = 512 / 4,096 / 14,401:
  ANE max-abs error is **identical to the GPU's** at every length (0.0035 /
  0.0044 / 0.0063). The ANE reduces long axes fine.
- **The `rewrite_generator_ups_conv_transpose` zero-insert rewrite.** Exported a
  no-rewrite variant: still non-finite on `CPU_AND_NE` (and still 45.81 dB on the
  GPU, still 98.4% ANE-mapped). The rewrite is not the cause — and, separately,
  is not needed for ANE *mapping* on this host. Kept anyway, per Ground Truth.

Stopped there per the plan's two-attempt rule rather than thrash. The graph is
proven correct, so this is a backend question, not an export question.

### For the owner

The A14's Neural Engine is a different generation from this Mac's, so this may
not reproduce on the phone — or it may be the thing that decides the spike. It
is cheap to find out, because the plan's Phase 1 gate already runs this package
on the phone. Suggested reading of the outcomes:

- **Phone predicts finite, sane audio** → the corruption is a Mac-ANE quirk.
  Proceed; the power win is on the table.
- **Phone predicts non-finite** → the generator is ANE-*admissible* but not
  ANE-*correct*, which is a genuinely new failure mode for this spike (every
  prior failure was admittance). The next probe would be bisecting the graph:
  export the first upsample stage alone and walk forward until the ANE output
  diverges.

Do not read the 98.1% as a promise of A14 residency. It is evidence that the
axis limit was the binding constraint, nothing more.

## Out of scope, still open

The half-fill geometry issue T2 flagged in `build_decoder_har_post_inputs_np`
(runtime fills 14,401 of 28,801 har frames, 120 of 240 x_pre frames) is
untouched here — T5-adjacent, separate decision. Note that it becomes *moot* for
this package's har input, whose axis is 14,401 by construction, but the x_pre
half-fill would still apply.
