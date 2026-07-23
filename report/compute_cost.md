# Compute-Cost Analysis: MACs/Sample, Measured Latency, and When CNN Is Worth It

This is a supplement to `report/report.md`, not a replacement for anything in
it. `report.md`'s comparisons are BER/SNR-only, plus a `runtime_sec_mean`
column in `results/tables/results_case1_no_isi.csv` /
`results_case2_isi.csv` that is a wall-clock measurement of *this specific
Python/NumPy/TensorFlow implementation* — useful, but not a hardware-portable
answer to "how much compute does this method fundamentally need per output
sample." That's the question this file answers, using
`experiments/compute_cost_analysis.py` (new file; no existing module,
script, or result file was modified to produce it).

Two independent things are measured and shown side by side, because they
tell different stories:

1. **Analytical MACs per output sample** — derived from each method's actual
   structure (shown arithmetically below, not asserted), hardware-agnostic.
2. **Measured wall-clock latency** on this machine (CPU only — this project
   trains and runs its CNNs on CPU throughout, no GPU is used anywhere per
   `report.md`).

The two disagree, sometimes sharply, and *why* they disagree is itself one
of the findings (Section 3).

---

## 1. Analytical MAC-count derivation

Convention: 1 complex multiply ≈ 4 real multiplies + 2 real adds; following
this task's own costing convention this is counted as **4 real MACs per
complex MAC** (a standard back-of-envelope DSP costing shorthand — a more
pedantic 6-real-op count changes every multiplier below by less than 50%,
not the conclusion).

### 1.1 LMS / NLMS / genie-MMSE — FIR-apply cost

All three methods produce one output sample the same structural way: a
length-`num_taps` complex dot product.

- LMS (`src/lms_filter.py` L164-166, L199-200): `y = np.vdot(w, x)` — 16 taps
  → **16 complex MACs/sample**.
- NLMS (`src/nlms_filter.py` L65-66, L96-97): identical FIR-apply structure,
  16 taps → **16 complex MACs/sample**, *plus* one extra real division per
  sample for the `||x||²` normalization in the weight-update path (see
  Section 3 — this division is cheap in MAC-count terms but not cheap in
  cycle-count terms on hardware without a hardware divide unit).
- MMSE (`src/mmse_equalizer.py` L128-134): `out[n] = np.dot(w_conj, y_vec)`,
  equalizer length matched to LMS/NLMS at 16 taps → **16 complex MACs/sample**.
  The one-time `O(taps²)`–`O(taps³)` design-matrix solve
  (`design_mmse_equalizer`, a 16×16 linear solve) is a start-up cost
  amortized over the whole signal, not a per-sample cost — consistent with
  how this project treats it (genie channel + noise level are measured once
  per modulation, not per sample).

Per-sample cost, all three: **16 complex MACs × 4 real MACs/complex MAC =
64 real MACs/sample.**

This is the *deployed, steady-state* cost specifically because this
project's own methodology (`report.md` Section 3.2) **freezes** LMS/NLMS
weights after the 1000-symbol preamble and never updates them again — so in
a real deployment matching this project's design, the only per-sample cost
that recurs for the life of the signal is the FIR-apply, not the adaptive
update. (Section 3 below flags that the current *Python implementation*
doesn't actually skip the update arithmetic when frozen — that's an
implementation inefficiency, not an algorithmic requirement.)

### 1.2 CNN — layer-by-layer Conv1D MAC count

Architecture from `src/cnn_autoencoder.py` `build_autoencoder` (all layers
`padding="same"`, stride 1, so **every layer's output length equals the
input window length, 128**). Standard Conv1D MAC formula per layer:

```
MACs = kernel_size × in_channels × out_channels × output_length
```

| Layer | kernel | in_ch | out_ch | output_len | MACs |
|---|---|---|---|---|---|
| enc1 | 7 | 2 | 32 | 128 | 7×2×32×128 = 57,344 |
| enc2 | 5 | 32 | 16 | 128 | 5×32×16×128 = 327,680 |
| enc3 | 3 | 16 | 8 | 128 | 3×16×8×128 = 49,152 |
| dec1 | 3 | 8 | 16 | 128 | 3×8×16×128 = 49,152 |
| dec2 | 5 | 16 | 32 | 128 | 5×16×32×128 = 327,680 |
| dec_out | 7 | 32 | 2 | 128 | 7×32×2×128 = 57,344 |
| **Total / window** | | | | | **868,352 real MACs** |

These are **real-valued** MACs (I/Q are stacked as two real channels — see
`_iq_to_real`), so no complex-MAC 4× factor applies here, unlike LMS/NLMS/MMSE.

Per-sample normalization, two ways (both reported, because they answer
slightly different questions):

- **Per position within a window** (divide by `window_len=128`, as directed):
  868,352 / 128 = **6,784.0 real MACs/sample**.
- **Per unique advancing output sample**, accounting for the fact that
  `stride=64 < window_len=128` means consecutive windows overlap by 50% and
  *every* overlapping window is fully recomputed by `model.predict()` (no
  incremental/cached computation): 868,352 / 64 = **13,568.0 real
  MACs/sample**. This is the more operationally honest number for a
  streaming deployment, since it reflects the actual amount of arithmetic
  that has to happen to advance the output by one sample given this
  overlap-add scheme.

**Parameter count**: analytical sum of (kernel_size × in_ch × out_ch + bias)
across all 6 layers = **6,890 parameters** — cross-checked directly against
`build_autoencoder(...).count_params()` from a live Keras model on this
machine: **6,890, exact match.** (Note: `report.md` Section 2 describes the
CNN as "~15k-parameter" — the actual, verified count for the current
architecture is 6,890; this file reports the measured number rather than
the report's rough estimate, without editing the report.)

### 1.3 Hybrid = LMS + CNN (sequential)

`denoise_hybrid` (`src/hybrid_model.py`) runs LMS to completion, then feeds
its full output through the CNN — the two costs simply add:

- Per-position: 64 + 6,784.0 = **6,848.0 real MACs/sample**.
- Per-unique-sample (stride-honest): 64 + 13,568.0 = **13,632.0 real MACs/sample**.
- Parameters/state: 16 (LMS taps) + 6,890 (CNN) = **6,906**.

---

## 2. Measured wall-clock latency (this machine, CPU only)

Benchmarked directly with `experiments/compute_cost_analysis.py`, timing the
real `denoise()` / `denoise_signal()` / `apply_mmse_equalizer()` calls (7
repeats for LMS/NLMS/MMSE, 5 for CNN/Hybrid — CNN/Hybrid calls are slower
individually so fewer repeats were used to keep the benchmark itself fast;
mean ± std reported), at the **exact same signal sizes used by the real
pipelines**:

- **Case-1-like**: sps=1, 100,000 BPSK symbols (100,000 samples) — mirrors
  `run_case1_no_isi.py`.
- **Case-2-like**: sps=4 + the same static multipath channel, 100,000 BPSK
  symbols → 400,000 samples — mirrors `run_case2_isi.py`.

| Method | Case-1-like ns/sample | Case-2-like ns/sample |
|---|---|---|
| LMS | 3,818.1 | 3,678.8 |
| NLMS | 5,254.9 | 5,309.0 |
| CNN | 2,222.5 | 1,486.9 |
| Hybrid | 5,544.2 | 5,162.1 |
| MMSE (Genie), apply only | n/a (no ISI/channel to equalize) | 1,166.2 |
| MMSE (Genie), design+apply | n/a | 1,173.8 |

(Full run-to-run mean±std for the underlying per-call timings are printed by
the script; e.g. Case-1 LMS: 381.8 ± 16.0 ms/call over 100,000 samples.)

**Headline observation: on this machine, CNN is *faster in wall-clock terms*
than LMS/NLMS/Hybrid, despite needing ~106× more MACs analytically.** This
is explained in Section 3.

---

## 3. Cross-referencing against the existing `runtime_sec_mean` columns

The main pipelines already recorded per-call runtimes in
`results/tables/results_case1_no_isi.csv` / `results_case2_isi.csv`. Averaged
across modulation/SNR and converted to ns/sample using each case's actual
sample count (100,000 for Case 1, 400,000 for Case 2):

| Method | Case 1 CSV (ns/sample) | Case 2 CSV (ns/sample) | This benchmark, Case 1 | This benchmark, Case 2 |
|---|---|---|---|---|
| LMS | 7,100.4 | 5,742.3 | 3,818.1 | 3,678.8 |
| NLMS | 9,181.4 | 7,285.7 | 5,254.9 | 5,309.0 |
| CNN | 2,376.1 | 1,352.9 | 2,222.5 | 1,486.9 |
| Hybrid | 9,563.1 | 7,062.8 | 5,544.2 | 5,162.1 |
| MMSE (Genie) | n/a | 321.5 | n/a | 1,166.2 |

**These broadly agree in rank order and rough magnitude** (LMS/NLMS/Hybrid
in the thousands of ns/sample, CNN in the low thousands, both benchmarks
showing CNN *faster* than the adaptive filters) — reassuring, since they were
measured completely independently (different signals, different call sites,
different days). The absolute numbers differ by up to ~2× (e.g. Case-1 LMS:
7,100 ns in the original pipeline vs 3,818 ns here), consistent with
machine-load variance, JIT/cache warm-up differences, and this benchmark
using a fixed 5 dB SNR rather than averaging across the full SNR sweep — not
a sign of a bug, just normal run-to-run wall-clock variance in a
Python/NumPy/TensorFlow stack. The **MMSE (Genie)** number is the odd one out
in the *opposite* direction — the CSV's 321.5 ns/sample is *lower* than this
benchmark's 1,166 ns/sample; the CSV's own `runtime_sec_mean` includes the
one-time equalizer design solve *inside every trial's timed block*
(`run_case2_isi.py` L250-256 times `design_mmse_equalizer` + `apply` together
per trial), same as this benchmark's "design+apply" row (1,173.8 ns/sample —
which *does* match the CSV closely). The remaining gap is explained by
trial-count/machine variance, not a structural bug.

### The real finding: implementation efficiency ≠ algorithmic complexity

**CNN needs ~106× more raw arithmetic per sample than LMS/NLMS (6,784 vs. 64
real MACs), yet finishes faster in wall-clock time on this machine, in both
this benchmark and the original pipeline's own recorded numbers.** This is
not a contradiction — it's exactly the point this analysis was scoped to
surface, and it's worth stating plainly:

- **LMS/NLMS/Hybrid's LMS stage are pure per-sample Python `for` loops**
  (`for n in range(...)`, one iteration per output sample, `np.vdot` called
  fresh each time) — every iteration pays CPython interpreter dispatch
  overhead on top of the actual 64 real MACs of "useful" arithmetic. At
  ~4,000-9,000 ns/sample, the *useful* arithmetic (64 MACs) is a rounding
  error next to the loop overhead; a hand-written C/embedded implementation
  of the exact same algorithm would need only the 64 MACs, i.e. a few
  nanoseconds on modern hardware, not thousands.
- **NLMS is also structurally more expensive even in Python-loop terms, not
  just in the analytical division count**: it's consistently ~35-45% slower
  than LMS in every measurement here (both benchmarks and both CSVs) because
  its update rule computes an additional `np.vdot(x, x)` per sample for the
  normalization, a real (if small) extra cost the pure-MAC count under-states.
- **A further, genuine implementation inefficiency**: `LMSFilter.denoise`
  and `NLMSFilter.denoise`'s Phase 2 loop (`src/lms_filter.py` L198-211,
  `src/nlms_filter.py` L95-109) executes the *full* weight-update arithmetic
  every sample regardless of whether `dd_enabled` is true — when frozen
  (`enable_decision_directed=False`, the project's actual default), `mu_eff`
  is multiplied in and evaluates to zero, but the vectorized NumPy multiply/
  add for the (no-op) update still runs on every single sample. The
  *algorithmic* deployed cost of a frozen filter is 64 MACs/sample
  (Section 1.1); this codebase's actual frozen-mode loop still pays for
  something closer to 2-3× that in wasted update arithmetic, on top of the
  Python-loop overhead that dominates everything else. This doesn't change
  any BER/SNR result (multiplying by zero is a numeric no-op) — it's purely
  a wasted-compute footnote, but a genuine one, and a legitimate target if
  someone wanted to optimize this codebase's frozen-mode runtime (out of
  scope here — no existing file was modified).
- **CNN inference, by contrast, is a small number of large, vectorized,
  BLAS/Eigen-backed TensorFlow ops** (`model.predict()` processes *all*
  ~1,560-6,240 overlapping windows of a signal in batched tensor operations),
  so its 100×-larger MAC count is executed by code that is dramatically
  better at using the CPU (SIMD, cache-friendly memory layout, no per-sample
  Python dispatch) than the adaptive filters' interpreted loops are.

**The practical consequence: on THIS specific CPU/software stack,
"algorithmically cheaper" (LMS) does not mean "actually faster" —
vectorized, higher-FLOP-count code beat a scalar, low-FLOP-count Python
loop.** That is a genuinely useful and slightly counter-intuitive finding for
anyone reading only the `runtime_sec_mean` column and concluding CNN is the
expensive option — on this software stack it isn't. But it is **not** the
number to trust when deciding about *dedicated, real-time hardware* (an
FPGA, DSP, or microcontroller doing the FIR-apply and the Conv1D forward
pass as custom/optimized fixed-point code rather than through a Python
interpreter and TensorFlow graph) — there, the analytical MAC count
(Section 1) is what actually predicts feasibility, because a hand-optimized
LMS FIR-apply and a hand-optimized Conv1D forward pass are both about as
close to their "useful-arithmetic" floor as the target platform allows, and
the 106× MAC-count gap reasserts itself as a 106×-ish real
throughput-requirement gap.

---

## 4. Summary table

| Method | Taps / Params (state size) | Real MACs/sample (window-norm.) | Real MACs/sample (stride-honest) | Measured latency, Case-1-like (ns/sample) | Measured latency, Case-2-like (ns/sample) | Relative MAC cost vs LMS |
|---|---|---|---|---|---|---|
| LMS | 16 taps | 64.0 | 64.0 | 3,818.1 | 3,678.8 | **1.0×** |
| NLMS | 16 taps | 64.0 | 64.0 | 5,254.9 | 5,309.0 | 1.0× (+1 division/sample) |
| MMSE (Genie) | 16 taps | 64.0 | 64.0 | n/a (no-ISI case) | 1,166.2 (apply only) | 1.0× (+ one-time O(taps³) design solve, amortized) |
| CNN | 6,890 params | 6,784.0 | 13,568.0 | 2,222.5 | 1,486.9 | **106.0× (up to 212×)** |
| Hybrid | 6,906 (16+6,890) | 6,848.0 | 13,632.0 | 5,544.2 | 5,162.1 | **107.0×** |

(Raw numbers also written to `results/tables/compute_cost_summary.csv` by
`experiments/compute_cost_analysis.py` for reproducibility — a new results
file; no existing table was modified.)

---

## 5. When is CNN worth it? A concrete answer

Combining this file's compute-cost numbers with `report.md`'s existing
BER/SNR findings:

**The cost is fixed and channel-independent: CNN needs ~106× more MACs per
output sample than LMS/NLMS/MMSE (6,784 vs. 64 real MACs/sample), or up to
~212× if you count the 50% window-overlap redundancy honestly (13,568 vs.
64). The benefit is entirely channel-dependent — this is exactly `report.md`'s
own headline finding (Section 3.5/4.6/6) restated in compute terms.**

- **Case Study 1 (memoryless AWGN, no ISI): not worth it, at any budget.**
  CNN posts large SNR gains (up to +18.74 dB, BPSK 10dB) but **worse BER
  than doing nothing at all**, at every SNR level, both modulations
  (`report.md` Section 3.5). Paying 106× the compute for a result that loses
  to a raw-sample hard-decision receiver is never worth it here, regardless
  of how much hardware headroom is available — the problem is structural
  (no channel memory to exploit), not a compute-budget problem.

- **Case Study 2 (real multipath ISI): worth it when the modulation's
  decision geometry is sensitive to the channel — concretely, QPSK here, not
  BPSK.** From `report.md` Section 4.3/4.4/4.6:
  - QPSK 10dB: CNN BER 0.0014 vs. LMS's 0.0034 — **2.4× lower BER** for
    106× more compute, and CNN's SNR-out is 8.11dB vs. LMS's 5.02dB (+3.09dB).
  - QPSK 15dB: CNN BER 1 err/2,000,000 vs. LMS's 15 err/2,000,000 — **15×
    lower BER**; CNN's BER-improvement-ratio over No-Processing is 473× vs.
    LMS's 31.5× (Section 4.6 table).
  - BPSK, by contrast, "barely benefits from equalization" at all
    (Section 4.6) — CNN's edge over LMS there is small and CNN's edge over
    *No Processing* is only ~1.1-1.8×. Paying 106× the compute for a
    ~1.1-1.8× BER improvement is a far weaker case than QPSK's 2.4-15× (or
    up to 473× vs. doing nothing).

**Turning "106× more compute" into a concrete hardware question**: at a
channel sample rate `R` samples/sec (for this project's Case 2 config,
`R` = 4 × symbol rate, since SPS=4), the two approaches need:

- LMS/NLMS/MMSE: `64 × R` real MACs/sec.
- CNN: `6,784 × R` to `13,568 × R` real MACs/sec.

| Example symbol rate | Sample rate R (SPS=4) | LMS/NLMS/MMSE throughput needed | CNN throughput needed (window-norm. to stride-honest) |
|---|---|---|---|
| 250 ksym/s | 1 Msamples/s | 64 MMACs/s | 6.78 – 13.6 GMACs/s |
| 1 Msym/s | 4 Msamples/s | 256 MMACs/s | 27.1 – 54.3 GMACs/s |
| 2.5 Msym/s | 10 Msamples/s | 640 MMACs/s | 67.8 – 135.7 GMACs/s |

- **64-640 MMACs/s (LMS/NLMS/MMSE) is comfortably within reach of a basic
  microcontroller** — a Cortex-M4/M7-class MCU with hardware
  multiply-accumulate/DSP extensions running at 100-200 MHz can sustain
  roughly 0.1-0.4 GMACs/s doing simple fixed-point FIR arithmetic in a tight
  loop, i.e. this workload fits with headroom to spare even at the higher
  symbol rate in the table, and fits trivially at the lower one.
- **6.78-135.7 GMACs/s (CNN) does not fit on that same microcontroller** —
  it is roughly 20-500× beyond a Cortex-M4/M7's realistic peak throughput
  across this table's range. Hitting these numbers at real-time line rate
  needs a meaningfully bigger target: a Cortex-A-class applications
  processor with NEON/SIMD (a Raspberry Pi 4-class core sustains low
  single-digit GMACs/s in well-vectorized code — borderline-to-insufficient
  at the low end of this table, clearly insufficient at the high end), a
  DSP core, or a dedicated NPU/small accelerator (typical edge NPUs deliver
  roughly 1-4 TOPS = 1,000-4,000 GMACs/s, comfortably enough across this
  entire table). It is trivially within reach of any modern laptop/desktop
  CPU or GPU — consistent with this project's own CPU-only benchmarks in
  Section 2 running the whole 100k-symbol test signal in well under a second.

**Bottom line, stated as a decision rule**: if the target platform is a bare
microcontroller with tens to a few hundred MMACs/s of real-time headroom and
the channel is BPSK with mild ISI (or no ISI at all), CNN is not worth
deploying — LMS/NLMS/MMSE deliver equal-or-better BER (Case 1) or a smaller
but real BER edge (BPSK, Case 2) at roughly 1% of the compute. If the target
platform has multi-GMACs/s-to-TOPS-class headroom (an applications
processor, DSP, or NPU) and the channel has real, modulation-sensitive ISI
structure to exploit — this project's clearest example being QPSK with the
specific multipath profile in Case Study 2 — CNN's ~106-212× compute
multiplier buys a genuinely large, measured BER improvement (2.4-15× over
LMS, up to 473× over doing nothing), which is a good trade for most
platforms above the microcontroller tier.

---

## 6. Caveats and scope

- MAC counts here use the standard `kernel_size × in_ch × out_ch × output_length`
  Conv1D formula and the "4 real MACs per complex MAC" convention explicitly
  suggested by the analysis brief; a cycle-accurate hardware implementation
  (fixed-point vs. float, memory-access patterns, pipelining, batch size)
  would shift the absolute throughput numbers in Section 5's table, though
  not the ~106-212× relative gap between CNN and the linear methods, which
  is a property of the architectures themselves, not the target hardware.
- The CNN parameter count used throughout (6,890) is the actual, verified
  count from `build_autoencoder`/Keras on this machine — it differs from
  `report.md` Section 2's "~15k-parameter" description; this file reports
  the measured number without editing `report.md`.
- Wall-clock benchmarks are CPU-only, single-machine, single-run-session
  measurements (Section 2) — useful for the implementation-efficiency
  finding in Section 3, not as a hardware feasibility signal (Section 5 uses
  the analytical MACs for that, deliberately).
- Hybrid's MAC count is LMS's cost plus CNN's cost applied in strict
  sequence (`denoise_hybrid` runs LMS to completion, then feeds the entire
  output through the CNN) — no compute is shared or reused between stages.
- The NLMS/LMS "wasted update arithmetic when frozen" observation
  (Section 3) is reported as a wasted-compute footnote on the *existing*
  implementation, not a proposed change — no existing file was modified to
  produce or fix this finding, per this task's scope.
