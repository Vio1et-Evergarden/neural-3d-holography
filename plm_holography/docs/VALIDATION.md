# Validation status — 2026-10-19

Current evidence is consolidated here. `../HANDOFF.txt` is the single backlog and
meeting-summary entry. Implementation, passing functional checks and demonstrated
research quality are different claims.

## Current CPU audit

Python 3.11.11, PyTorch 2.2.2, NumPy 1.26.4, Intel macOS, existing project `.venv`.
CUDA is unavailable. **40 tests passed in 18.041 s** after configuration cleanup.
Log: `.cache/repository_audit_tests.txt`. Matplotlib dependency deprecation
warnings did not fail tests.

Coverage:

- ASM identity, plane waves, power/phase, odd rectangular padding/crop,
  evanescent/NA masks, wavelength-specific depths and kernel caching.
- Finite-difference continuous gradients and whole-sequence temporal gradients.
- RGB duty, exposure weighting, startup prefixes, no wrapping, intensity addition.
- Null/explicit stride, W8/S4, short tails, fixed causal history, all methods and
  quantizers, SGD and amplitude loss.
- Circular/nonuniform LUTs, physical codes, hard-forward values, checkpoint
  selection and embedded-calibration reload.
- Zero-motion equivalence, translation direction, matched moving-point tracking,
  shared masks and equal time/color/depth weighting despite mask area.
- Target loading, sRGB, interpolation, inherited config paths and rejection of
  invalid counts/dimensions/pitch and nonfinite smoothing widths.

All **19 complete saved runs** passed fresh-process verification, with **zero**
maximum raw and perceived reconstruction difference and all protected snapshot
hashes unchanged. Evidence: `.cache/repository_audit.json` (14 benchmark runs,
smoke, motion example, reconstruction demo and two stride checks).
Evaluation refreshes only `evaluation.json`.
Hashes of `arrays.npz`, `metadata.json`, `config.json` and `loss.csv` are checked
before/after to protect snapshots. The audit records each run's exact error.

## Existing results and their limits

| Results | Evidence | Limitation |
|---|---|---|
| `reconstruction_demo` | Static 64×64 global; full-window MSE 0.0365478881 → 0.0031683063 | Gain 1.5, 400 post-quantized updates; cannot rank against gain-3/smooth benchmark |
| `stride_validation/causal`, `global` | W8/S4, 48 exposures, three depths; endpoints 24,36,48 | 16×16, four updates, functional checks only |
| `motion_example` | Optimization and reload with supplied translation | 32×32, 25 updates; not a matched-tracking quality comparison |
| `smoke` | Small CLI integration and reload | Functional only |
| `perception_alignment_check` | Fixed-eye versus matched-tracking targets | Targets only, no hologram optimization |
| `benchmark_cpu` | 14 complete paired runs | Ten missing, no complete report |

The full `sliding_w8_s4_*` and `holospeed_known_motion_*` configurations load,
but their 64×64/200-update quality experiments have not been run.

## Benchmark inventory

`results/benchmark_cpu/summary.csv` has **14/24** rows: all 12 static runs and
moving seed0 independent/joint. No `summary.json` or `REPORT.md` yet. This is the
original null-stride, motion-off benchmark, not W8/S4 or eye tracking.

Every completed scene/seed group shares initial-phase and target hashes. Every
run has 9,600 phase updates, 19,296 phase propagations and 57,888 depth
propagations. Equal phase budgets do not imply equal temporal-processing cost or
wall time. Historical concurrent-job timings are not isolated speed measurements.

Completed final full-window MSE values are approximately 0.19–0.21. This shows
executable methods, not sufficient convergence or high perceptual quality.
Source power, ROI light loss, target feasibility, quantization and optimization
need controlled diagnosis. Do not silently fit brightness or mix the separate
lower-power illustration into rankings.

## Preserved regression evidence

- Initial implementation: 26 tests (`.cache/test_results.txt`).
- Stride extension: 36 tests (`.cache/stride_test_results.txt`). Outputs for
  32 legacy combinations were exactly equal with omitted/null stride;
  explicit-stride combinations, CLI runs and old snapshot reload passed.
- Holospeed audit: 38 tests (`.cache/holospeed_audit/test_results.txt`). A moving
  point tracks to a sharp point versus a fixed-eye trail; see
  `HOLOSPEED_ALIGNMENT.md` for equation correspondence and limitations.
- Earlier legacy-grid propagation comparison with the original repository and
  saved tutorial phase had zero complex-field difference across three depths.

Historical counts are provenance; the current suite has 40 tests. Implementation
and cleanup changes are confined to `plm_holography/`.

## Unvalidated / intentionally excluded

No measured hardware LUT/source/geometry validation, camera/display control,
neural training, CUDA/full-resolution validation, real-time guarantee or complete
Holospeed reproduction. FFT batch size does not bound the entire sequence's
memory: full targets/outputs and global autograd storage remain. Controlled
simulation experiments can proceed without neural training or hardware control.
