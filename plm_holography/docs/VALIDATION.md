# Validation status — 2026-10-19

The existing Python 3.11.11 / PyTorch 2.2.2 environment was used on CPU.
The initial 26 unit tests passed; that log is `.cache/test_results.txt`.
Tests cover ASM and gradients, spectral/crop conventions, scheduling and
intensity integration, motion alignment, LUT mapping, optimizer modes,
causal history, input/configuration validation and saved-state reproduction.

Saved smoke, motion-example and reconstruction-demo runs passed fresh-process
evaluation. For `results/reconstruction_demo`, full-window intensity MSE is
0.0031683063 and reloaded raw and integrated reconstructions have zero maximum
absolute difference. Its fixed gains of 1.5 and 400 post-quantized updates differ
from the standard benchmark; it is an illustration, not a matched comparison.

The legacy ASM grid was also compared against the original repository using
the saved tutorial phase and its optical parameters; maximum complex-field
difference was zero across the three depths.

The full 24-run benchmark is incomplete at handoff: 14 completed summary rows
were present at the last inspection. Consult `../HANDOFF.txt` for continuation
commands and the current result files for subsequent progress. Full benchmark
ranking, isolated runtime comparisons, CUDA, full-resolution memory use and
physical hardware calibration have not been validated.

## Configurable stride extension

After adding `schedule.stride_cycles`, all **36 tests** passed in 17.436 s;
the log is `.cache/stride_test_results.txt`. Added coverage includes RGB/mono
window membership, startup and short tails, invalid strides, gradients reaching
all participating exposures, immutable committed history, and old benchmark
resume compatibility. Independent and joint results are invariant to stride.

Before editing, outputs were captured for 32 combinations of four temporal
methods, four phase methods, and motion on/off. After editing, omitted/null stride
produced exactly equal phases, raw/integrated intensities and best-loss histories
for all 32 combinations. A further 32 combinations with explicit stride passed,
including static/moving scenes and unequal exposure durations. Zero motion,
SGD/amplitude loss, and fresh-process block-mode reload also passed.

Both explicit W=8, S=4 CLI modes completed small CPU checks: 48 physical exposures,
16x16 phases, three depths, four smooth-surrogate updates per phase. Outputs are
in `results/stride_validation/causal` and `results/stride_validation/global`.
Both selected physical endpoints 24, 36, 48 (saved zero-based as 23, 35, 47),
and both reloaded with exactly zero raw/integrated reconstruction difference.
These short runs verify function, not converged reconstruction quality.

A temporary copy of the original `reconstruction_demo` snapshot was also
evaluated with the new code: its full-window MSE remains 0.0031683063, with zero
raw/integrated reconstruction difference. Original reference-run files were
preserved. Changes are confined to `plm_holography/`; original repository code
outside that folder is unchanged. CUDA/full-resolution stride validation remains
future work.
