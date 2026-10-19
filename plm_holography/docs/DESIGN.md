# Model and implementation

## Interfaces

| Module | Responsibility |
|---|---|
| `config.py` | Resolve editable JSON, inherit profiles, validate physical dimensions |
| `propagation.py` | `ASM(phases[S,H,W], colors[S]) -> complex[S,D,h,w]` |
| `schedule.py` | Physical colors, timestamps, exposures, gains, windows and block endpoints |
| `targets.py` | Prepare `intensity[S,C,D,h,w]` from synthetic, NPZ or image inputs |
| `quantization.py` | Physical LUT mapping and hard-forward derivative surrogates |
| `temporal.py` | Duration-weighted causal box means; color-expanded intensity integration |
| `motion.py` | Optional timestamped spatial warping and shared validity masks |
| `losses.py` | Fixed-gain intensity/amplitude MSE, equal weighting per time/color/depth |
| `optimization.py` | Independent, group-joint, global and fixed-history causal solvers |
| `evaluation.py` | Fresh reconstruction from saved phase/code maps and scientific metrics |
| `artifacts.py` | Immutable run inputs, numeric outputs and fixed-scale previews |

Python usage: `load_config(path)`, `ctx = prepare(config)`,
`result = optimize(ctx)`, `save_run(ctx, result)`.
`evaluate_run(run_directory)` independently reconstructs and verifies a run.

## ASM

For color c, incident field u = exp(i phi),

H_c,d(fx,fy) = M_c(fx,fy) exp(i 2 pi z_d sqrt(lambda_c^-2 - fx^2 - fy^2)).

The physical-support mask excludes evanescent frequencies. An optional NA cutoff
also removes frequencies with lambda_c sqrt(fx^2+fy^2) > NA. There is no silent
extra aperture or original-repository F_aperture parameter: numerical aperture
has a different definition from a fraction of the sampled grid's corner radius.

The input complex field is zero padded, FFT transformed with orthonormal scaling,
multiplied by H, inversely transformed, unpadded and cropped. Zero padding does
not mean adding zero-phase illuminated pixels. Without filtering, power is
conserved on the complete computational grid, not necessarily inside the crop.
Padding reduces circular wraparound; it is not a guarantee against all sampling
aliasing for arbitrary distance, wavelength, aperture and grid choices.

FFT mode uses `fftfreq` without fftshift. Float64 kernel construction reduces
large propagation-phase roundoff, followed by casting to the field precision.
Legacy mode reproduces the original endpoint-inclusive centered frequency grid
and float32 construction when used with float32 phases. It exists only for
reference comparisons; old and corrected grids need not produce identical results.

## Temporal model

Let c_s and dt_s identify the active color and duration at physical update s.
The instantaneous color-expanded reconstruction is

R_s,c,d = 1[c_s=c] g_c |ASM_c,d(phi_s)|^2.

For endpoint e, J_e = [max(0,e-W+1), ..., e], W = colors * window_cycles.
Perceived reconstruction is sum(dt_s R_s)/sum(dt_s), with the same temporal
operator applied to the desired *full-color* video V. Thus the reference is not
artificially forced to contain the device's color-sequential artifacts.

Source gains g are fixed. Independent optimization compares g_c duty_c I_s,d
to V_s,c,d; this agrees with the balanced static temporal target convention.
Different exposure durations require explicit illumination choices, not hidden
per-window normalization. The model assumes contiguous exposures and does not
include switch blanking or hardware synchronization delays.

For optional motion, each historical frame at i is expressed relative to the
eye at e. At output coordinate x, sample the source at x+eye_i-eye_e. Bilinear
sampling uses pixel-center coordinates (`align_corners=False`). Each window's
valid mask is the intersection of all contributing source footprints. Normalize
spatial error within this mask separately for every time/color/depth, then
average those values equally. A trajectory with no common valid pixels raises
an error instead of reporting a spuriously low loss.

With omitted/null `stride_cycles`, global uses all endpoints including startup,
and causal optimizes one new phase per physical update. This legacy behavior is
preserved. With explicit S=`stride_cycles`, define W=colors*window_cycles and
K=colors*S. Select exclusive ends W,W+K,W+2K,... and append the sequence end if
needed. For a sequence shorter than W, select only its end. Require 1<=S<=W/colors
to prevent gaps in coverage. All physical phases participate in an objective.

Global averages loss across those selected endpoints while updating all phases.
Causal partitions new variables as [0,first_end), [first_end,next_end), etc.
At end e it integrates history plus new phases over [max(0,e-W),e); historical
intensities are detached and committed phases never change. Only this endpoint
is scored in a causal stage. First jointly solve the full initial window, then
solve S new cycles per stage, except for a shorter tail. This requires the whole
new block's targets before its first exposure; it is causal between committed
blocks, not a claim of instantaneous, future-free operation within a block.

Joint still uses non-overlapping groups of W, and independent still solves each
phase separately; neither changes with stride. Every method is finally evaluated
on the complete sequence at every physical endpoint, including startup. The
selected-window score is also saved and rechecked on reload, separately from the
all-endpoint score. Endpoint arrays in artifacts use zero-based physical indices.

## Quantization derivative

For physical levels l_j, let delta_j = atan2(sin(l_j-phi), cos(l_j-phi)).
Nearest abs(delta_j) gives the true physical state q_hard.

For the smooth surrogate:

w_j = softmax(-delta_j^2 / (2 tau^2)),
q_soft = phi + sum(w_j delta_j).

The forward tensor is `q_hard + (q_soft - q_soft.detach())`, which numerically
equals q_hard, with the derivative of q_soft. STE replaces q_soft with phi.
Tau anneals exponentially from 0.5*(2pi/16) to 0.1*(2pi/16) within each solver
stage. Nonuniform measured LUTs retain this declared nominal-width convention.
Circular-distance branches and hard nearest-level boundaries are nondifferentiable;
the derivative is explicitly a surrogate. We do not promise monotonic improvement.

Each stage evaluates initialization, then every post-update actual phase state.
Best states are detached clones, not live parameter aliases. For continuous
mode the selected state is continuous. For all other modes it is hard discrete,
even when training used a continuous forward pass (`post_quantized`). No separate
final re-quantization is allowed to silently change the selected checkpoint.

## Compute accounting and limits

Each stage performs one initial checkpoint propagation and two propagations per
update (training and post-update evaluation). The run also propagates the common
initial sequence for reporting. Counters count each phase exposure and each
depth-plane propagation, not just Python calls. Final artifact generation reuses
the selected raw intensities; separate reload evaluation has its own propagation.
Measured optimization time excludes plots and compressed output writing.

The global graph retains all frames and all evaluated endpoints. Propagation
batch size limits a single batched FFT, not total autograd storage. Causal avoids
history gradients but the first release retains complete targets and outputs
for reporting. Full-resolution sequences can require substantial CPU/GPU RAM and
disk space. Streaming output and checkpointed global backprop are future work.

No amplitude modulation, measured fill factor, wavelength-dependent efficiency,
device crosstalk, aberrations, camera noise, phase-height dispersion, display
bitplane packing, RGB-D masks, learned propagation or hardware training is
included. These are explicit future extensions, not validated behavior.
