# PLM holography research library

Self-contained ASM simulations for a phase-only, 4-bit PLM: RGB focal stacks,
incoherent temporal integration, global and causal sequence optimization, and an
optional eye-motion model. The original repository is preserved as a reference;
this package does not import its code or hardware dependencies.

Progress, the supervisor-meeting summary and remaining tasks are consolidated in
**`HANDOFF.txt`**. Verification is in `docs/VALIDATION.md`: 40 CPU tests pass;
the standard benchmark has 14/24 complete runs. Functional checks do not yet
establish dynamic reconstruction quality or superiority of sliding windows.

The default motion-disabled runs are fixed-eye baselines, not full Holospeed
reproductions. See `docs/HOLOSPEED_ALIGNMENT.md` for the paper/code audit.
`holospeed_known_motion_causal.json` and `holospeed_known_motion_global.json`
enable known tracking for the supplied moving scene and score every physical
endpoint; their RGB optics and LUT surrogate remain explicit adaptations.

## Run

Use the existing project environment. No installation is required:

```bash
cd /Users/zack/Computing_PhD/neural-3d-holography/plm_holography
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m plmholo evaluate --run results/smoke
```

This checkout already has `results/smoke`. To create a new small check, use
`optimize --config configs/smoke.json` in a fresh checkout, or copy that config
inside `configs/` and select a new `output`. Completed runs cannot be overwritten.
Current W8/S4 work uses `sliding_w8_s4_causal.json` / `sliding_w8_s4_global.json`;
the base profile retains its original global/null-stride defaults.

To continue the partial CPU comparison, first ensure another copy is not running:

```bash
../.venv/bin/python -m plmholo benchmark --config configs/benchmark_cpu.json --resume
```

This runs 24 experiments: two scenes, three paired seeds, and four temporal
methods. Each uses 64x64 phases, three depths, 48 physical RGB subframes, an
overlapping 24-subframe window, and 200 Adam updates per phase exposure.
`benchmark --resume --config ...` skips completed runs only when configurations
match. Use a new output path when changing any settings. A normal `optimize`
command refuses to overwrite a completed run.

## Configuration map

These profiles serve distinct purposes; their scores should not be mixed.

| Profile | Purpose / status |
|---|---|
| `lab_rgb_ideal.json` | Shared ideal optics, gain 3, 200 updates; global/null-stride default |
| `sliding_w8_s4_causal.json`, `sliding_w8_s4_global.json` | Configurable-window research entry; full quality runs pending |
| `holospeed_known_motion_causal.json`, `holospeed_known_motion_global.json` | Matched deterministic tracking at every physical endpoint; full quality runs pending |
| `benchmark_cpu.json` | Four methods, two scenes, three seeds; 14/24 complete |
| `reconstruction_demo.json` | Static gain-1.5, 400-update post-quantized illustration |
| `motion_example.json` | Small motion integration example, not matched-tracking validation |
| `smoke.json` | Short functional run |
| `tutorial_reference.json` | Earlier monochrome optical conventions, new loss/targets |
| `full_resolution_cuda.json` | Future GPU template; memory and execution unvalidated |

Source is in `plmholo/`, tests in `tests/`, calibration examples in `data/`,
experiment snapshots in ignored `results/`, and audit/render caches in ignored
`.cache/`. Preserve calibration and experiment data when cleaning caches.

## Main settings

Configuration files are JSON and can inherit another JSON file using the
`extends` key. File paths introduced in each configuration resolve relative to
that file, including paths in inherited configurations. Outputs must stay inside
this project folder. `config.json` in each run records all resolved values.

| Setting | Meaning / default |
|---|---|
| `optimization.method` | `independent`, `joint`, `global`, or `causal` |
| `optimization.optimizer` | `adam` (default) or plain `sgd` |
| `optimization.loss` | `intensity_mse` (default) or `amplitude_mse` |
| `optimization.iterations` | 200 updates per optimized phase |
| `optimization.lr` | null selects 0.01 continuous/post-quantized, 0.001 STE/smooth |
| `quantization.method` | `continuous`, `post_quantized`, `ste`, or `smooth` (default) |
| `schedule.window_cycles` | 8: eight patterns per color, 24 physical updates in RGB |
| `schedule.stride_cycles` | null preserves per-physical-update optimization; integer selects block stride in RGB/mono cycles |
| `schedule.cycles` | 16: total RGB cycles in the simulated sequence |
| `schedule.order` | `["G","R","B"]`; one occurrence of each configured color |
| `schedule.rate_hz` | 1440 physical updates per second, not 1440 RGB cycles |
| `schedule.gains` | fixed source gains in `optics.colors` order |
| `motion.enabled` | false; true enables the optional translation model |
| `device` | `cpu`, `cuda`, or `auto` (CUDA if available, otherwise CPU) |

`optics.colors` is the logical channel order, usually R,G,B; `schedule.order`
is the physical playback order. They are deliberately separate.

`lab_rgb_ideal.json` contains provisional lab-oriented wavelengths and 10.8 um
pitch. Its ideal 16-state LUT and 3.4/4.4/5.7 mm demonstration depths are **not a
calibrated model of the experimental setup**. `tutorial_reference.json` keeps the
earlier optical parameters and legacy grid, but uses new synthetic targets and
fixed brightness; it is not a replay of the old adaptive-scale tutorial loss.
`full_resolution_cuda.json` is an unvalidated 800x1358 template, not a promise of
real-time throughput or GPU memory sufficiency.

`reconstruction_demo.json` is a separate fixed-lower-power illustration: source
gains 1.5 per color, 400 continuous updates, and selection of the best actual
post-quantized state. It explores a different illumination budget; it is **not**
part of the common-gain benchmark or evidence that its method beats that benchmark.
Its gains are explicitly specified, never fitted during optimization.

## What each method means

- **Independent:** optimize each phase against the instantaneous active-color
  focal stack. Multiply raw intensity by source gain times that color's duty
  fraction, so the target convention agrees with the temporal methods.
- **Joint:** optimize each non-overlapping frame group against its integrated
  target. Evaluation still measures every overlapping window, including those
  crossing group boundaries.
- **Global:** update all phases together using the selected window endpoints.
  With the default null stride, use every endpoint. It has access to the whole
  target sequence.
- **Causal:** cache and freeze previously committed intensities. With the default
  null stride, optimize only the newest physical phase at each update. With an
  explicit cycle stride, jointly optimize the first window, then only each new
  block. Targets beyond that block never enter its objective. A block must be
  planned before its first exposure and requires its targets in advance; this
  is block-causal planning, not zero-lookahead single-exposure streaming.

There is no claim that these objectives have equal information access or that
one method always wins. The benchmark pairs initial phases and seeds, fixes the
optical model and brightness, and reports actual propagation counts and runtime.

## Configurable overlapping windows

Both global and causal modes accept, for example:

```json
"schedule": {"window_cycles": 8, "stride_cycles": 4}
```

Here W=8 and S=4 are **RGB cycles**, not individual physical PLM updates. For
monochrome, a cycle is one physical update. In RGB, the windows cover cycles
1–8, 5–12, 9–16: physical updates 1–24, 13–36, 25–48. An integer stride must
satisfy 1 <= S <= W; S=W gives non-overlapping windows. An omitted or null stride
retains the old behavior exactly, including optimizing startup prefixes.
`stride_cycles: 1` instead means one complete RGB cycle, not one physical update.

In explicit-stride causal mode, jointly optimize cycles 1–8 first, then freeze
them. For window 5–12, only cycles 9–12 can change; cycles 5–8 contribute fixed
historical intensity. Global mode instead updates the whole sequence together,
averaging errors at the selected window endpoints. It does not separately commit
one window at a time. Independent and fixed-group joint baselines ignore stride;
their optimization definitions remain unchanged.

Explicit-stride optimization starts at the first complete window, without adding
separate startup-prefix objectives. If the entire sequence is shorter than W,
optimize its one available prefix. Always include the sequence's final endpoint:
when the remaining new block is shorter than S, optimize all of that block with
the available trailing history. This last step can be shorter than the configured
stride, and the final window still has length W when enough history exists.
There is no wraparound, duplicated phase, or dropped tail.

**Evaluation always covers every physical update**, including intermediate and
startup windows, regardless of optimization stride. Thus a large stride cannot
hide intermediate-frame errors by removing them from the main quality metrics.
`configured_loss_selected_windows` is the selected-window diagnostic (the global
training objective); `configured_loss_all_timestamps` includes every endpoint.
For causal mode, `loss.csv` records each committed block's own objective.
Saved `selected_window_endpoints` are zero-based physical indices, not RGB cycles.

Ready-to-run examples (separate output directories):

```bash
../.venv/bin/python -m plmholo optimize --config configs/sliding_w8_s4_causal.json
../.venv/bin/python -m plmholo evaluate --run results/sliding_w8_s4_causal

../.venv/bin/python -m plmholo optimize --config configs/sliding_w8_s4_global.json
../.venv/bin/python -m plmholo evaluate --run results/sliding_w8_s4_global
```

Both inherit the same optics, targets, gain, seed and quantizer. Each phase still
receives the configured number of optimizer updates; changing S changes which
variables are optimized together, not their per-phase update budget. These are
offline simulations; no real-time block-computation guarantee is made.

## Physical conventions

Uniform unit-amplitude illumination gives `u0 = exp(i*phase)`. Each physical
subframe uses its assigned wavelength and propagates independently to all
depths. Time multiplexing accumulates `abs(field)**2`, not fields or amplitudes.

RGB subframes light only one channel at a time. Raw per-frame intensity excludes
source gains; the color-expanded video multiplies it by the fixed active source
gain. Duration-weighted window means are then computed independently for each
color and depth. Gains `[3,3,3]` compensate the one-third duty cycle in normalized
simulation units. They are not measured laser powers. Unequal exposures retain
their actual duty fractions; they do not trigger automatic gain fitting.

The desired RGB video is a full-color linear-intensity reference sampled on the
physical timeline. It is integrated through the same temporal and motion model,
not gated to the actual PLM's active color. This makes discrepancies caused by
the physical color schedule visible. For a static target and complete balanced
window, ideal RGB duty compensation reproduces the reference brightness.

At startup, integrate the available prefix with its actual duration. Startup
errors participate in default null-stride global/causal optimization; explicit
stride instead starts at the first full window as described above. Startup is
always evaluated and reported separately; the main benchmark score uses complete
windows. No end-to-start wrapping occurs.
Intensity MSE is the default. Amplitude MSE takes square roots **after** temporal
intensity integration. Neither fits a brightness multiplier.

## Replace the LUT

Supply a CSV with 16 rows per configured color:

```text
color,code,phase_rad
R,0,0.0
R,1,0.38
...
G,0,0.0
...
B,15,5.91
```

This snippet is schematic; a real file must contain every code 0 through 15
exactly once for each color. A complete ideal table is provided at
`data/ideal_lut.csv`. Set `quantization.lut_path` to the new file. Phase responses
may be nonuniform and nonmonotonic; table order does not redefine physical codes.
Nearest circular distance selects a state, with the first CSV row breaking exact
ties. LUT phases are in radians at the specified wavelength, not micrometers of
mirror displacement. Code maps are not TI transport bitplanes.

Quantization modes:

- `continuous`: reference continuous phases; no discrete code map is claimed.
- `post_quantized`: train with continuous propagation, evaluate and retain the
  best hard-quantized candidate at every update, including initialization.
- `ste`: hard optical forward pass, identity derivative surrogate.
- `smooth`: hard optical forward pass with a soft circular-distance derivative,
  annealed from 0.5 to 0.1 nominal level spacings.

Smooth and STE gradients are surrogates, not derivatives of the true step
function. Actual discrete loss may rise. Each local optimizer stage retains its
best evaluated physical state. The smooth method is our explicit implementation,
not the Agustsson et al. soft-rounding function identified in Holospeed's supplement.

## Inputs and optional motion

Built-in `static` and `moving` scenes are procedural RGB focal stacks. Their
textures and depth variations are prescribed tests, not physically rendered
incoherent scene focal stacks or a guarantee of holographic realizability.

Numerical targets: set `target.kind="npz"` and `target.path` to an NPZ containing
`intensity[T,C,D,H,W]` and, for T>1, strictly increasing `times_s[T]`. Intensities
must be nonnegative and finite. A single time sample broadcasts. Different image
sizes are aspect-preserving center-cropped, then resized in linear intensity.

Image manifests: set `target.kind="manifest"` and supply JSON:

```json
{
  "encoding": "srgb",
  "frames": [
    {"time_s": 0.0, "depth_images": ["z0.png", "z1.png", "z2.png"]},
    {"time_s": 0.02, "depth_images": ["z0_next.png", "z1_next.png", "z2_next.png"]}
  ]
}
```

Image paths resolve against the manifest. RGB images supply their corresponding
channels; monochrome profiles also accept grayscale. Use `encoding="linear"`
only when the images already store linear intensity. Dynamic data are linearly
interpolated, with endpoints held outside their range. This is offline target
preparation; a real streaming application must supply its high-speed targets
without relying on future low-rate frames.

Motion defaults to off and is not imported in the core runtime. See
`configs/motion_example.json`. Use `motion.velocity_px_s=[vx,vy]` for a supplied
constant trajectory, or `motion.path` for JSON/NPZ with `times_s` and
`positions_px[T,2]` in x,y order. Positions are interpolated to exposure midpoints.
Both reconstruction and target undergo the same shifts. The shared mask excludes
pixels outside any contributing frame, so padding cannot lower the loss.
Offsets are in output-grid pixels; this is a retinal translation approximation,
not a calibrated eye-angle or pupil model. No motion-estimation dependency is
required, and a constant eye position exactly follows the motion-off path.

## Saved outputs and interpretation

- `config.json`, `metadata.json`: parameters, versions, hashes, gains, elapsed
  optimization time, compute counts, and initial/final metrics.
- `arrays.npz`: actual phases/code maps, embedded LUT, exposure schedule, sampled
  target, raw intensities, perceived intensities, masks, and eye positions.
- `loss.csv`: training and actual evaluated loss, best within each local stage,
  and gradient RMS. Local-stage curves are not one global convergence curve.
- `comparison.png`: final-time target/initial/final at each depth. All previews
  use fixed linear [0,1]; overshoots are clipped only for display and counted in
  metadata. Numeric arrays and metrics remain unclipped.
- `temporal_error.png`: errors at every physical endpoint, including startup.
- `optimization.png`: current/best stage losses, with lines broken at stage
  boundaries.
- `sequence_preview.gif`: intentionally slowed to 10 displayed frames/s;
  left=target, right=reconstruction, depths stacked vertically. It is not PLM
  playback or a perceptual color calibration.
- `evaluation.json`: fresh reconstruction agreement and re-evaluated metrics.

PSNR uses a fixed peak intensity of 1. Temporal error-difference RMS measures
changes in reconstruction error, not absolute image motion or a complete human
vision score. Evaluate uses embedded target, LUT, and trajectory data, so replacing
the external LUT later cannot silently change an old run.

For the physics-to-code map, mathematical definitions, tests, and reference
details, see `docs/DESIGN.md` and `docs/REFERENCES.md`. Recorded verification and
benchmark findings are in `docs/VALIDATION.md` after the implementation run.
