# PLM holography research library

Self-contained ASM simulations for a phase-only, 4-bit PLM: RGB focal stacks,
incoherent temporal integration, global and causal sequence optimization, and an
optional eye-motion model. The original repository is preserved as a reference;
this package does not import its code or hardware dependencies.

## Run

Use the existing project environment. No installation is required:

```bash
cd /Users/zack/Computing_PhD/neural-3d-holography/plm_holography
../.venv/bin/python -m unittest discover -s tests -v
../.venv/bin/python -m plmholo optimize --config configs/smoke.json
../.venv/bin/python -m plmholo evaluate --run results/smoke
```

The smoke configuration is a small integration check. The full CPU comparison is:

```bash
../.venv/bin/python -m plmholo benchmark --config configs/benchmark_cpu.json
```

This runs 24 experiments: two scenes, three paired seeds, and four temporal
methods. Each uses 64x64 phases, three depths, 48 physical RGB subframes, an
overlapping 24-subframe window, and 200 Adam updates per phase exposure.
`benchmark --resume --config ...` skips completed runs only when configurations
match. Use a new output path when changing any settings. A normal `optimize`
command refuses to overwrite a completed run.

## Main settings

Configuration files are JSON and can `extend` another JSON file using the
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
- **Global:** update all phases together using every overlapping window endpoint.
  It has access to the whole target sequence.
- **Causal:** at each physical update, optimize only the newest phase; cache and
  freeze all previously committed intensities. Only targets up to that update
  enter the objective. It cannot revise frames already displayed.

There is no claim that these objectives have equal information access or that
one method always wins. The benchmark pairs initial phases and seeds, fixes the
optical model and brightness, and reports actual propagation counts and runtime.

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
errors participate in global/causal optimization but are reported separately;
the main benchmark score uses complete windows. No end-to-start wrapping occurs.
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
not a claimed reproduction of Holospeed's undocumented supplementary details.

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
