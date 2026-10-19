# Holospeed alignment audit — 2026-10-19

This audit distinguishes a shared mathematical objective from a full experimental
reproduction. The supplied 12-page Holospeed PDF was extracted and relevant pages
4, 6 and 10 visually checked. Sections 4–5, equations 12–17, and Figure 8 on page 10
provide the primary comparison. The authors' project and supplement were also
consulted:

- https://dorianchan.com/holospeed/
- https://dorianchan.com/holospeed/mainpaper.pdf
- https://dorianchan.com/holospeed/supplement.pdf

Local paper: `/Users/zack/Desktop/PhD/paper/Holospeed- High-Speed Holographic Displays for Dynamic Content.pdf`.

## The target is perceived, not just the reconstruction

Equation 12 defines a perception operator P: translate frames according to eye
motion, then integrate them with a temporal kernel. Equation 13 compares
P(reconstructed video, eye motion) with P(desired high-speed video, eye motion).
Applying the same operator to the target is therefore intentional and supported
by the paper. Replacing the perceived target unconditionally with the latest
sharp display-space frame would change that objective.

For a fixed eye, a translating object naturally spreads out in the temporal
integral. For an eye following the object, the object is approximately stationary
in retinal coordinates, so integration need not cause the same motion smear.
This does not guarantee perfectly sharp discretized reconstructions: sampling,
interpolation, speckle, and optimization still matter. The objective is correct
perception for the chosen eye trajectory, not universally sharp moving imagery.

`TemporalModel.perceive` is applied to both target and reconstruction.
`EyeMotion.warp` samples at x + eye_i - eye_endpoint. Thus an object moving with
eye_i is aligned to its endpoint position. The relative endpoint convention
differs from a fixed retinal origin by a common translation; shared valid masks
exclude samples outside the finite image. The same masks are used in the loss.

`motion.enabled=false` reduces this to a fixed-eye model. The small
`stride_validation` runs had motion disabled and are functional checks, not
demonstrations of Holospeed's motion-aware quality. They cannot support a claim
that tracking a moving object remains sharp.

## What matches and what differs

| Item | Current implementation / qualification |
|---|---|
| Temporal accumulation | Exposure-weighted intensities; paper's section 5 uses a rectangular 16.66 ms kernel. Our 24 updates at 1440 Hz span 16.67 ms. |
| Target processing | Same temporal and eye-motion operator as reconstruction, consistent with equations 12–13. |
| Known-motion objective | Supported as a single deterministic supplied/synthetic eye trajectory. |
| Full-sequence optimization | All phases updated together, corresponding to the direct approach discussed after equation 13; our RGB/depth objective is an adaptation. |
| Sequential optimization | With null stride, newest physical phase only and fixed detached history: deterministic counterpart of equation 17. |
| High-speed regularization | Null stride scores every physical endpoint. Integer cycle strides score fewer endpoints; evaluation at all endpoints does not replace missing optimization constraints. |
| W=8, S=4 | Twelve physical updates per step in RGB: half-window endpoint spacing. Figure 8 includes a half-window regularization comparison, but our first-window joint initialization and new-block commits are not equation 17's single-pattern procedure. |
| Motion uncertainty | Paper equations 14 and 15 cover stochastic motion distributions and kernel-based alternatives. Neither is implemented here. |
| Fixed-group baseline | Our joint mode fits an integrated dynamic target within each group. Paper's conventional baseline fits one low-rate target per group. These coincide for a static target but are not the same dynamic baseline. |
| RGB hardware | Our wavelength-specific GRB sequence with fixed source gains is a lab-oriented extension. Paper section 5 uses 520 nm and synthesizes color results from single-wavelength captures. |
| Optics and scene | Our uncalibrated ASM, tutorial depths and prescribed focal stacks are not the paper's calibrated forward model or rendered scenes. |
| Quantization | Our circular LUT softmax surrogate is different. Supplement section 6 cites Agustsson et al. soft rounding and anneals its strength; it is not accurate to call the authors' method wholly unspecified. |
| Startup / finite images | Actual-duration prefix normalization and shared valid-border masks are our explicit numerical conventions. Do not claim exact paper startup or boundary reproduction. |

The paper also explores motion-unaware high-speed regularization (Figure 8).
Thus fixed-eye experiments are useful baselines, but should be labeled as such.
Changing stride and turning off eye motion must not silently be described as a
full implementation of the strongest motion-aware method.

## Explicit known-tracking profiles

`configs/holospeed_known_motion_causal.json` and
`configs/holospeed_known_motion_global.json` use the existing deterministic motion
path and null stride, with RGB and three depths retained as our extension.
They do not change previous profiles or previously saved runs.

For the supplied 64x64 moving synthetic target, the translation between first
and last samples is 0.45*(64-1)/2 = 14.175 pixels. Across 48 equally spaced
samples at 1440 Hz, the elapsed sample time is 47/1440 s. Thus the matching
horizontal tracking velocity is 434.2978723404255 pixels/s. This is exact scene
metadata, not measured eye motion or optical-flow estimation. All three planes
share this rigid translation, making one trajectory appropriate for this example.

If crop width, number of samples, rate, exposure timing, or scene motion changes,
recompute the trajectory or supply a timestamped motion file. Do not reuse this
numeric speed blindly for other scenes. For equal exposures and this synthetic
generator, velocity = 0.45*(width-1)/2 / ((number_of_samples-1)/rate_hz).

These profiles retain the library's 200 updates and LUT surrogate, rather than
silently claiming the paper's usual 500-update calibrated experiment. The
per-physical-frame and tracking settings restore the relevant objective and
update structure within the declared deterministic model; stochastic/kernel
methods, the referenced rounding function and calibrated hardware remain future
alignment tasks.

## Checks performed

All 38 CPU tests passed after adding the audit profiles and two targeted tests
(`.cache/holospeed_audit/test_results.txt`, 16.816 s). A six-frame translating
point target on three planes produces six positions in the fixed-eye integral,
but a single unit-height point after matched eye tracking on the shared valid
region. This verifies alignment direction and that perceived-target integration
does not inherently require motion smear. The profile trajectories also match
the synthetic scene's analytic translation, and both select every physical
endpoint. These checks establish the deterministic model's logic; the new
200-update 64x64 profiles have not been run as a quality benchmark in this audit.
