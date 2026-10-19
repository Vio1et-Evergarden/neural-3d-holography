# Sources and provenance

The supplied local references were inspected during planning. Their content is
research background; laboratory procedure text is not an instruction to operate
hardware or change the machine.

1. **Neural 3D Holography: Learning Accurate Wave Propagation Models for 3D
   Holographic Virtual and Augmented Reality Displays**, 2021. See sections 3.3
   and 3.4 for the distinction between forward-model training and phase
   optimization. Its RGB-D masked supervision is different from this release's
   full focal-stack targets. Local file:
   `/Users/zack/Desktop/PhD/paper/Neural 3D Holography- Learning Accurate Wave Propagation Models for 3D Holographic Virtual and Augmented Reality Displays.pdf`.

2. **Holospeed: High-Speed Holographic Displays for Dynamic Content**, 2025.
   Sections 4 and 4.2 motivate temporal perception, frequent overlapping-window
   constraints and fixed-history sequential optimization. Equation 17 freezes
   past frames and optimizes the newest frame. Section 5 describes a 4-bit,
   1440-Hz PLM and soft rounding. Supplement section 6 identifies Agustsson et al.'s
   soft-rounding function and annealing of its strength. Our circular LUT softmax
   surrogate is different, not an exact replica of that referenced method.
   Motion uncertainty and optical-flow estimation are outside this release.
   Local file:
   `/Users/zack/Desktop/PhD/paper/Holospeed- High-Speed Holographic Displays for Dynamic Content.pdf`.
   Author supplement: https://dorianchan.com/holospeed/supplement.pdf.
   Equation-by-equation correspondence and remaining differences are recorded
   in `HOLOSPEED_ALIGNMENT.md`.

3. **Laboratory DMD PLM notes**. Pages 5, 16, 19, 25-26 and 49 inform code/state
   separation, eight frames per color, 10.8 um pitch, wavelength examples and
   replaceable measured LUTs. The document records evolving calibration and
   timing configurations. It does not supply a verified machine-readable LUT
   here. No graph is digitized and treated as a measurement. Default optical
   profiles and color order remain editable simulation choices. Local file:
   `/Users/zack/Desktop/DMD PLM.pdf`.

The parent Stanford repository is distributed under the MIT license. The new
implementation is self-contained; the retained upstream notice in `../LICENSE` acknowledges
the reference implementation and any derived numerical conventions.
