# Validation and acceptance

## Purpose

No single test proves that a tightly focused field is correct. SCPIC uses a
validation ladder in which each stage isolates a different failure mode:

```text
units and phasor convention
        |
        v
analytic field and geometry identities
        |
        v
surface and spectrum convergence
        |
        v
Maxwell and energy diagnostics
        |
        v
published paper benchmarks
        |
        v
EPOCH file-reader and propagation tests
        |
        v
campaign-specific convergence and physics checks
```

## 1. Automated unit and regression tests

Run:

```bash
.venv/bin/python -m pytest -q
```

The suite covers:

- 2D and 3D mirror geometry and quadrature;
- incident propagation and polarisation directions;
- PEC physical-optics field reconstruction;
- TM01 component formulae and effective area;
- finite-Rayleigh TM01 reduction at the waist;
- Gauss--Laguerre power conservation;
- wavelength-to-frequency Jacobians and spectral phase;
- Jolly chromatic tilt, curvature, and trefoil phase scaling;
- position--frequency phase application by every 3D incident-field class;
- carrier-referenced envelope reconstruction for EPOCH;
- broadband energy recovery;
- chunked reconstruction equivalence;
- EPOCH phase conversion, precision, shape, and ordering;
- continuous unwrapped phase for linear interpolation;
- low-amplitude phase-tail regularisation;
- focus-defined Gaussian-pulse duration and boundary group delay in 2D;
- vectorised/chunked and threaded 2D propagation equivalence;
- monochromatic and time-domain Maxwell equations;
- electromagnetic energy and Poynting-flux factors;
- Vallières focal-profile regressions;
- Fourmaux transmission-parabola geometry.

Passing these tests detects implementation regressions. It does not replace
convergence for a new geometry or injection plane.

## 2. Numerical convergence

### Surface quadrature

Use at least three increasing pairs of radial and azimuthal order. A typical
sequence is:

```python
orders = ((12, 24), (18, 36), (24, 48), (32, 64))
```

Evaluate the complete quantity used by EPOCH, preferably the full complex
injection plane at the shortest wavelength in the retained spectrum.

The `surface_quadrature_convergence()` stopping metric is:

\[
\epsilon_S =
\frac{\lVert(\Delta\mathbf E,c\Delta\mathbf B)\rVert_2}
{\lVert(\mathbf E,c\mathbf B)\rVert_2}.
\]

Suggested interpretation:

| Relative change | Interpretation |
|---:|---|
| \(>10^{-2}\) | Insufficient |
| \(10^{-3}\) to \(10^{-2}\) | Screening or exploratory work |
| \(10^{-4}\) to \(10^{-3}\) | Usually adequate for profile campaigns |
| \(<10^{-4}\) | High-accuracy optical comparison, if other grids also converge |

These are operational suggestions, not universal error bounds. A local norm
can hide a small but physics-critical component; separately inspect peak
field, phase, longitudinal fraction, focal width, and integrated flux.

### Spectral convergence

Refine both the component count and the retained frequency extent. Verify:

- recovered incident energy;
- peak time and peak intensity;
- pulse duration;
- focal width at the pulse peak;
- phase-sensitive observables;
- absence of periodic replicas in the requested time window.

For EPOCH spatiotemporal data, converge the carrier-referenced envelope rather
than sampling the full analytic carrier. Verify algebraically or numerically
that:

\[
\widetilde{\mathbf E}_{\mathrm{envelope}}(t)e^{-i\omega_ct}
=\widetilde{\mathbf E}_{\mathrm{analytic}}(t).
\]

The period is \(2\pi/\Delta\omega\). The requested time span must be shorter
than this period.

For a non-separable spatio-spectral phase, additionally verify:

- the phase and group-delay maps on the input pupil;
- individual frequency-component centroids and focal positions;
- every vector component at several longitudinal planes;
- convergence of the STC effect itself, not only total peak intensity;
- recovery of the uncoupled result when all chromatic coefficients vanish.

### Observation grid

Refine spacing and extent independently. A grid can resolve a central lobe
while still truncate annular rings or longitudinal tails.

For energy-volume calculations, increase the volume until the integrated
energy stops changing at the required tolerance. Dumont found that extents of
roughly 25 wavelengths could be required, but this is only a starting point.
When the volume is enlarged, reconverge the mirror quadrature.

## 3. Maxwell diagnostics

For monochromatic phasors, `maxwell_residuals()` checks:

\[
\nabla\cdot\mathbf E=0,\quad
\nabla\cdot\mathbf B=0,\quad
\nabla\times\mathbf E-i\omega\mathbf B=0,\quad
\nabla\times\mathbf B+\frac{i\omega}{c^2}\mathbf E=0.
\]

For sampled broadband fields, `time_domain_maxwell_residuals()` checks:

\[
\nabla\cdot\mathbf E=0,\quad
\nabla\cdot\mathbf B=0,\quad
\nabla\times\mathbf E+\partial_t\mathbf B=0,\quad
\nabla\times\mathbf B-\frac{1}{c^2}\partial_t\mathbf E=0.
\]

Residuals are normalised RMS quantities. They combine:

- Stratton--Chu quadrature error;
- observation-grid finite-difference error;
- time-grid finite-difference error;
- boundary trimming choices.

Refine these contributions separately before assigning physical meaning to a
residual. The present small direct Stratton--Chu regression has all four
monochromatic residuals below 3%; the analytic plane-wave tests are much
tighter.

## 4. Energy and flux

For complex phasors or analytic-signal envelopes, the default cycle-averaged
energy density is:

\[
\langle u\rangle =
\frac{1}{4}\left(\epsilon_0|\mathbf E|^2
+\frac{|\mathbf B|^2}{\mu_0}\right).
\]

The cycle-averaged Poynting vector is:

\[
\langle\mathbf S\rangle =
\frac{1}{2\mu_0}\Re(\mathbf E\times\mathbf B^*).
\]

Use `integrated_field_energy()` for a volume and
`integrated_poynting_flux()` for a plane. A signed flux depends on the plane
normal.

For a lossless perfect-conductor calculation, discrepancies can arise from:

- finite aperture clipping;
- an observation plane or volume that does not intercept all energy;
- insufficient surface quadrature;
- insufficient transverse resolution or extent;
- inconsistent incident-field effective area;
- confusing analytic-signal, phasor, and instantaneous factors.

Finite coating losses must be applied separately.

## 5. Published benchmarks

### Vallières et al. 2023

The fine benchmark results are recorded in
[methodology_3d.md](methodology_3d.md). Current agreement is:

- all focal widths and Rayleigh lengths within 2.6%;
- linear peak intensities within 4.3%;
- TM01 peak intensities within 8.3%;
- numerical refinement below 0.12% for the reported fine comparison.

The remaining TM01 peak difference is therefore not explained by the tested
surface, spectrum, or profile-grid refinements. Do not compensate new TM01
results by multiplying them by the paper-to-code ratio.

### Dumont et al. 2017

The implementation matches the supplement's analytic effective-area ratios,
finite-Rayleigh TM01 waist limit, Gauss--Laguerre mode power, broadband
normalisation, and observation-domain decomposition.

### Fourmaux et al. 2025

The published geometry is reproduced:

- parent focal length: 5.65 mm;
- illuminated diameter: 65 mm;
- central opening: 24.5 mm;
- ray angles: 38.3--85.4 degrees;
- generalised numerical aperture: approximately 0.96.

The measured phase maps and fitted Zernike coefficients are unavailable, so
the reported 6% uncorrected and 68.1% corrected peak-intensity ratios are not
independent regression targets.

## 6. EPOCH integration

Run the local tests described in [the EPOCH guide](../epoch_tests/README.md):

```bash
.venv/bin/python epoch_tests/run_local.py
.venv/bin/python epoch_tests/analyse_local.py
.venv/bin/python epoch_tests/run_local_3d.py
.venv/bin/python epoch_tests/analyse_local_3d.py
```

The staged acceptance is:

1. Static profile: precision, ordering, interpolation, amplitude scale, and
   constant phase offset.
2. Phase ramp or tilt: sign-sensitive transverse Poynting direction.
3. Full SCPIC focus: complete mirror-to-boundary-to-propagation path.

A broadband spatiotemporal production test must additionally confirm that:

- the file contains the carrier-referenced envelope;
- EPOCH's `lambda` matches the selected carrier;
- `t_start`, `t_end`, `n_t`, and transverse axes match the file;
- an additional `t_profile` is omitted unless intentional;
- the carrier has not been included in the file phase.

For `generate_epoch2d_oap_pulse()`, also converge mirror points, retained
spectral count/span and the electric-field derivative step. Confirm that the
manifest's boundary arrival equals the requested focus time minus the vacuum
flight time within the temporal sampling error. Compare EPOCH's focal waist
and longitudinal/transverse ratio with the direct carrier reference stored in
the manifest; agreement with the boundary file alone does not test
`simple_laser` reconstruction.

For static and spatiotemporal files, inspect the maximum phase difference
between adjacent samples. A jump close to \(2\pi\) usually indicates a wrapped
branch cut that EPOCH will interpolate incorrectly. Genuine optical
singularities require a grid and branch-cut treatment appropriate to the
physical field; phase at exactly zero amplitude is undefined.

`epoch_phase_diagnostics()` automates a first check. A nonzero winding-cell
count means that a reliable two-dimensional cell contains phase circulation
and no global scalar unwrapping exists. This is particularly relevant to the
space-time vortices that can arise from tightly focused chromatic coupling.

For a production injection:

- repeat with the production EPOCH grid and timestep;
- compare multiple downstream planes;
- compare complex transverse field where possible, not only intensity FWHM;
- check propagation direction and \(E/B\);
- quantify the error caused by omitting the longitudinal electric field;
- use an interior antenna if that error affects the intended physics.

## 7. Suggested campaign acceptance record

Archive one machine-readable record per profile containing:

```text
code commit
mirror class and all dimensions
surface quadrature orders
incident field class and parameters
spectrum samples, total energy and spectral phase
wavefront source, units, sign, pupil mapping and preprocessing
coating throughput assumption
observation-plane coordinates and ordering
time samples and discrete period
surface and spectral convergence results
Maxwell residuals
energy or Poynting-flux result
exported field_scale and phase_reference
EPOCH deck, executable commit and grid
downstream EPOCH comparison metrics
```

This information is more important for reproducibility than retaining only
the final amplitude and phase files.

## 8. Current acceptance boundary

The optical field generator is suitable for controlled scientific use.
Production confidence still depends on the injection mechanism:

- moderate-angle transverse fields can be tested through `simple_laser`;
- near-unity-NA fields with a strong longitudinal component require a
  quantified boundary approximation;
- exact full-vector injection ultimately requires an EPOCH-side interior
  current sheet or equivalent antenna.
