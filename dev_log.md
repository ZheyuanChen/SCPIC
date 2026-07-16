# SCPIC Developer's Log

This log documents the design decisions, mathematical formulations, software engineering milestones, and physical validations for the **SCPIC** (Stratton-Chu Particle-in-Cell) preprocessing package.

---

## Phase 1: Project Initialization & Core Physics Models

### 1. Mathematical Formulation (2D TM Mode)
The goal is to calculate the electromagnetic fields focused by an Off-Axis Parabolic (OAP) mirror to inject them as a boundary source into the EPOCH PIC code. 

For a 2D Transverse Magnetic (TM) wave with exp(-i*ω*t) time dependence, the magnetic field has only a y-component, **B** = B_y(x, z) **ŷ**, and the electric fields lie in the xz-plane:
*   E_x = -(i * c / k) * (∂B_y / ∂z)
*   E_z =  (i * c / k) * (∂B_y / ∂x)

The focused magnetic field B_y(r_0) at an observation point r_0 is calculated using the 2D Green's function representation of the Stratton-Chu boundary integral over the mirror surface S:

$$B_y(\mathbf{r}_0) = \int_{S} \left[ B_y(\mathbf{r}) \frac{\partial G(\mathbf{r}, \mathbf{r}_0)}{\partial n} - G(\mathbf{r}, \mathbf{r}_0) \frac{\partial B_y(\mathbf{r})}{\partial n} \right] dl$$

Where:
*   `G(r, r_0) = (i/4) * H_0^(1)(k * |r - r_0|)` is the 2D free-space Green's function, where H_0^(1) is the Hankel function of the first kind of order 0.
*   `∂/∂n = n · ∇` is the normal derivative pointing outward from the mirror surface.
*   `dl` is the infinitesimal line element along the mirror profile.

### 2. Geometry Module (`mirrors.py`)
Implemented the `ParabolicMirror2D` class to handle standard parent parabola calculations and support 90-degree off-axis parabolas (OAP90).
*   Generates the 2D spatial coordinates (x_m, z_m) of the reflective surface.
*   Computes analytical unit normal vectors (n_x, n_z) pointing toward the focal region.
*   Extracts local differential line elements `dl = sqrt(dx^2 + dz^2)` for numerical integration.

### 3. Incident Field Module (`fields.py`)
Implemented `IncidentFieldTM` to model the incoming laser beam travelling along the -z-axis.
*   Models a 2D Gaussian spatial profile: 

$$B_y(x, z) = B_0 \exp\left(-\frac{(x - x_c)^2}{w_0^2}\right) e^{-i k z}$$

*   Calculates analytical spatial derivatives to provide `∂B_y/∂n = n_x * ∂B_y/∂x + n_z * ∂B_y/∂z` directly on the mirror surface, eliminating numerical error on the boundary.

---

## Phase 2: Numerical Integration Core (`solvers.py`)

*   Implemented the Stratton-Chu boundary integrator `evaluate_SC_2D` using SciPy's Hankel functions.
*   The reflecting-mirror path now applies the PEC physical-optics boundary values, $B_{y,\mathrm{surface}} = 2B_{y,\mathrm{inc}}$ and $\partial B_y/\partial n = 0$. The earlier implementation inserted incident values into a generic Kirchhoff representation and mostly reconstructed the incoming field rather than the reflected field.
*   Mirror integration uses trapezoidal arc-length weights. The solver is currently a clear serial reference implementation; Numba acceleration remains future work rather than a completed feature.

---

## Phase 3: Binary Export Pipeline (`export.py`)

*   **Format Shift (HDF5 to Raw Binary):** Transitioned the output module from complex HDF5 files to raw C-style binary streams using NumPy's `ndarray.tofile()`. This eliminates library overhead and directly interfaces with custom Fortran input routines in modified EPOCH setups.
*   **Amplitude/Phase Splitting:** The single transverse electric phasor required by an EPOCH laser block is written as normalised `amplitude.dat` and `phase.dat` streams. EPOCH constructs the associated magnetic field; `Ex/Ez/By` volume dumps are not a supported laser input format.
*   **Phase conversion:** SCPIC uses $\mathrm{Re}[F e^{-i\omega t}]$, while EPOCH-mod uses $A\sin(\omega t + \phi)$. The exporter therefore writes $\phi_{\mathrm{EPOCH}} = \pi/2 - \arg(F)$, with an optional global phase reference.
*   **Precision and ordering:** `epoch_dev` hard-codes `REAL(num)` to eight bytes. Files are headerless native `float64` streams in C order; the NumPy axis order is reversed relative to the corresponding Fortran declaration, so no transpose is needed.

---

## Phase 4: Build System & Workspace Restructuring

To resolve Hatchling build-backend compilation failures (`ValueError` and package autodetection mismatches) during editable pip installations, the repository workspace was explicitly structured.

```text
SCPIC/
├── src/
│   └── scpic/
│       ├── __init__.py
│       ├── export.py
│       ├── fields.py
│       ├── mirrors.py
│       └── solvers.py
├── tests/
│   └── test_scpic.py
├── pyproject.toml
├── README.md
├── dev_log.md
└── main.py
```
### Key Packaging Actions:
* Added an empty `src/scpic/__init__.py` to mark the folder as an importable module.
* Declared explicit wheel target pathways in `pyproject.toml` to bypass fragile auto-discovery heuristics:

  [tool.hatch.build.targets.wheel]
  packages = ["src/scpic"]

* Separated development tools (such as `pytest` and `black`) into an optional dependency block (`[project.optional-dependencies] dev`) to ensure clean production builds.

---

## Phase 5: Physical Validation & Benchmarking (`benchmark.py`)

To verify the physical and mathematical correctness of our Stratton-Chu solver before running simulations inside EPOCH, we designed and executed a dual-regime validation framework.

                  Incident Gaussian Beam
                  (Propagating along -z)
                            │
                            ▼
                /───────────────────────\  OAP90 Mirror
                |                       |  (Rotates beam by 90 deg)
                \───────────────────────/
                            │
                            ▼
                   Focused Beam Profile
                  (Propagating along -x)
                            │
                            ▼
                [ FOCAL OBSERVATION GRID ]
                 Ex: Longitudinal Field
                 Ez: Transverse Field

### 1. Paraxial Regime (Weak Focusing Validation)
* Configuration: Focal length f_0 = 200 μm, mirror diameter D = 120 μm, and incident beam waist w_inc = 20 μm. The six-waist aperture suppresses clipping so that the untruncated Gaussian formula is a valid reference.
* Physical Metric: In this paraxial limit, the focal spot waist must match the analytical 2D paraxial Gaussian beam waist formula:

  w_focus ≈ (2 * lambda * f_0) / (pi * w_inc)

* Result: Fitted a Gaussian profile to the transverse E_z field slice at x = 0. The corrected PEC physical-optics solution agrees with the analytical waist to approximately 0.03%. The original configuration clipped the beam strongly and, combined with the incorrect boundary model, actually gave a 247% error; the earlier <2% entry was not reproducible.

### 2. Tight-Focusing Regime (Non-Paraxial Validation)
* Configuration: Focal length f_0 = 10 μm, mirror diameter D = 20 μm, and incident beam waist w_inc = 8 μm (High Numerical Aperture).
* Physical Signatures Verified:
  1. Longitudinal Field Coupling (E_x): The longitudinal field component E_x (which is negligible in paraxial approximations) grew to a significant fraction (~10%+) of the primary transverse field E_z.
  2. Antisymmetric Phase Signature: Since E_x is proportional to ∂B_y/∂z, and B_y peaks symmetrically at z = 0, the longitudinal field profile exhibited a perfect antisymmetric double-lobed structure with a deep null at the coordinate center.

---

## Phase 6: Automated Regression Framework (`tests/`)

* Implemented continuous verification via unit tests (`tests/test_scpic.py`) targeting mirror geometry and quadrature, incident-wave direction, Maxwell field recovery, exact time-domain reconstruction of the EPOCH phase convention, float64 stream size and ordering, invalid export parameters, the PEC paraxial waist, and generic Kirchhoff input validation.

---

## Phase 7: EPOCH-mod Compatibility and Local Integration (July 2026)

The exporter was checked directly against `epoch2d/src/user_interaction/custom_laser.f90`, `epoch2d/src/laser.f90`, and `DOCUMENTATION_LASER_INJECTION.tex` in the local `epoch_dev` branch.

Three local cases now live under `epoch_tests/`: a static Gaussian, a phase-ramp sign test, and the full OAP90 focus. All three load and run successfully with both one and two MPI ranks. Quantitative checks on the generated SDF files give:

* static Gaussian field waist: 2.121 μm for a 2.0 μm injected waist; median $E_y/B_z = 3.0394\times10^8$ m/s;
* phase-ramp integrated Poynting angle: +11.08° for a +10° target;
* f/1 focused waist at x ≈ 0: 1.126 μm in EPOCH versus 0.944 μm in SCPIC, a 19.3% difference on the initial coarse grid.

The final result is deliberately recorded as a limitation, not a pass at production accuracy. EPOCH's `simple_laser` boundary cannot impose the longitudinal electric component and approximates high-angle content. A grid-convergence and downstream complex-field comparison are required before physics production; an interior current-sheet injector is the preferred fallback if this boundary error remains material.

---

## Phase 8: Full 3D Vector Solver and Paper Reproduction (July 2026)

The implementation was extended from its 2D TM reduction to the vector
physical-optics Stratton--Chu method used by Vallières *et al.*:

* `ParabolicMirror3D` represents projected circular and annular cuts of the
  parent paraboloid, including the paper's 220 mm OAP90. It generates
  Gauss--Legendre surface and oriented-rim quadratures.
* `LinearPolarisedSuperGaussian3D` evaluates consistent incident E and B
  phasors for arbitrary orthogonal propagation and polarisation vectors.
* `evaluate_SC_3D` returns all three reflected E and B components, restores
  the SI factor of c suppressed in the papers' natural-unit equations, chunks
  observation points, and optionally retains Dumont *et al.*'s contour term.
* `SuperGaussianSpectrum` implements the order-seven, 90 nm FWHM spectrum,
  the paper's discrete-frequency pulse reconstruction, and 20 J energy
  normalisation.

`paper_benchmark_3d.py` now reproduces the linearly polarised OAP90 entry of
Table 1. Workstation defaults give 2.663×10²³ W/cm² peak intensity, 0.600 µm
meridional FWHM, 0.503 µm sagittal FWHM and 0.650 µm Rayleigh length, compared
with 2.66×10²³ W/cm², 0.600 µm, 0.520 µm and 0.690 µm in the paper.

Three EPOCH3D tests were also added and run locally. The static astigmatic
profile and two-axis phase tilt verify raw stream ordering and signs. The full
OAP case propagates to 1.187 µm FWHM on both axes in EPOCH, compared with
direct SCPIC references of 1.247 µm and 1.249 µm. These remain coarse-grid
boundary-injection tests rather than production convergence evidence.

The literature review in `docs/literature_review.md` records four close prior
implementations: the in-house StrattoCalculator, Nielsen's public MIT-licensed
C++/CUDA vector-integral solver, Bulanov *et al.*'s recent multi-mirror
library, and Popov's unrelated 2009 3D PIC code also named SCPIC.

---

## Phase 9: Wavefronts, Radial Polarisation and Independent Solver Checks (July 2026)

Nielsen's released C++/CUDA diffraction code was inspected at source level
before deciding whether to adopt it. Its specialised electric-field
components agree algebraically with SCPIC's general physical-optics integral
for a centred paraboloid. The active release is not a suitable project base,
however: its OAP offset and Zernike paths are disabled, its incident profile
and grid are compile-time definitions, two transverse magnetic components
omit the observation-plane z coordinate, and the generated field files do not
include the header required by its own loader. No Nielsen source was copied.

The useful functionality was instead implemented within SCPIC's existing API:

* `ZernikeWavefront` represents orthonormal OSA/ANSI modes with coefficients
  in metres of RMS optical path difference. Arbitrary measured wavefronts can
  be supplied as callables returning OPD at the mirror points. The phase is
  evaluated as `k * OPD` separately for every spectral component.
* `RadiallyPolarisedSuperGaussian3D` supplies the local transverse radial
  electric direction and a consistent magnetic field. Its on-axis value is
  zero to regularise the otherwise undefined direction at that measure-zero
  point. An axisymmetric HNAP regression converges between 12×24 and 20×40
  surface quadratures and produces the expected purely longitudinal electric
  field at the focus.
* `maxwell_residuals` checks the two divergence and two curl equations for
  `exp(-i omega t)` phasors on a regular 3D grid. A plane-wave unit test gives
  curl residuals below 0.3%; the small directly propagated HNAP regression has
  all four normalised residuals below 3%.
* A corrected specialised paraboloid component expansion, derived
  independently from the published formulation, matches the general vector
  integrand to floating-point precision and guards against the missing-z
  magnetic-field error found in the Nielsen release.
* `evaluate_SC_3D(..., backend="cupy")` is now an optional GPU execution path.
  It keeps surface data resident on the device and transfers observation
  chunks, while returning NumPy arrays and retaining NumPy as the canonical
  implementation. This workstation has no CUDA runtime, so execution and
  performance parity on a GPU remain to be verified before production use.

The direct `pytest` console entry point now includes the repository root via
the project configuration, matching `python -m pytest`. At this stage the
paper's radial and HNAP/TP cases still required converged benchmarks; the
remaining performance work was a NumPy/CuPy comparison on CUDA hardware.

---

## Phase 10: Complete Vallières Benchmark Suite (July 2026)

The outstanding paper reproduction has now been completed for all six entries
in Vallières *et al.* Tables 1--2:

* `TM01RadiallyPolarisedBeam3D` implements Eqs. (16)--(17), including the
  on-axis longitudinal electric field and the azimuthal magnetic field. This
  is distinct from the retained reduced transverse radial-envelope class.
* Broadband TM01 energy normalisation now uses the exact frequency-dependent
  longitudinal-flux area, `pi/k_n**2`; the spectrum API accepts either a
  scalar area or one area per spectral component and recovers 20 J in both
  cases.
* Mirror factories now encode the paper's 58 mm HNAP and the optimal 20% TP.
  The latter derives an 89.44 mm opening and 20 mm parent focal length from
  the paper's geometry equations instead of embedding rounded values.
* `paper_benchmark_3d.py` evaluates linear or TM01 input on HNAP, OAP90 or TP,
  provides a six-case suite, and performs independent surface and spectral
  convergence studies. The incident intensity FWHM is the paper's 200 mm;
  the mirror outer diameter remains 220 mm.

At 32×64 surface nodes, 47 frequencies and 161 profile samples, every focal
width and Rayleigh length is within 2.6% of the published tables. Linear peak
intensities are within 4.3%, while TM01 peak intensities are 4.8--8.3% below
the paper. Refining 24×48/31/121 to 32×64/47/161 changes any computed quantity
by no more than 0.12%, so the remaining TM01 peak discrepancy is not a
quadrature, spectral or profile-grid convergence artefact.

Six reduced-resolution paper regressions have been added to the test suite,
alongside direct tests of the TM01 equations, mirror dimensions and broadband
energy recovery. The standalone 2D TM reduction and the EPOCH export paths are
unchanged by this phase.

---

## Phase 11: Dumont Pulse Extensions and Experimental TP Geometry (July 2026)

Dumont *et al.* (2017), including its four-page supplementary material, was
revisited as an implementation source rather than only a derivation reference.
The continuous-frequency normalization with an explicit frequency increment
was compared against SCPIC's periodic discrete representation. The two are
equivalent when `T = 2*pi/delta_omega` is used consistently. Dumont's analytic
linear and radial energy factors also give the same relative effective areas
already implemented in SCPIC. This independently supports the existing TM01
normalization and does not explain the remaining 5--8% Vallières radial peak
difference.

The following additions were made without changing Vallières-compatible
defaults:

* `SuperGaussianSpectrum` now carries arbitrary spectral phase and can return
  complex component coefficients. Its default narrow-band wavelength
  conversion is unchanged, while `conversion="exact_wavelength_density"`
  applies the exact `|dlambda/domega|` Jacobian for large bandwidths.
* `SampledSpectrum` accepts measured energy density in either angular
  frequency or wavelength, unwraps/interpolates spectral phase, and resamples
  onto the uniform angular-frequency grid required by reconstruction. The
  period, optical-carrier Nyquist step and explicit time-grid validation are
  exposed.
* `FiniteRayleighTM01Beam3D` implements Dumont's complex-q radially polarised
  Gaussian, including curvature and the longitudinal electric field. It is
  numerically identical to `TM01RadiallyPolarisedBeam3D` at the waist.
* `ParaxialGaussLaguerreBeam3D` implements the axisymmetric radial-mode
  expansion of the supplement for linearly polarised, well-collimated input.
  Complex modal coefficients and the orthogonality-derived effective area are
  supported. Its paraxial limitation is explicit.
* `iter_broadband_field_chunks` limits spectral working storage to one
  observation chunk. Supplying an mpi4py communicator partitions observations
  into balanced contiguous rank-local ranges without forcing a collective
  output format. `paper_benchmark_3d.py` now uses this path.
* `surface_quadrature_convergence` refines the mirror against the complete
  observation set and uses a combined `(E,cB)` relative norm. It is intended
  to be run at the shortest relevant wavelength on the actual EPOCH injection
  plane, where the near-focus phase cancellation may be weaker.
* Diagnostics now cover time-domain Maxwell residuals, electromagnetic energy
  density, rectilinear volume energy, signed plane Poynting flux, and relative
  energy error. A production energy check must converge both the integration
  volume and mirror quadrature; Dumont's roughly 25-wavelength extent is a
  starting point rather than a hard universal value.

The 2025 Fourmaux *et al.* paper, *High peak intensity characterization and
optimization with a tight-focusing transmission parabola*, was also reviewed
from the supplied five-page article. `ParabolicMirror3D.fourmaux_tp_2025()`
encodes its 5.65 mm parent focal length, 65 mm illuminated diameter, and
24.5 mm central opening. The derived 38.3--85.4 degree ray-angle interval
matches the publication and corresponds to generalized-solid-angle NA 0.96.

The experiment is strong evidence for the measured-OPD workflow: deformable-
mirror correction changed the measured wavefront from 9.3 wavelengths
peak-to-valley and 2.13 wavelengths RMS to 1.02 and 0.16 wavelengths, and the
authors' Stratton--Chu peak rose from 6% to 68.1% of ideal. However, the phase
maps and fitted Zernike coefficients are not public. SCPIC therefore records
the geometry and can ingest those data through `ZernikeWavefront` or an OPD
callable, but does not claim to reproduce the two aberrated intensity maps.

Nine new focused regressions cover the wavelength Jacobian, spectral phase,
finite-Rayleigh waist limit, Gauss--Laguerre power conservation, Fourmaux
geometry, time-domain Maxwell equations, energy/flux factors, chunked
broadband equivalence, observation partitioning and whole-grid convergence.
The complete local suite passes with 41 tests. GPU execution, multi-rank MPI
I/O and a large 25-wavelength focused-volume energy calculation remain runtime
validation tasks for suitable hardware; they are not inferred from unit tests.

---

## Phase 12: Operational Documentation and Release Boundary (July 2026)

The public API and repository examples were audited against the work required
for a real EPOCH profile campaign. No additional optical feature was identified
as a correctness blocker for the supported perfect-conductor workflow. The
remaining high-value development was separated into explicit future projects:

* a full-vector EPOCH interior current-sheet or antenna injector, because
  `simple_laser` cannot impose the high-NA longitudinal electric field;
* measured wavefront, near-field and coating adapters built from representative
  instrument files rather than assumed formats;
* declarative campaign configuration and provenance capture;
* production CUDA and multi-rank MPI scaling and I/O validation;
* arbitrary surfaces, multiple reflectors and general spatiotemporal coupling.

A detailed documentation set was added:

* `docs/index.md` maps the documentation and states the supported release
  boundary;
* `docs/user_guide.md` gives the end-to-end 2D/3D, monochromatic/broadband,
  wavefront, convergence, MPI, export and diagnostic workflow;
* `docs/validation.md` defines the validation ladder and campaign acceptance
  record;
* `docs/api_reference.md` records the public objects, array shapes and return
  values;
* `docs/roadmap.md` distinguishes worthwhile extensions from unsafe silent
  defaults.

The guide records common convention failures, including the wavelength-density
Jacobian, TM01 coefficient meaning, reflective surface-height factor, EPOCH
array order, cosine-to-sine phase conversion and analytic-signal energy
factors. It also provides a minimum production checklist and specifies the
metadata that should accompany every generated profile.

During the documentation audit, the EPOCH-mod time-dependent reader was checked
again at source level. It supplies `omega*time` internally and multiplies any
deck `t_profile` by the file amplitude. `reconstruct_complex_envelope` and the
matching `carrier_angular_frequency` broadband option were therefore added.
They reconstruct
`2*sum(E_n*exp[-i*(omega_n-omega_carrier)*t])`, preventing accidental
double application of the carrier. Spectrum objects now expose an
envelope-detuning Nyquist timestep and a separate envelope time-grid
validator. A regression verifies that multiplying the envelope by
`exp(-i*omega_carrier*t)` recovers the original analytic signal exactly.

The phase writer was also changed to unwrap phase along every profile axis by
default. EPOCH interpolates the stored phase as an ordinary real number, so a
wrapped `[-pi,pi)` branch cut produces incorrect intermediate values even
though the samples themselves differ only by `2*pi`. An opt-out remains for
callers that manage phase continuity themselves, and a regression checks both
continuous interpolation output and exact sample-point reconstruction.

The local EPOCH suites were rerun with two MPI ranks after this change. All six
cases loaded and completed. The 2D OAP smoke test changed from the historical
wrapped-phase result to 1.066 µm FWHM against 0.944 µm in SCPIC, reducing the
coarse discrepancy to 13.0%. The 3D result became 1.117 µm and 1.106 µm
against 1.247 µm and 1.249 µm; the sign-sensitive phase tilts remained correct
at +9.53° and -5.94°. These are still boundary and grid smoke tests rather
than production convergence evidence.

Final release checks completed with 43 unit and regression tests passing,
successful byte-code compilation, and a clean formatting and whitespace
audit. The full six-case Vallières benchmark was also rerun: focal widths and
Rayleigh lengths remain within 2.6% of the paper, linear peak intensities
within 4.2%, and TM01 peak intensities within 8.4%.

---

## Phase 13: Non-separable Space-Time Couplings (July 2026)

Jolly *et al.*, *Space-time couplings in ultrashort lasers with arbitrary
nonparaxial focusing* (2025), was reviewed against the SCPIC implementation.
Its integral branch uses the same frequency-by-frequency physical-optics
Stratton--Chu strategy as SCPIC, but applies Zernike coefficients that vary
with frequency. This permits an incident field whose spatial and spectral
dependence is not separable.

The incident-field API was extended with
`spatio_spectral_phase(points, angular_frequency)`, returning phase in radians.
It is available on all five 3D incident-field families. The existing
`wavefront_opd(points)` input remains separate and continues to return metres
of physical path, contributing `k*OPD` at each frequency. This distinction
prevents fixed metrology and arbitrary chromatic phase from being conflated.

`ChromaticZernikePhase` now supplies general OSA/ANSI expansions with scalar or
frequency-callable coefficients. Three constructors reproduce the incident
phases in Jolly's equations (44)--(46):

* input angular dispersion / pulse-front tilt;
* chromatic curvature / longitudinal chromatism;
* chromatic trefoil.

The broadband propagator requires no new numerical kernel. It already
evaluates incident fields independently at every spectral component, so the
new phase is applied on mirror and contour quadrature points before the
existing Stratton--Chu integration and time reconstruction.

Space-time vortices in the paper revealed an EPOCH representation risk that is
different from an ordinary wrapped branch cut. A true phase singularity has
nonzero circulation and cannot be made globally continuous.
`epoch_phase_diagnostics()` was added to report low-amplitude coverage,
maximum reliable adjacent phase step, and phase-winding cells on every
two-dimensional slice of a 1D, 2D, or 3D profile.

Eleven regressions were added for the exact Jolly group-delay scalings,
trefoil symmetry, all incident-field classes, broadband frequency evaluation,
input validation, vortex detection, and the opposite focal centroids produced
by low- and high-frequency components after Stratton--Chu propagation. The
complete local suite now passes with 54 tests.

The original paraxial benchmark still agrees with its analytical waist to
0.03%, and the complete six-case Vallières suite is numerically unchanged
after the shared Zernike and incident-phase refactor.
