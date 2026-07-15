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
the project configuration, matching `python -m pytest`. The remaining paper
work is a converged radial-polarisation and HNAP/TP reproduction; the remaining
performance work is a NumPy/CuPy comparison on CUDA hardware.
