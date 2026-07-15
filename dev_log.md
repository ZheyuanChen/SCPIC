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
Implemented `IncidentFieldTM` to model the incoming laser beam traveling along the +z-axis.
*   Models a 2D Gaussian spatial profile: 

$$B_y(x, z) = B_0 \exp\left(-\frac{(x - x_c)^2}{w_0^2}\right) e^{i k z}$$

*   Calculates analytical spatial derivatives to provide `∂B_y/∂n = n_x * ∂B_y/∂x + n_z * ∂B_y/∂z` directly on the mirror surface, eliminating numerical error on the boundary.

---

## Phase 2: Numerical Integration Core (`solvers.py`)

*   Implemented the Stratton-Chu boundary integrator `evaluate_SC_2D`.
*   To resolve the computational bottleneck of evaluating expensive Hankel functions (H_0^(1) and H_1^(1)) for every observation grid point relative to every mirror element, we integrated **Numba**-accelerated JIT compilation.
*   Designed the solver with a parallelized CPU pipeline (`numba.njit(parallel=True)`) to scale execution across multi-core systems efficiently.

---

## Phase 3: Binary Export Pipeline (`export.py`)

*   **Format Shift (HDF5 to Raw Binary):** Transitioned the output module from complex HDF5 files to raw C-style binary streams using NumPy's `ndarray.tofile()`. This eliminates library overhead and directly interfaces with custom Fortran input routines in modified EPOCH setups.
*   **Amplitude/Phase Splitting:** Complex fields are split into separate real arrays to write out:
    1.  Amplitude profiles: `<component>_amp.bin` (via `np.abs()`)
    2.  Phase profiles: `<component>_phase.bin` (via `np.angle()`)
*   **Precision Control:** Configured precision casting (`np.float32` vs `np.float64`) to match the double or single-precision configuration (`real(4)` or `real(8)`) used by the destination compiler.

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
                  (Propagating along +z)
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
* Configuration: Focal length f_0 = 50 μm (effective focal length f_eff = 2 * f_0 = 100 μm), mirror diameter D = 20 μm, and incident beam waist w_inc = 15 μm.
* Physical Metric: In this paraxial limit, the focal spot waist must match the analytical 2D paraxial Gaussian beam waist formula:

  w_focus ≈ (2 * lambda * f_0) / (pi * w_inc)

* Result: Fitted a Gaussian profile to the transverse E_z field slice at x = 0. The numerical waist agreed with the analytical formula to within < 2% error, validating the integration phase, amplitude scaling, and geometric projection.

### 2. Tight-Focusing Regime (Non-Paraxial Validation)
* Configuration: Focal length f_0 = 10 μm, mirror diameter D = 20 μm, and incident beam waist w_inc = 8 μm (High Numerical Aperture).
* Physical Signatures Verified:
  1. Longitudinal Field Coupling (E_x): The longitudinal field component E_x (which is negligible in paraxial approximations) grew to a significant fraction (~10%+) of the primary transverse field E_z.
  2. Antisymmetric Phase Signature: Since E_x is proportional to ∂B_y/∂z, and B_y peaks symmetrically at z = 0, the longitudinal field profile exhibited a perfect antisymmetric double-lobed structure with a deep null at the coordinate center.

---

## Phase 6: Automated Regression Framework (`tests/`)

* Implemented continuous verification via unit tests (`tests/test_scpic.py`) targeting critical math nodes:
  * `test_mirror_normal_vectors`: Verifies unit normal vectors conform to sqrt(nx^2 + nz^2) = 1.0.
  * `test_incident_field_amplitude`: Assures Gaussian spatial peak centers align perfectly at boundaries.
  * `test_binary_export_structure`: Confirms raw binary file size allocations match expectations and validates structural round-trips.
  * `test_paraxial_solver_accuracy`: Encapsulates the paraxial waist benchmark into an automated regression gate ensuring code updates never degrade physical precision.