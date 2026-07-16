# SCPIC

SCPIC (Stratton-Chu for Particle-in-Cell simulations) computes two-dimensional TM or fully vectorial three-dimensional laser
fields reflected by parabolic mirrors, then exports transverse electric-field
profiles for the custom laser reader in the modified EPOCH checkout. The 3D
path follows the physical-optics Stratton--Chu method of Vallières *et al.*
(2023) and Dumont *et al.* (2017), with implementation insights from
C. F. Nielsen (2022). It supports OAP90, on-axis and annular
transmission-parabola apertures, including the NA = 0.96 experimental geometry
reported by Fourmaux *et al.* (2025).

The code uses one complex convention throughout:

```text
physical field = Re[phasor * exp(-i omega t)]
```

The EPOCH modification instead injects `amplitude * sin(omega t + phase)`.
`export_epoch_profile()` performs the required conversion
`phase_epoch = pi/2 - angle(phasor)` and writes normalised amplitude plus phase
as headerless native `float64` streams. Phase is unwrapped along every file
axis by default because EPOCH interpolates it linearly; wrapping to
`[-pi, pi)` would create false ramps at branch cuts.

## Installation and checks

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python benchmark.py
.venv/bin/python paper_benchmark_3d.py --suite
.venv/bin/python paper_benchmark_3d.py --convergence --mirror OAP90 --polarisation radial
```

Install `'.[dev,mpi]'` instead when using the optional mpi4py observation-domain
workflow. CuPy remains a separate installation because it must match the local
CUDA runtime.

The first benchmark checks the 2D PEC physical-optics solution against an
analytical paraxial Gaussian waist. The second reconstructs all six 20 J,
800 nm HNAP, OAP90 and TP cases from Tables 1--2 of Vallières *et al.* for
linear and exact TM01 radial input. `main.py` generates a static EPOCH2D
injection profile:

```bash
.venv/bin/python main.py
```

It writes `epoch_injection_data/amplitude.dat` and `phase.dat` and prints the
electric-field `amp` value for the EPOCH laser block.

## 3D example

```python
import numpy as np

from scpic import (
    LinearPolarisedSuperGaussian3D,
    ParabolicMirror3D,
    ZernikeWavefront,
    evaluate_SC_3D,
)

wavelength = 800e-9
mirror = ParabolicMirror3D.vallieres_oap90()
surface = mirror.surface_quadrature(n_radial=24, n_azimuthal=48)
incident = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
    200e-3,
    wavelength=wavelength,
    spatial_order=16,
    centre=(*mirror.aperture_centre, 0.0),
)
E_inc, B_inc = incident.fields(surface.points)
E_focus, B_focus = evaluate_SC_3D(
    np.array([[0.0, 0.0, 0.0]]),
    surface,
    E_inc,
    B_inc,
    2 * np.pi / wavelength,
)
```

Measured or modelled wavefront errors are supplied as optical path difference
in metres. For example, this applies 30 nm RMS OSA/ANSI defocus over the OAP
pupil:

```python
wavefront = ZernikeWavefront(
    pupil_radius=mirror.D / 2,
    coefficients={(2, 0): 30e-9},
    centre=(*mirror.aperture_centre, 0.0),
)
incident = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
    mirror.D,
    wavelength=wavelength,
    centre=(*mirror.aperture_centre, 0.0),
    wavefront_opd=wavefront,
)
```

`TM01RadiallyPolarisedBeam3D` implements the exact incident model in Eqs.
(16)--(17) of Vallières *et al.*, including its longitudinal electric field
and frequency-dependent energy normalisation. The older
`RadiallyPolarisedSuperGaussian3D` remains available as a deliberately simpler
transverse radial envelope; it is not the paper's TM01 model. Any callable
returning one finite OPD value per point can replace `ZernikeWavefront`,
allowing measured maps to be interpolated onto the mirror.

## Dumont broadband and finite-distance extensions

The Vallières-compatible narrow-band wavelength conversion remains the default.
For a very broad spectrum defined as energy per unit wavelength, request the
exact Jacobian explicitly. Spectral phase may be a scalar, an array, or a
callable of angular frequency:

```python
from scpic import SuperGaussianSpectrum

spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
    central_wavelength=800e-9,
    wavelength_fwhm=150e-9,
    total_energy=20.0,
    conversion="exact_wavelength_density",
).with_spectral_phase(lambda omega: 1.0e-29 * (omega - omega.mean()) ** 2)
```

`SampledSpectrum.from_wavelength_samples()` accepts measured `dE/dlambda`
data, applies `|dlambda/domega|`, and resamples it onto the uniform angular-
frequency grid required by the discrete Fourier representation.

`FiniteRayleighTM01Beam3D` implements Dumont's complex-q radial Gaussian,
including curvature and the longitudinal electric field. It reduces exactly
to `TM01RadiallyPolarisedBeam3D` at its waist. The complementary
`ParaxialGaussLaguerreBeam3D` accepts an axisymmetric set of radial
Gauss--Laguerre coefficients for a linearly polarised, well-collimated beam.
The latter is a paraxial upstream model and deliberately does not claim
high-order longitudinal corrections.

## Memory-bounded propagation and convergence

`iter_broadband_field_chunks()` calculates all frequencies for one observation
chunk, reconstructs the requested times, yields the result, and discards the
spectral intermediates. This bounds working memory by the chunk size rather
than the complete injection plane or focal volume. `propagate_broadband_3d()`
collects those chunks when the reconstructed local result fits in memory.

The default reconstruction includes the optical carrier and is intended for
field diagnostics. For an EPOCH spatiotemporal file, pass
`carrier_angular_frequency=omega0`. This returns the slowly varying complex
envelope because EPOCH supplies `omega0 * time` internally. Exporting the full
analytic signal would apply the carrier twice.

An mpi4py communicator can be supplied without changing the numerical kernel:

```python
from mpi4py import MPI
from scpic import iter_broadband_field_chunks

for chunk in iter_broadband_field_chunks(
    observations,
    surface,
    incident,
    spectrum,
    times=[0.0],
    communicator=MPI.COMM_WORLD,
):
    # Each rank receives a disjoint, globally indexed [chunk.start:chunk.stop].
    write_rank_chunk(chunk)
```

No MPI collective or parallel file format is imposed. This avoids recreating
the HDF5 output bottleneck reported by Dumont and lets campaigns use per-rank
files, a later assembly step, or a site-specific parallel writer.

`surface_quadrature_convergence()` refines mirror quadrature against the full
requested observation grid using a combined `(E, cB)` norm. For an EPOCH
profile, run it on the actual injection plane at the shortest relevant
wavelength; focal-point convergence alone can be misleading away from the
parabolic focus.

The diagnostics module now provides:

- monochromatic and time-domain source-free Maxwell residuals;
- cycle-averaged or instantaneous electromagnetic energy density;
- rectilinear volume-energy and plane-Poynting-flux integrals;
- a signed relative energy-conservation error.

An energy-conservation volume must be demonstrated to be large enough. Dumont
found that longitudinal-field tails can require extents of roughly 25
wavelengths, and expanding that volume can in turn require finer mirror
quadrature.

## Experimental transmission parabola (2025)

`ParabolicMirror3D.fourmaux_tp_2025()` returns the published 5.65 mm parent
focal length, 65 mm illuminated diameter and 24.5 mm central aperture. The
derived acute focusing-angle range is 38.3--85.4 degrees, corresponding to the
reported generalized-solid-angle NA of 0.96.

Fourmaux *et al.* measured 9.3 wavelengths peak-to-valley and 2.13 wavelengths
RMS before deformable-mirror correction, and 1.02 and 0.16 wavelengths after
correction. Their Stratton--Chu calculation reached 6% and 68.1% of the ideal
peak respectively. The phase maps and fitted Zernike coefficients are not
publicly available, so SCPIC includes the geometry and measured-wavefront input
path but does not present those intensity ratios as reproduced benchmarks.
The solver remains a perfect-conductor physical-optics model. For a direct
comparison with this experiment, the spectrum energy should be specified after
the paper's reported 2.5% gold-coating reflection loss (or an equivalent
frequency-dependent coating transfer should be applied externally).

For large individual observation chunks,
`evaluate_SC_3D(..., backend="cupy")` uses a locally installed CuPy build and
transfers one observation chunk at a time to the GPU. CuPy must be installed
separately to match the machine's CUDA version; the default NumPy path remains
the tested reference implementation.

## Documentation

| Document | Purpose |
|---|---|
| [`docs/index.md`](docs/index.md) | Documentation map and release status |
| [`docs/user_guide.md`](docs/user_guide.md) | End-to-end optical and EPOCH workflow |
| [`docs/validation.md`](docs/validation.md) | Convergence, Maxwell, energy, paper, and EPOCH acceptance |
| [`docs/api_reference.md`](docs/api_reference.md) | Public classes, functions, shapes, and return values |
| [`docs/methodology_3d.md`](docs/methodology_3d.md) | Equations and benchmark evidence |
| [`epoch_tests/README.md`](epoch_tests/README.md) | Local EPOCH2D and EPOCH3D integration tests |
| [`docs/literature_review.md`](docs/literature_review.md) | Related papers and implementations |
| [`docs/roadmap.md`](docs/roadmap.md) | Worthwhile but intentionally deferred features |

The most important remaining scientific development is a full-vector interior
current-sheet injector inside EPOCH. It is not a missing optical-solver feature:
it is needed because `simple_laser` cannot directly impose the longitudinal
electric field of a near-unity-NA focus. Measured-instrument adapters and
CUDA/MPI production validation are also worthwhile, but depend on representative
data formats or suitable hardware.

## EPOCH mapping

For the supplied OAP90 geometry, the reflected beam propagates towards `-x`.
Use an EPOCH2D `x_max` laser with `pol = 0`; SCPIC's physical `z` coordinate
maps to EPOCH's transverse `y`, and SCPIC `Ez` maps to EPOCH `Ey`. Array order
must be:

| EPOCH mode | NumPy shape before `.tofile()` |
|---|---|
| 2D static | `(n_y,)` |
| 2D spatiotemporal | `(n_t, n_y)` |
| 3D static | `(n_tr2, n_tr1)` |
| 3D spatiotemporal | `(n_t, n_tr2, n_tr1)` |

No transpose or header is required. EPOCH-mod hard-codes eight bytes per real,
so SCPIC deliberately rejects `float32` export.

Static files contain a monochromatic complex phasor. Spatiotemporal files must
contain a complex envelope relative to the `lambda`/carrier specified in the
EPOCH deck. If the temporal envelope is already present in the file, omit
`t_profile`; otherwise EPOCH multiplies the two envelopes.

Three EPOCH2D and three EPOCH3D workstation-sized cases cover array ordering,
sign-sensitive phase gradients and complete SCPIC focus paths. See
[`epoch_tests/README.md`](epoch_tests/README.md) for commands and acceptance
checks.

## EPOCH boundary limitation

EPOCH's `simple_laser` boundary accepts transverse electric profiles and
constructs the associated magnetic field itself. A second laser block can
drive the other transverse polarisation, but the boundary cannot directly
impose the longitudinal field predicted for a high-NA focus. Its
characteristic treatment is also approximate for large angular content. The
local tests are therefore integration and propagation checks, not proof that
the boundary reproduces every component of the Stratton--Chu solution. An
interior current-sheet laser antenna remains the likely route if that error is
unacceptable for production simulations.
