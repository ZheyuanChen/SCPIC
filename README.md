# SCPIC

SCPIC computes two-dimensional TM or fully vectorial three-dimensional laser
fields reflected by parabolic mirrors, then exports transverse electric-field
profiles for the custom laser reader in the modified EPOCH checkout. The 3D
path follows the physical-optics Stratton--Chu method of Vallières *et al.*
(2023) and supports OAP90, on-axis and annular transmission-parabola apertures.

The code uses one complex convention throughout:

```text
physical field = Re[phasor * exp(-i omega t)]
```

The EPOCH modification instead injects `amplitude * sin(omega t + phase)`.
`export_epoch_profile()` performs the required conversion
`phase_epoch = pi/2 - angle(phasor)` and writes normalised amplitude plus phase
as headerless native `float64` streams.

## Installation and checks

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/python benchmark.py
.venv/bin/python paper_benchmark_3d.py
```

The first benchmark checks the 2D PEC physical-optics solution against an
analytical paraxial Gaussian waist. The second reconstructs the 20 J,
800 nm OAP90 case from Vallières *et al.* and compares its intensity and focal
dimensions with their Table 1. `main.py` generates a static EPOCH2D injection
profile:

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
    evaluate_SC_3D,
)

wavelength = 800e-9
mirror = ParabolicMirror3D.vallieres_oap90()
surface = mirror.surface_quadrature(n_radial=24, n_azimuthal=48)
incident = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
    mirror.D,
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

See [`docs/methodology_3d.md`](docs/methodology_3d.md) for equations,
conventions, benchmark results and current limitations. The existing-code
survey is in [`docs/literature_review.md`](docs/literature_review.md).

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
