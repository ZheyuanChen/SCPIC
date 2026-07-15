# SCPIC

SCPIC computes a two-dimensional TM laser field reflected by an off-axis
parabolic mirror and exports a transverse electric-field profile for the custom
laser reader in the modified EPOCH checkout.

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
```

The paraxial benchmark uses an aperture wide enough to avoid clipping and
checks the reflected PEC physical-optics solution against the analytical 2D
Gaussian waist. `main.py` generates a static EPOCH2D injection profile:

```bash
.venv/bin/python main.py
```

It writes `epoch_injection_data/amplitude.dat` and `phase.dat` and prints the
electric-field `amp` value for the EPOCH laser block.

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

Three workstation-sized EPOCH cases cover a flat Gaussian, a sign-sensitive
phase ramp, and the complete SCPIC focus path. See
[`epoch_tests/README.md`](epoch_tests/README.md) for commands and acceptance
checks.

## Current physical limitation

EPOCH's `simple_laser` boundary accepts one transverse electric profile and
constructs the associated magnetic field itself. It cannot directly impose the
longitudinal `Ex` predicted for a high-NA focus, and its characteristic
boundary treatment is approximate for large angular content. The included f/1
local case currently gives a 1.126 µm EPOCH waist versus a 0.944 µm SCPIC
reference (19.3% difference on the coarse grid). This must be converged and
judged before production use; an interior current-sheet laser antenna is the
likely next step if the boundary error is unacceptable.
