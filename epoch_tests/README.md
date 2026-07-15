# Local EPOCH validation cases

These cases exercise the exact custom-profile reader in the sibling
`epoch_dev` checkout. They are intentionally small enough for a workstation.

Run all three from the SCPIC repository root:

```bash
.venv/bin/python epoch_tests/run_local.py
```

Then run the first quantitative checks (using the Python SDF extension built
in the sibling `epoch_dev` tree):

```bash
.venv/bin/python epoch_tests/analyse_local.py
```

Use `--epoch-bin PATH` if `epoch_dev/epoch2d/bin/epoch2d` is elsewhere, and
`--ranks 2` to repeat the file loading and interpolation under MPI. Generated
profiles, logs, and SDF dumps live under `epoch_tests/runs/` and are ignored by
Git.

The cases form a staged validation:

1. `static_gaussian` verifies the float64 stream size, C/Fortran ordering,
   static interpolation, normalised amplitude, and the constant cosine-to-sine
   phase offset. At the pulse peak, `Ey` a few cells inside `x_min` should be a
   centred Gaussian with a 2 µm field waist; `Bz` should have the corresponding
   positive-going-wave sign and `|Ey|/|Bz|` close to `c` away from the boundary.
2. `phase_ramp` adds a 10-degree positive transverse wavevector. The injected
   phase is `pi/2 - k_y y`, not `+k_y y`. The cycle-averaged transverse
   Poynting flux reconstructed from `Ex`, `Ey`, and `Bz` should point towards
   positive `y`. This is the sign-sensitive regression for the SCPIC
   `exp(-i omega t)` to EPOCH `sin(omega t + phase)` conversion.
3. `scpic_focus` exercises the complete OAP90 → PEC physical-optics integral →
   `Ez` → EPOCH path. The physical SCPIC `z` coordinate maps to EPOCH `y`, and
   the field is injected at `x_max` because it propagates towards `-x`. Compare
   transverse slices around `x=0` with the SCPIC prediction. The current coarse
   local case treats a waist difference below 30% as a smoke-test pass and
   reports the measured discrepancy; production acceptance should be tightened
   after a grid-convergence study.

Passing these cases proves file compatibility and provides a first propagation
check; it does **not** prove exact high-NA injection. EPOCH's `simple_laser`
boundary generates its own magnetic field from one transverse electric profile
and cannot directly impose SCPIC's longitudinal `Ex`. For the f/1 case, measure
`Ex`, `Ey`, and `Bz` several cells downstream against the SCPIC reference before
using the result for production physics. If that error is material, an interior
current-sheet/antenna injector is the appropriate next development step.

## EPOCH3D cases

The 3D suite uses the corresponding local executable and is run separately:

```bash
.venv/bin/python epoch_tests/run_local_3d.py
.venv/bin/python epoch_tests/analyse_local_3d.py
```

Use `--epoch-bin PATH`, `--ranks N`, or list individual case names in the same
way as the 2D runner. Generated data live under `epoch_tests/runs_3d/`.

1. `static_gaussian_3d` loads a `(n_z, n_y)` static stream with different
   waists on the two axes. It catches transposition, axis mapping and spatial
   interpolation mistakes.
2. `phase_tilt_3d` applies +8° and −5° transverse phase gradients. The signs
   and magnitudes of both reconstructed Poynting angles test the phasor-to-EPOCH
   conversion and 3D raw-file ordering.
3. `scpic_focus_3d` evaluates the full vector OAP90 integral on the `x_max`
   injection plane, exports its dominant `Ez` component, and compares the two
   focal-plane widths after EPOCH propagation with direct SCPIC references.

All three cases pass with the local EPOCH3D build. On the present coarse grids,
the static fitted field waists are 2.474 µm and 1.438 µm for targets of
2.5 µm and 1.5 µm; the phase-tilt Poynting angles are +9.23° and −5.76°; and
the propagated OAP focus has 1.187 µm FWHM on both axes versus direct SCPIC
references of 1.247 µm and 1.249 µm. These tolerances should be tightened in a
future grid-convergence study.
