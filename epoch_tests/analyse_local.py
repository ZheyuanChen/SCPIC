"""Field-level checks for completed local EPOCH2D integration cases."""

import glob
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent


def _import_sdf():
    try:
        import sdf

        return sdf
    except ImportError:
        candidates = glob.glob(
            str(HERE.parents[1] / "epoch_dev" / "SDF" / "utilities" / "build" / "lib.*")
        )
        if not candidates:
            raise SystemExit(
                "Could not import sdf or find epoch_dev/SDF/utilities/build/lib.*"
            )
        sys.path.insert(0, candidates[0])
        import sdf

        return sdf


sdf = _import_sdf()


def _dumps(case):
    files = list((HERE / "runs" / case).glob("*.sdf"))
    if not files:
        raise SystemExit(f"No SDF output found for {case}; run run_local.py first")
    return sorted(
        (sdf.read(str(path), dict=True) for path in files),
        key=lambda dump: dump["Header"]["time"],
    )


def _fit_waist(y, values, initial):
    def gaussian(coord, amplitude, waist):
        return amplitude * np.exp(-((coord / waist) ** 2))

    fitted, _ = curve_fit(
        gaussian, y, np.abs(values), p0=(np.max(np.abs(values)), initial)
    )
    return abs(fitted[1])


def check_static_gaussian():
    dumps = _dumps("static_gaussian")
    dump = max(dumps, key=lambda item: np.sum(item["Electric Field/Ey"].data ** 2))
    ey = dump["Electric Field/Ey"].data
    bz = dump["Magnetic Field/Bz"].data
    y = dump["Grid/Grid_mid"].data[1]
    ix = np.unravel_index(np.argmax(np.abs(ey)), ey.shape)[0]
    waist = _fit_waist(y, ey[ix], 2e-6)
    mask = np.abs(ey[ix]) > 0.25 * np.max(np.abs(ey[ix]))
    impedance = float(np.median(ey[ix][mask] / bz[ix][mask]))
    assert abs(waist / 2e-6 - 1) < 0.1
    assert abs(impedance / 299_792_458.0 - 1) < 0.05
    print(f"static_gaussian: waist={waist * 1e6:.3f} µm, " f"Ey/Bz={impedance:.4e} m/s")


def check_phase_ramp():
    dumps = _dumps("phase_ramp")
    dump = max(dumps, key=lambda item: np.sum(item["Electric Field/Ey"].data ** 2))
    ex = dump["Electric Field/Ex"].data
    ey = dump["Electric Field/Ey"].data
    bz = dump["Magnetic Field/Bz"].data
    sx = np.sum(ey * bz)
    sy = np.sum(-ex * bz)
    angle = float(np.degrees(np.arctan2(sy, sx)))
    assert 6.0 < angle < 14.0
    print(f"phase_ramp: integrated Poynting angle={angle:.2f}° (target +10°)")


def check_scpic_focus():
    dumps = _dumps("scpic_focus")
    planes = []
    for dump in dumps:
        x = dump["Grid/Grid_mid"].data[0]
        ix = int(np.argmin(np.abs(x)))
        ey = dump["Electric Field/Ey"].data[ix]
        planes.append((np.sum(ey**2), dump, ey))
    _, dump, ey = max(planes, key=lambda item: item[0])
    y = dump["Grid/Grid_mid"].data[1]
    waist = _fit_waist(y, ey, 1e-6)
    reference_waist = 0.944e-6
    relative_error = abs(waist / reference_waist - 1)
    peak_time = dump["Header"]["time"]
    assert 22e-15 < peak_time < 28e-15
    assert relative_error < 0.3
    print(
        f"scpic_focus: EPOCH waist={waist * 1e6:.3f} µm, "
        f"SCPIC reference={reference_waist * 1e6:.3f} µm, "
        f"difference={relative_error:.1%}"
    )


def main():
    check_static_gaussian()
    check_phase_ramp()
    check_scpic_focus()


if __name__ == "__main__":
    main()
