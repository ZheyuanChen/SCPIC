"""Field-level checks for completed local EPOCH3D integration cases."""

import glob
import json
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
    files = list((HERE / "runs_3d" / case).glob("*.sdf"))
    if not files:
        raise SystemExit(f"No SDF output found for {case}; run run_local_3d.py first")
    return sorted(
        (sdf.read(str(path), dict=True) for path in files),
        key=lambda dump: dump["Header"]["time"],
    )


def _metadata(case):
    return json.loads((HERE / "runs_3d" / case / "profile.json").read_text())


def _fit_waist(coordinate, values, initial):
    def gaussian(position, amplitude, waist):
        return amplitude * np.exp(-((position / waist) ** 2))

    fitted, _ = curve_fit(
        gaussian,
        coordinate,
        np.abs(values),
        p0=(np.max(np.abs(values)), initial),
    )
    return abs(fitted[1])


def _fwhm(coordinate, intensity):
    peak = int(np.argmax(intensity))
    half = intensity[peak] / 2
    roots = []
    for indices in (range(peak - 1, -1, -1), range(peak, len(intensity) - 1)):
        for index in indices:
            neighbour = index + 1
            if (intensity[index] - half) * (intensity[neighbour] - half) <= 0:
                fraction = (half - intensity[index]) / (
                    intensity[neighbour] - intensity[index]
                )
                roots.append(
                    coordinate[index]
                    + fraction * (coordinate[neighbour] - coordinate[index])
                )
                break
    return abs(roots[1] - roots[0])


def check_static_gaussian():
    """Check that EPOCH preserves both injected Gaussian widths."""
    case = "static_gaussian_3d"
    metadata = _metadata(case)
    dumps = _dumps(case)
    dump = max(dumps, key=lambda item: np.sum(item["Electric Field/Ey"].data ** 2))
    ey = dump["Electric Field/Ey"].data
    x, y, z = dump["Grid/Grid_mid"].data
    ix = np.unravel_index(np.argmax(np.abs(ey)), ey.shape)[0]
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z)))
    waist_y = _fit_waist(y, ey[ix, :, iz], metadata["expected_y_waist_m"])
    waist_z = _fit_waist(z, ey[ix, iy, :], metadata["expected_z_waist_m"])
    assert abs(waist_y / metadata["expected_y_waist_m"] - 1) < 0.12
    assert abs(waist_z / metadata["expected_z_waist_m"] - 1) < 0.12
    print(
        f"static_gaussian_3d: y waist={waist_y * 1e6:.3f} µm, "
        f"z waist={waist_z * 1e6:.3f} µm"
    )


def check_phase_tilt():
    """Recover the two imposed phase tilts from integrated Poynting flux."""
    case = "phase_tilt_3d"
    metadata = _metadata(case)
    dumps = _dumps(case)
    dump = max(dumps, key=lambda item: np.sum(item["Electric Field/Ey"].data ** 2))
    ex = dump["Electric Field/Ex"].data
    ey = dump["Electric Field/Ey"].data
    bx = dump["Magnetic Field/Bx"].data
    bz = dump["Magnetic Field/Bz"].data
    sx = np.sum(ey * bz)
    angle_y = np.degrees(np.arctan2(np.sum(-ex * bz), sx))
    angle_z = np.degrees(np.arctan2(np.sum(-ey * bx), sx))
    assert abs(angle_y - metadata["expected_angle_y_deg"]) < 4.0
    assert abs(angle_z - metadata["expected_angle_z_deg"]) < 4.0
    print(f"phase_tilt_3d: Poynting angles y={angle_y:.2f}°, z={angle_z:.2f}°")


def check_scpic_focus():
    """Compare both EPOCH focal widths with the direct 3D references."""
    case = "scpic_focus_3d"
    metadata = _metadata(case)
    candidates = []
    for dump in _dumps(case):
        ez = dump["Electric Field/Ez"].data
        x, y, z = dump["Grid/Grid_mid"].data
        ix = int(np.argmin(np.abs(x)))
        candidates.append((np.sum(ez[ix] ** 2), dump, ez[ix], y, z))
    _, dump, plane, y, z = max(candidates, key=lambda item: item[0])
    iy = int(np.argmin(np.abs(y)))
    iz = int(np.argmin(np.abs(z)))
    fwhm_y = _fwhm(y, plane[:, iz] ** 2)
    fwhm_z = _fwhm(z, plane[iy, :] ** 2)
    error_y = abs(fwhm_y / metadata["expected_y_fwhm_m"] - 1)
    error_z = abs(fwhm_z / metadata["expected_z_fwhm_m"] - 1)
    assert error_y < 0.35
    assert error_z < 0.35
    print(
        f"scpic_focus_3d: EPOCH FWHM y={fwhm_y * 1e6:.3f} µm, "
        f"z={fwhm_z * 1e6:.3f} µm; SCPIC references "
        f"{metadata['expected_y_fwhm_m'] * 1e6:.3f}, "
        f"{metadata['expected_z_fwhm_m'] * 1e6:.3f} µm"
    )


def main():
    """Run all EPOCH3D output checks."""
    check_static_gaussian()
    check_phase_tilt()
    check_scpic_focus()


if __name__ == "__main__":
    main()
