"""Generate profiles for the local EPOCH3D integration cases."""

import json
from pathlib import Path

import numpy as np

from scpic.export import export_epoch_profile
from scpic.fields import LinearPolarisedSuperGaussian3D
from scpic.mirrors import ParabolicMirror3D
from scpic.solvers import evaluate_SC_3D


def _write(case_dir, field, y, z, **metadata):
    result = export_epoch_profile(case_dir, field, **metadata.pop("export", {}))
    description = {
        "shape": result.shape,
        "field_scale_v_per_m": result.field_scale,
        "y_min_m": float(y[0]),
        "y_max_m": float(y[-1]),
        "z_min_m": float(z[0]),
        "z_max_m": float(z[-1]),
        **metadata,
    }
    (case_dir / "profile.json").write_text(json.dumps(description, indent=2) + "\n")


def generate_static_gaussian(case_dir):
    y = np.linspace(-5e-6, 5e-6, 65)
    z = np.linspace(-5e-6, 5e-6, 61)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    field = np.exp(-((Y / 2.5e-6) ** 2) - (Z / 1.5e-6) ** 2).astype(complex)
    _write(case_dir, field, y, z, expected_y_waist_m=2.5e-6, expected_z_waist_m=1.5e-6)


def generate_phase_tilt(case_dir):
    wavelength = 1e-6
    y = np.linspace(-5e-6, 5e-6, 65)
    z = np.linspace(-5e-6, 5e-6, 61)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    angle_y = np.deg2rad(8.0)
    angle_z = np.deg2rad(-5.0)
    phase = 2 * np.pi / wavelength * (np.sin(angle_y) * Y + np.sin(angle_z) * Z)
    field = np.exp(-((Y**2 + Z**2) / (2.5e-6) ** 2)) * np.exp(1j * phase)
    _write(
        case_dir,
        field,
        y,
        z,
        expected_angle_y_deg=8.0,
        expected_angle_z_deg=-5.0,
    )


def _fwhm(coordinate, values):
    peak = int(np.argmax(values))
    half = values[peak] / 2
    roots = []
    for indices in (range(peak - 1, -1, -1), range(peak, len(values) - 1)):
        for index in indices:
            neighbour = index + 1
            if (values[index] - half) * (values[neighbour] - half) <= 0:
                fraction = (half - values[index]) / (values[neighbour] - values[index])
                roots.append(
                    coordinate[index]
                    + fraction * (coordinate[neighbour] - coordinate[index])
                )
                break
    return float(abs(roots[1] - roots[0]))


def generate_scpic_focus_3d(case_dir):
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=10e-6, D=20e-6, mirror_type="OAP90")
    surface = mirror.surface_quadrature(24, 48)
    incident = LinearPolarisedSuperGaussian3D(
        w0=8e-6,
        wavelength=wavelength,
        spatial_order=2,
        centre=(*mirror.aperture_centre, 0.0),
    )
    E_inc, B_inc = incident.fields(surface.points)

    y = np.linspace(-5e-6, 5e-6, 65)
    z = np.linspace(-5e-6, 5e-6, 65)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    observations = np.column_stack((np.full(Y.size, 4e-6), Y.ravel(), Z.ravel()))
    electric, _ = evaluate_SC_3D(observations, surface, E_inc, B_inc, k, chunk_size=32)
    # An x-boundary laser with pol=90 drives Ez, the dominant transverse
    # component of this OAP90/polarisation geometry.
    field = electric[:, 2].reshape(Y.shape)
    reference = np.angle(field.ravel()[np.argmax(np.abs(field))])

    focal = np.linspace(-2e-6, 2e-6, 161)
    focal_observations = np.concatenate(
        (
            np.column_stack((np.zeros_like(focal), focal, np.zeros_like(focal))),
            np.column_stack((np.zeros_like(focal), np.zeros_like(focal), focal)),
        )
    )
    focal_electric, _ = evaluate_SC_3D(focal_observations, surface, E_inc, B_inc, k)
    intensity = np.sum(np.abs(focal_electric) ** 2, axis=1)
    _write(
        case_dir,
        field,
        y,
        z,
        injection_x_m=4e-6,
        expected_y_fwhm_m=_fwhm(focal, intensity[: len(focal)]),
        expected_z_fwhm_m=_fwhm(focal, intensity[len(focal) :]),
        export={"phase_reference": reference},
    )


GENERATORS_3D = {
    "static_gaussian_3d": generate_static_gaussian,
    "phase_tilt_3d": generate_phase_tilt,
    "scpic_focus_3d": generate_scpic_focus_3d,
}


def generate_all(root):
    root = Path(root)
    for name, generator in GENERATORS_3D.items():
        case_dir = root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        generator(case_dir)


if __name__ == "__main__":
    generate_all(Path(__file__).parent / "runs_3d")
