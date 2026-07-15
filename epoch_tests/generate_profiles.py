"""Generate the binary profile pairs used by the local EPOCH2D cases."""

import json
from pathlib import Path

import numpy as np

from scpic.export import export_epoch_profile
from scpic.fields import IncidentFieldTM, electric_from_magnetic_tm
from scpic.mirrors import ParabolicMirror2D
from scpic.solvers import evaluate_SC_2D


def _write(case_dir, field, y, **export_kwargs):
    result = export_epoch_profile(case_dir, field, **export_kwargs)
    metadata = {
        "shape": result.shape,
        "field_scale_v_per_m": result.field_scale,
        "y_min_m": float(y[0]),
        "y_max_m": float(y[-1]),
    }
    (case_dir / "profile.json").write_text(json.dumps(metadata, indent=2) + "\n")


def generate_static_gaussian(case_dir):
    y = np.linspace(-6e-6, 6e-6, 257)
    field = np.exp(-((y / 2e-6) ** 2)).astype(complex)
    _write(case_dir, field, y)


def generate_phase_ramp(case_dir):
    wavelength = 1e-6
    angle = np.deg2rad(10.0)
    y = np.linspace(-6e-6, 6e-6, 257)
    envelope = np.exp(-((y / 2.5e-6) ** 2))
    field = envelope * np.exp(1j * (2 * np.pi / wavelength) * np.sin(angle) * y)
    _write(case_dir, field, y)


def generate_scpic_focus(case_dir):
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror2D(f0=10e-6, D=20e-6, mirror_type="OAP90")
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=2000)
    incident = IncidentFieldTM(w0=8e-6, wavelength=wavelength, E0=1.0)
    By_inc = incident.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = incident.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)

    y = np.linspace(-8e-6, 8e-6, 321)
    x_injection = 5e-6
    dx = 20e-9
    x = np.array([x_injection - dx, x_injection, x_injection + dx])
    X, Z = np.meshgrid(x, y, indexing="ij")
    By = evaluate_SC_2D(
        X.ravel(), Z.ravel(), x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k
    ).reshape(X.shape)
    _, Ez = electric_from_magnetic_tm(By, x, y, k)
    field = Ez[1]
    reference = np.angle(field[np.argmax(np.abs(field))])
    _write(case_dir, field, y, phase_reference=reference)


GENERATORS = {
    "static_gaussian": generate_static_gaussian,
    "phase_ramp": generate_phase_ramp,
    "scpic_focus": generate_scpic_focus,
}


def generate_all(root):
    root = Path(root)
    for name, generator in GENERATORS.items():
        case_dir = root / name
        case_dir.mkdir(parents=True, exist_ok=True)
        generator(case_dir)


if __name__ == "__main__":
    generate_all(Path(__file__).parent / "runs")
