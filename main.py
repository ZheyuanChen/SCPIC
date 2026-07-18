"""Generate a static EPOCH2D injection profile from an OAP90 mirror."""

import numpy as np

from scpic.export import export_epoch_profile
from scpic.fields import IncidentFieldTM, electric_from_magnetic_tm
from scpic.mirrors import ParabolicMirror2D
from scpic.solvers import evaluate_SC_2D


def main():
    """Generate one monochromatic demonstration profile for ``x_max``."""
    wavelength = 1e-6
    k = 2 * np.pi / wavelength

    mirror = ParabolicMirror2D(f0=10e-6, D=20e-6, mirror_type="OAP90")
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=2000)
    incident = IncidentFieldTM(w0=8e-6, wavelength=wavelength, E0=1.0)
    By_inc = incident.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = incident.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)

    # The focused wave travels towards -x, so this profile belongs on an
    # EPOCH x_max laser boundary.  Physical z maps to EPOCH's transverse y.
    x_injection = 5e-6
    z = np.linspace(-8e-6, 8e-6, 321)
    dx = 20e-9
    x = np.array([x_injection - dx, x_injection, x_injection + dx])
    X, Z = np.meshgrid(x, z, indexing="ij")
    By = evaluate_SC_2D(
        X.ravel(),
        Z.ravel(),
        x_m,
        z_m,
        nx,
        nz,
        dl,
        By_inc,
        dBy_dn_inc,
        k,
    ).reshape(X.shape)
    _, Ez = electric_from_magnetic_tm(By, x, z, k)
    profile = Ez[1]

    # Remove the propagator's arbitrary global piston phase while preserving
    # the wavefront variation across the injection line.
    phase_reference = np.angle(profile[np.argmax(np.abs(profile))])
    result = export_epoch_profile(
        "epoch_injection_data", profile, phase_reference=phase_reference
    )
    print(f"Wrote {result.shape[0]} float64 samples to {result.amplitude_file}")
    print(f"Wrote EPOCH phases to {result.phase_file}")
    print(f"Set the EPOCH laser amp to {result.field_scale:.9e} V/m")
    print("Use an x_max laser with pol = 0 and map SCPIC z to EPOCH y.")


if __name__ == "__main__":
    main()
