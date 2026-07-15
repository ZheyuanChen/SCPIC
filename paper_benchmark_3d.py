"""Reproduce the six focal-field cases tabulated by Vallières et al. (2023).

The default configuration is sized for a workstation.  ``--suite`` evaluates
all three reflectors with both linear and TM01 incident fields.  ``--convergence``
runs independent surface-quadrature and spectral refinements for one case.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scpic.fields import (
    C,
    LinearPolarisedSuperGaussian3D,
    TM01RadiallyPolarisedBeam3D,
)
from scpic.mirrors import ParabolicMirror3D
from scpic.pulse import (
    SuperGaussianSpectrum,
    electric_intensity,
    reconstruct_analytic_signal,
)
from scpic.solvers import evaluate_SC_3D

INCIDENT_FWHM = 200e-3
SPATIAL_ORDER = 16

PAPER_REFERENCE = {
    "linear": {
        "HNAP": {
            "peak_intensity_W_cm2": 5.03e23,
            "fwhm_x_um": 0.62,
            "fwhm_y_um": 0.33,
            "rayleigh_length_um": 0.39,
        },
        "OAP90": {
            "peak_intensity_W_cm2": 2.66e23,
            "fwhm_x_um": 0.60,
            "fwhm_y_um": 0.52,
            "rayleigh_length_um": 0.69,
        },
        "TP": {
            "peak_intensity_W_cm2": 1.49e23,
            "fwhm_x_um": 0.87,
            "fwhm_y_um": 1.26,
            "rayleigh_length_um": 0.56,
        },
    },
    "radial": {
        "HNAP": {
            "peak_intensity_W_cm2": 4.13e23,
            "fwhm_x_um": 0.35,
            "fwhm_y_um": 0.35,
            "rayleigh_length_um": 0.45,
        },
        "OAP90": {
            "peak_intensity_W_cm2": 1.07e23,
            "fwhm_x_um": 0.69,
            "fwhm_y_um": 0.68,
            "rayleigh_length_um": 0.53,
        },
        "TP": {
            "peak_intensity_W_cm2": 2.45e23,
            "fwhm_x_um": 0.37,
            "fwhm_y_um": 0.37,
            "rayleigh_length_um": 0.56,
        },
    },
}


def _fwhm(coordinate, values):
    """Return the full extent above half of the global maximum.

    Using the outermost crossings also handles the annular, double-lobed TP
    focus reported for linear polarisation.
    """
    coordinate = np.asarray(coordinate, dtype=float)
    values = np.asarray(values, dtype=float)
    half = np.max(values) / 2
    above = np.flatnonzero(values >= half)
    if not len(above) or above[0] == 0 or above[-1] == len(values) - 1:
        return np.nan

    def interpolate(left):
        right = left + 1
        fraction = (half - values[left]) / (values[right] - values[left])
        return coordinate[left] + fraction * (coordinate[right] - coordinate[left])

    return interpolate(above[-1]) - interpolate(above[0] - 1)


def _mirror(mirror_type):
    factories = {
        "HNAP": ParabolicMirror3D.vallieres_hnap,
        "OAP90": ParabolicMirror3D.vallieres_oap90,
        "TP": ParabolicMirror3D.vallieres_tp,
    }
    try:
        return factories[mirror_type]()
    except KeyError as error:
        raise ValueError("mirror_type must be 'HNAP', 'OAP90', or 'TP'") from error


def _incident_field(polarisation, mirror):
    centre = (*mirror.aperture_centre, 0.0)
    w0 = INCIDENT_FWHM / (2 * (np.log(2) / 2) ** (1 / SPATIAL_ORDER))
    if polarisation == "linear":
        return LinearPolarisedSuperGaussian3D(
            w0=w0,
            spatial_order=SPATIAL_ORDER,
            centre=centre,
        )
    if polarisation == "radial":
        return TM01RadiallyPolarisedBeam3D(w0=w0, centre=centre)
    raise ValueError("polarisation must be 'linear' or 'radial'")


def _transverse_observations(mirror_type, coordinate):
    zeros = np.zeros_like(coordinate)
    if mirror_type == "OAP90":
        # The chief ray travels along -x; paper x is the OAP meridional (z)
        # direction and paper y is the sagittal (global y) direction.
        profile_x = np.column_stack((zeros, zeros, coordinate))
        profile_y = np.column_stack((zeros, coordinate, zeros))
    else:
        profile_x = np.column_stack((coordinate, zeros, zeros))
        profile_y = np.column_stack((zeros, coordinate, zeros))
    return np.concatenate((profile_x, profile_y))


def _axial_observations(mirror_type, coordinate, offset_x, offset_y):
    if mirror_type == "OAP90":
        # offset_x and offset_y are expressed in the paper's transverse axes.
        return np.column_stack(
            (
                coordinate,
                np.full_like(coordinate, offset_y),
                np.full_like(coordinate, offset_x),
            )
        )
    return np.column_stack(
        (
            np.full_like(coordinate, offset_x),
            np.full_like(coordinate, offset_y),
            coordinate,
        )
    )


def _propagate_components(
    observations, mirror, surface, incident, spectrum, amplitudes
):
    components = np.empty(
        (len(spectrum.angular_frequencies), len(observations), 3), dtype=complex
    )
    for index, (omega, amplitude) in enumerate(
        zip(spectrum.angular_frequencies, amplitudes)
    ):
        k = omega / C
        electric_incident, magnetic_incident = incident.fields(
            surface.points,
            k=k,
            amplitude=amplitude,
            # The parent-paraboloid optical path to the focus is 2*f0.
            spectral_phase=-2 * k * mirror.f0,
        )
        components[index], _ = evaluate_SC_3D(
            observations,
            surface,
            electric_incident,
            magnetic_incident,
            k,
        )
    analytic = reconstruct_analytic_signal(
        components, spectrum.angular_frequencies, [0.0]
    )[0]
    return electric_intensity(analytic)


def run_case(
    mirror_type="OAP90",
    polarisation="linear",
    *,
    n_radial=24,
    n_azimuthal=48,
    n_profile=121,
    n_components=31,
    profile_half_width=1.5e-6,
    axial_half_width=2.0e-6,
):
    """Evaluate one paper case and return computed/reference diagnostics."""
    if n_profile < 5:
        raise ValueError("n_profile must be at least 5")
    mirror = _mirror(mirror_type)
    surface = mirror.surface_quadrature(n_radial, n_azimuthal)
    incident = _incident_field(polarisation, mirror)
    spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
        central_wavelength=800e-9,
        wavelength_fwhm=90e-9,
        spectral_order=7,
        total_energy=20.0,
        n_components=n_components,
    )
    if polarisation == "radial":
        effective_area = incident.effective_area(spectrum.angular_frequencies / C)
    else:
        effective_area = incident.effective_area
    amplitudes = spectrum.component_amplitudes(effective_area)

    transverse = np.linspace(-profile_half_width, profile_half_width, n_profile)
    transverse_intensity = _propagate_components(
        _transverse_observations(mirror_type, transverse),
        mirror,
        surface,
        incident,
        spectrum,
        amplitudes,
    )
    intensity_x = transverse_intensity[:n_profile]
    intensity_y = transverse_intensity[n_profile:]
    # The paper defines z_R through the spatial point of maximum focal-plane
    # intensity.  This matters for the linear TP's annular focal spot.
    offset_x = transverse[int(np.argmax(intensity_x))]
    offset_y = transverse[int(np.argmax(intensity_y))]
    if np.max(intensity_x) >= np.max(intensity_y):
        offset_y = 0.0
    else:
        offset_x = 0.0
    axial = np.linspace(-axial_half_width, axial_half_width, n_profile)
    axial_intensity = _propagate_components(
        _axial_observations(mirror_type, axial, offset_x, offset_y),
        mirror,
        surface,
        incident,
        spectrum,
        amplitudes,
    )

    computed = {
        "peak_intensity_W_cm2": max(np.max(intensity_x), np.max(intensity_y)) / 1e4,
        "fwhm_x_um": _fwhm(transverse, intensity_x) * 1e6,
        "fwhm_y_um": _fwhm(transverse, intensity_y) * 1e6,
        "rayleigh_length_um": _fwhm(axial, axial_intensity) * 0.5e6,
        "incident_energy_J": spectrum.recovered_energy(amplitudes, effective_area),
    }
    reference = PAPER_REFERENCE[polarisation][mirror_type]
    relative_difference = {key: computed[key] / reference[key] - 1 for key in reference}
    return {
        "case": {"mirror_type": mirror_type, "polarisation": polarisation},
        "configuration": {
            "n_radial": n_radial,
            "n_azimuthal": n_azimuthal,
            "n_profile": n_profile,
            "n_components": n_components,
            "incident_fwhm_mm": INCIDENT_FWHM * 1e3,
            "profile_half_width_um": profile_half_width * 1e6,
            "axial_half_width_um": axial_half_width * 1e6,
        },
        "computed": computed,
        "paper_reference": reference,
        "relative_difference": relative_difference,
    }


def run_benchmark(**kwargs):
    """Backward-compatible alias for the linear OAP90 benchmark."""
    result = run_case("OAP90", "linear", **kwargs)
    # Preserve the original diagnostic names for existing callers.
    result["computed"]["fwhm_meridional_um"] = result["computed"]["fwhm_x_um"]
    result["computed"]["fwhm_sagittal_um"] = result["computed"]["fwhm_y_um"]
    result["relative_difference"]["fwhm_meridional_um"] = result["relative_difference"][
        "fwhm_x_um"
    ]
    result["relative_difference"]["fwhm_sagittal_um"] = result["relative_difference"][
        "fwhm_y_um"
    ]
    return result


def run_suite(**kwargs):
    """Evaluate all six cases from Tables 1 and 2."""
    return {
        polarisation: {
            mirror_type: run_case(mirror_type, polarisation, **kwargs)
            for mirror_type in ("HNAP", "OAP90", "TP")
        }
        for polarisation in ("linear", "radial")
    }


def run_convergence(mirror_type="OAP90", polarisation="linear", **kwargs):
    """Run independent surface and spectrum refinements for one case."""
    common = dict(kwargs)
    common.setdefault("n_profile", 121)
    surface = [
        run_case(
            mirror_type,
            polarisation,
            n_radial=n_radial,
            n_azimuthal=2 * n_radial,
            n_components=31,
            **common,
        )
        for n_radial in (12, 18, 24)
    ]
    spectral = [
        run_case(
            mirror_type,
            polarisation,
            n_radial=24,
            n_azimuthal=48,
            n_components=n_components,
            **common,
        )
        for n_components in (15, 31, 47)
    ]
    return {"surface_quadrature": surface, "spectrum": spectral}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", choices=("HNAP", "OAP90", "TP"), default="OAP90")
    parser.add_argument(
        "--polarisation", choices=("linear", "radial"), default="linear"
    )
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--n-radial", type=int, default=24)
    parser.add_argument("--n-azimuthal", type=int, default=48)
    parser.add_argument("--n-profile", type=int, default=121)
    parser.add_argument("--n-components", type=int, default=31)
    parser.add_argument("--profile-half-width-um", type=float, default=1.5)
    parser.add_argument("--axial-half-width-um", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    common = {
        "n_profile": args.n_profile,
        "profile_half_width": args.profile_half_width_um * 1e-6,
        "axial_half_width": args.axial_half_width_um * 1e-6,
    }
    if args.convergence:
        results = run_convergence(args.mirror, args.polarisation, **common)
    elif args.suite:
        results = run_suite(
            n_radial=args.n_radial,
            n_azimuthal=args.n_azimuthal,
            n_components=args.n_components,
            **common,
        )
    else:
        results = run_case(
            args.mirror,
            args.polarisation,
            n_radial=args.n_radial,
            n_azimuthal=args.n_azimuthal,
            n_components=args.n_components,
            **common,
        )
    serialised = json.dumps(results, indent=2)
    print(serialised)
    if args.output:
        args.output.write_text(serialised + "\n")


if __name__ == "__main__":
    main()
