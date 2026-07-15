"""Reproduce the linearly polarised OAP90 case of Vallières et al. (2023).

The defaults are deliberately workstation-sized.  Increase the quadrature,
profile and spectral component counts for a publication-quality convergence
study; the command prints its complete configuration with the results.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from scpic.fields import C, LinearPolarisedSuperGaussian3D
from scpic.mirrors import ParabolicMirror3D
from scpic.pulse import (
    SuperGaussianSpectrum,
    electric_intensity,
    reconstruct_analytic_signal,
)
from scpic.solvers import evaluate_SC_3D

PAPER_REFERENCE = {
    "peak_intensity_W_cm2": 2.66e23,
    "fwhm_meridional_um": 0.60,
    "fwhm_sagittal_um": 0.52,
    "rayleigh_length_um": 0.69,
}


def _fwhm(coordinate, values):
    coordinate = np.asarray(coordinate)
    values = np.asarray(values)
    peak = int(np.argmax(values))
    half = values[peak] / 2

    crossings = []
    for indices in (range(peak - 1, -1, -1), range(peak, len(values) - 1)):
        crossing = None
        for index in indices:
            neighbour = index + 1
            if (values[index] - half) * (values[neighbour] - half) <= 0:
                fraction = (half - values[index]) / (values[neighbour] - values[index])
                crossing = coordinate[index] + fraction * (
                    coordinate[neighbour] - coordinate[index]
                )
                break
        crossings.append(crossing)
    if any(value is None for value in crossings):
        return np.nan
    return abs(crossings[1] - crossings[0])


def run_benchmark(
    *,
    n_radial=24,
    n_azimuthal=48,
    n_profile=121,
    n_components=31,
    profile_half_width=1.5e-6,
    axial_half_width=2.0e-6,
):
    mirror = ParabolicMirror3D.vallieres_oap90()
    surface = mirror.surface_quadrature(n_radial, n_azimuthal)
    incident = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
        mirror.D,
        spatial_order=16,
        centre=(*mirror.aperture_centre, 0.0),
    )
    spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
        central_wavelength=800e-9,
        wavelength_fwhm=90e-9,
        spectral_order=7,
        total_energy=20.0,
        n_components=n_components,
    )
    amplitudes = spectrum.component_amplitudes(incident.effective_area)

    transverse = np.linspace(-profile_half_width, profile_half_width, n_profile)
    axial = np.linspace(-axial_half_width, axial_half_width, n_profile)
    # The focused chief ray travels along -x.  z is the OAP meridional axis
    # and y is the sagittal axis.
    observations = np.concatenate(
        (
            np.column_stack(
                (np.zeros_like(transverse), np.zeros_like(transverse), transverse)
            ),
            np.column_stack(
                (np.zeros_like(transverse), transverse, np.zeros_like(transverse))
            ),
            np.column_stack((axial, np.zeros_like(axial), np.zeros_like(axial))),
        )
    )
    components = np.empty((n_components, len(observations), 3), dtype=complex)
    for index, (omega, amplitude) in enumerate(
        zip(spectrum.angular_frequencies, amplitudes)
    ):
        k = omega / C
        E_inc, B_inc = incident.fields(
            surface.points,
            k=k,
            amplitude=amplitude,
            # r_focus - z_surface = 2*f0 on the parent paraboloid.
            spectral_phase=-2 * k * mirror.f0,
        )
        components[index], _ = evaluate_SC_3D(observations, surface, E_inc, B_inc, k)

    analytic = reconstruct_analytic_signal(
        components, spectrum.angular_frequencies, [0.0]
    )[0]
    intensity = electric_intensity(analytic)
    meridional = intensity[:n_profile]
    sagittal = intensity[n_profile : 2 * n_profile]
    axial_intensity = intensity[2 * n_profile :]
    peak = max(np.max(meridional), np.max(sagittal))

    results = {
        "configuration": {
            "n_radial": n_radial,
            "n_azimuthal": n_azimuthal,
            "n_profile": n_profile,
            "n_components": n_components,
            "profile_half_width_um": profile_half_width * 1e6,
            "axial_half_width_um": axial_half_width * 1e6,
        },
        "computed": {
            "peak_intensity_W_cm2": peak / 1e4,
            "fwhm_meridional_um": _fwhm(transverse, meridional) * 1e6,
            "fwhm_sagittal_um": _fwhm(transverse, sagittal) * 1e6,
            "rayleigh_length_um": _fwhm(axial, axial_intensity) * 0.5e6,
            "incident_energy_J": spectrum.recovered_energy(
                amplitudes, incident.effective_area
            ),
        },
        "paper_reference": PAPER_REFERENCE,
    }
    results["relative_difference"] = {
        key: (results["computed"][key] / PAPER_REFERENCE[key] - 1)
        for key in (
            "peak_intensity_W_cm2",
            "fwhm_meridional_um",
            "fwhm_sagittal_um",
            "rayleigh_length_um",
        )
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-radial", type=int, default=24)
    parser.add_argument("--n-azimuthal", type=int, default=48)
    parser.add_argument("--n-profile", type=int, default=121)
    parser.add_argument("--n-components", type=int, default=31)
    parser.add_argument("--profile-half-width-um", type=float, default=1.5)
    parser.add_argument("--axial-half-width-um", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_benchmark(
        n_radial=args.n_radial,
        n_azimuthal=args.n_azimuthal,
        n_profile=args.n_profile,
        n_components=args.n_components,
        profile_half_width=args.profile_half_width_um * 1e-6,
        axial_half_width=args.axial_half_width_um * 1e-6,
    )
    serialised = json.dumps(results, indent=2)
    print(serialised)
    if args.output:
        args.output.write_text(serialised + "\n")


if __name__ == "__main__":
    main()
