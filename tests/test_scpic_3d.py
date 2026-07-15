import numpy as np
import pytest
from scipy.integrate import quad

from paper_benchmark_3d import run_benchmark
from scpic.fields import C, LinearPolarisedSuperGaussian3D
from scpic.mirrors import ParabolicMirror3D
from scpic.pulse import (
    SuperGaussianSpectrum,
    electric_intensity,
    reconstruct_analytic_signal,
)
from scpic.solvers import evaluate_SC_3D


def test_oap90_geometry_area_and_chief_ray():
    mirror = ParabolicMirror3D(f0=0.1, D=0.08, mirror_type="OAP90")
    surface = mirror.surface_quadrature(n_radial=8, n_azimuthal=16)

    assert np.sum(surface.projected_weights) == pytest.approx(
        mirror.projected_area, rel=2e-14
    )
    assert mirror.effective_focal_length == pytest.approx(0.2)

    chief_xy = mirror.aperture_centre[None, :]
    normal = mirror._normals_from_xy(chief_xy)[0]
    incident_direction = np.array([0.0, 0.0, -1.0])
    reflected_direction = (
        incident_direction - 2 * np.dot(incident_direction, normal) * normal
    )
    np.testing.assert_allclose(reflected_direction, [-1, 0, 0], atol=1e-14)


def test_annular_surface_and_contours_have_consistent_orientation():
    mirror = ParabolicMirror3D(f0=0.1, D=0.08, mirror_type="TP", inner_diameter=0.02)
    surface = mirror.surface_quadrature(n_radial=8, n_azimuthal=16)
    outer = mirror.contour_quadrature(64, rim="outer")
    inner = mirror.contour_quadrature(64, rim="inner")

    assert np.sum(surface.projected_weights) == pytest.approx(
        mirror.projected_area, rel=2e-14
    )
    # In projection the inner boundary has the opposite orientation.
    outer_signed_area = 0.5 * np.sum(
        outer.points[:, 0] * outer.d_ell[:, 1] - outer.points[:, 1] * outer.d_ell[:, 0]
    )
    inner_signed_area = 0.5 * np.sum(
        inner.points[:, 0] * inner.d_ell[:, 1] - inner.points[:, 1] * inner.d_ell[:, 0]
    )
    assert outer_signed_area > 0
    assert inner_signed_area < 0


def test_super_gaussian_width_direction_and_effective_area():
    width = 0.18
    field = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
        width, wavelength=0.8e-6, spatial_order=16
    )
    points = np.array([[width / 2, 0, 0], [0, 0, 0]])
    electric, magnetic = field.fields(points)

    assert abs(electric[0, 0]) ** 2 / abs(electric[1, 0]) ** 2 == pytest.approx(0.5)
    np.testing.assert_allclose(magnetic[:, 1], -electric[:, 0] / C)
    numerical_area = (
        2
        * np.pi
        * quad(
            lambda radius: radius
            * np.exp(-2 * (radius / field.w0) ** field.spatial_order),
            0,
            3 * field.w0,
        )[0]
    )
    assert field.effective_area == pytest.approx(numerical_area, rel=1e-10)


def test_stratton_chu_focus_converges_and_respects_axisymmetry():
    wavelength = 0.8e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=8e-6, D=8e-6, mirror_type="HNAP")
    incident = LinearPolarisedSuperGaussian3D(
        w0=20e-6, wavelength=wavelength, spatial_order=16
    )
    observation = np.array([[0.0, 0.0, 0.0]])
    results = []
    for n_radial, n_azimuthal in [(12, 24), (20, 40)]:
        surface = mirror.surface_quadrature(n_radial, n_azimuthal)
        E_inc, B_inc = incident.fields(surface.points)
        electric, magnetic = evaluate_SC_3D(observation, surface, E_inc, B_inc, k)
        results.append((electric[0], magnetic[0]))

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=2e-7, atol=1e-9)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=2e-7, atol=1e-16)
    assert abs(results[1][0][1]) < 1e-12 * abs(results[1][0][0])
    assert abs(results[1][0][2]) < 1e-12 * abs(results[1][0][0])
    assert abs(results[1][1][0]) < 1e-12 * abs(results[1][1][1])
    assert abs(results[1][1][2]) < 1e-12 * abs(results[1][1][1])


def test_paper_oap_contour_term_is_negligible_at_focus():
    wavelength = 800e-9
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D.vallieres_oap90()
    surface = mirror.surface_quadrature(12, 24)
    contour = mirror.contour_quadrature(64)
    incident = LinearPolarisedSuperGaussian3D.from_intensity_fwhm(
        mirror.D,
        wavelength=wavelength,
        spatial_order=16,
        centre=(*mirror.aperture_centre, 0.0),
    )
    E_surface, B_surface = incident.fields(surface.points)
    _, B_contour = incident.fields(contour.points)
    observation = np.array([[0.0, 0.0, 0.0]])
    E_surface_only, _ = evaluate_SC_3D(observation, surface, E_surface, B_surface, k)
    E_with_contour, _ = evaluate_SC_3D(
        observation,
        surface,
        E_surface,
        B_surface,
        k,
        contours=contour,
        B_inc_contours=(B_contour,),
    )
    relative_change = np.linalg.norm(E_with_contour - E_surface_only) / np.linalg.norm(
        E_surface_only
    )
    assert relative_change < 1e-4


def test_spectrum_recovers_energy_and_reconstructs_positive_frequencies():
    spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
        n_components=17, total_energy=20.0
    )
    amplitudes = spectrum.component_amplitudes(effective_area=0.01)
    assert spectrum.recovered_energy(amplitudes, 0.01) == pytest.approx(20.0)

    components = np.zeros((17, 2, 3), dtype=complex)
    components[:, :, 0] = amplitudes[:, None]
    reconstructed = reconstruct_analytic_signal(
        components, spectrum.angular_frequencies, times=[0.0]
    )
    np.testing.assert_allclose(reconstructed[0], 2 * np.sum(components, axis=0))
    assert np.all(electric_intensity(reconstructed) > 0)


def test_workstation_paper_benchmark_recovers_reported_focal_dimensions():
    result = run_benchmark(
        n_radial=12,
        n_azimuthal=24,
        n_profile=61,
        n_components=15,
    )
    differences = result["relative_difference"]
    assert abs(differences["fwhm_meridional_um"]) < 0.05
    assert abs(differences["fwhm_sagittal_um"]) < 0.05
    assert abs(differences["rayleigh_length_um"]) < 0.10
    assert abs(differences["peak_intensity_W_cm2"]) < 0.10
