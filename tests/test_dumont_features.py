import numpy as np
import pytest
from scipy.integrate import trapezoid

from scpic import (
    FiniteRayleighTM01Beam3D,
    LinearPolarisedSuperGaussian3D,
    ParabolicMirror3D,
    ParaxialGaussLaguerreBeam3D,
    SampledSpectrum,
    SuperGaussianSpectrum,
    TM01RadiallyPolarisedBeam3D,
    electromagnetic_energy_density,
    integrated_field_energy,
    integrated_poynting_flux,
    observation_partition,
    propagate_broadband_3d,
    reconstruct_analytic_signal,
    relative_energy_error,
    surface_quadrature_convergence,
    time_domain_maxwell_residuals,
)
from scpic.fields import C
from scpic.solvers import evaluate_SC_3D

EPSILON_0 = 8.854_187_8128e-12


def test_sampled_wavelength_spectrum_applies_jacobian_and_spectral_phase():
    wavelengths = np.array([700e-9, 800e-9, 900e-9])
    spectrum = SampledSpectrum.from_wavelength_samples(
        wavelengths,
        np.ones(3),
        total_energy=2.5,
        n_components=3,
    )
    # A flat dE/dlambda is not flat in omega: dE/domega scales as omega^-2.
    expected_ratio = (900 / 700) ** 2
    assert spectrum.relative_energy_density[0] / spectrum.relative_energy_density[
        -1
    ] == pytest.approx(expected_ratio)

    phased = spectrum.with_spectral_phase(
        lambda omega: 2e-30 * (omega - np.mean(omega)) ** 2
    )
    coefficients = phased.component_coefficients(0.01)
    np.testing.assert_allclose(np.angle(coefficients), phased.spectral_phase)
    assert phased.recovered_energy(coefficients, 0.01) == pytest.approx(2.5)


def test_exact_super_gaussian_conversion_is_opt_in():
    narrowband = SuperGaussianSpectrum.from_wavelength_bandwidth(n_components=9)
    exact = SuperGaussianSpectrum.from_wavelength_bandwidth(
        n_components=9,
        conversion="exact_wavelength_density",
    )
    assert narrowband.conversion == "narrowband"
    assert exact.conversion == "exact_wavelength_density"
    assert not np.allclose(
        narrowband.angular_frequencies,
        exact.angular_frequencies,
        rtol=1e-12,
        atol=0.0,
    )
    assert narrowband.recovered_energy(
        narrowband.component_amplitudes(0.02), 0.02
    ) == pytest.approx(20.0)


def test_finite_rayleigh_tm01_reduces_to_collimated_model_at_waist():
    wavelength = 800e-9
    w0 = 20e-6
    points = np.array([[0.0, 0.0, 0.0], [w0 / 3, 0.0, 0.0], [0.0, w0 / 2, 0.0]])
    collimated = TM01RadiallyPolarisedBeam3D(w0=w0, wavelength=wavelength)
    finite = FiniteRayleighTM01Beam3D(w0=w0, wavelength=wavelength)
    reference = collimated.fields(points, amplitude=2.3)
    result = finite.fields(points, amplitude=2.3)
    np.testing.assert_allclose(result[0], reference[0], rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(result[1], reference[1], rtol=2e-15, atol=1e-23)


def test_gauss_laguerre_effective_area_and_power_are_conserved():
    wavelength = 1e-6
    w0 = 8e-6
    field = ParaxialGaussLaguerreBeam3D(
        w0,
        {0: 1.0, 1: 0.5j},
        wavelength=wavelength,
    )
    k = 2 * np.pi / wavelength
    rayleigh_range = k * w0**2 / 2
    radius = np.linspace(0, 6 * w0, 5001)
    numerical_areas = []
    for longitudinal in (0.0, rayleigh_range):
        points = np.column_stack(
            (radius, np.zeros_like(radius), -np.full_like(radius, longitudinal))
        )
        electric, magnetic = field.fields(points)
        np.testing.assert_allclose(magnetic, np.cross(field.direction, electric) / C)
        radial_density = np.sum(np.abs(electric) ** 2, axis=1)
        numerical_areas.append(2 * np.pi * trapezoid(radial_density * radius, radius))
    assert numerical_areas[0] == pytest.approx(field.effective_area, rel=2e-6)
    assert numerical_areas[1] == pytest.approx(field.effective_area, rel=2e-6)


def test_fourmaux_transmission_parabola_geometry_matches_paper():
    mirror = ParabolicMirror3D.fourmaux_tp_2025()
    assert mirror.f0 == pytest.approx(5.65e-3)
    assert mirror.D == pytest.approx(65e-3)
    assert mirror.inner_diameter == pytest.approx(24.5e-3)
    angles = np.degrees(mirror.focusing_angle_range)
    np.testing.assert_allclose(angles, [38.3, 85.4], atol=0.15)
    assert mirror.generalized_numerical_aperture == pytest.approx(0.96, abs=0.01)


def test_time_domain_maxwell_residuals_for_sampled_plane_wave():
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    omega = k * C
    x = np.linspace(-0.1, 0.1, 5)
    y = np.linspace(-0.1, 0.1, 5)
    z = np.linspace(-0.25, 0.25, 41)
    times = np.linspace(-0.25 / C, 0.25 / C, 41)
    phase = k * z[None, None, None, :] - omega * times[:, None, None, None]
    carrier = np.broadcast_to(np.cos(phase), (len(times), len(x), len(y), len(z)))
    electric = np.zeros((*carrier.shape, 3))
    magnetic = np.zeros_like(electric)
    electric[..., 0] = carrier
    magnetic[..., 1] = carrier / C
    residuals = time_domain_maxwell_residuals(
        electric,
        magnetic,
        times,
        (x, y, z),
        omega,
        trim=1,
        time_trim=1,
    )
    assert residuals.divergence_e < 1e-13
    assert residuals.divergence_b < 1e-13
    assert residuals.faraday < 0.003
    assert residuals.ampere < 0.003


def test_energy_and_flux_integrals_have_correct_phasor_factors():
    x = np.linspace(-1.0, 1.0, 11)
    y = np.linspace(-1.5, 1.5, 13)
    z = np.linspace(-2.0, 2.0, 15)
    electric_plane = np.zeros((len(x), len(y), 3), dtype=complex)
    magnetic_plane = np.zeros_like(electric_plane)
    electric_plane[..., 0] = 2.0
    magnetic_plane[..., 1] = 2.0 / C
    expected_flux = 0.5 * EPSILON_0 * C * 2.0**2 * 2.0 * 3.0
    assert integrated_poynting_flux(
        electric_plane, magnetic_plane, (x, y), (0, 0, 1)
    ) == pytest.approx(expected_flux)

    electric_volume = np.broadcast_to(
        electric_plane[:, :, None, :], (len(x), len(y), len(z), 3)
    )
    magnetic_volume = np.broadcast_to(
        magnetic_plane[:, :, None, :], electric_volume.shape
    )
    expected_density = 0.5 * EPSILON_0 * 2.0**2
    expected_energy = expected_density * 2.0 * 3.0 * 4.0
    density = electromagnetic_energy_density(electric_volume, magnetic_volume)
    np.testing.assert_allclose(density, expected_density)
    assert integrated_field_energy(
        electric_volume, magnetic_volume, (x, y, z)
    ) == pytest.approx(expected_energy)
    assert relative_energy_error(
        expected_energy, expected_energy * 0.98
    ) == pytest.approx(-0.02)


def test_memory_bounded_broadband_propagation_matches_direct_reconstruction():
    wavelength = 1e-6
    mirror = ParabolicMirror3D(f0=5e-6, D=4e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(5, 10)
    incident = LinearPolarisedSuperGaussian3D(w0=8e-6)
    spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
        central_wavelength=wavelength,
        wavelength_fwhm=50e-9,
        total_energy=1e-6,
        n_components=3,
    ).with_spectral_phase([0.1, -0.2, 0.3])
    observations = np.array([[0.0, 0.0, 0.0], [0.1e-6, 0.0, 0.0], [0.0, 0.1e-6, 0.0]])
    result = propagate_broadband_3d(
        observations,
        surface,
        incident,
        spectrum,
        [0.0],
        observation_chunk_size=2,
    )

    coefficients = spectrum.component_coefficients(incident.effective_area)
    components_e = []
    components_b = []
    for omega, coefficient in zip(spectrum.angular_frequencies, coefficients):
        k = omega / C
        incident_fields = incident.fields(surface.points, k=k, amplitude=coefficient)
        electric, magnetic = evaluate_SC_3D(observations, surface, *incident_fields, k)
        components_e.append(electric)
        components_b.append(magnetic)
    reference_e = reconstruct_analytic_signal(
        np.asarray(components_e), spectrum.angular_frequencies, [0.0]
    )
    reference_b = reconstruct_analytic_signal(
        np.asarray(components_b), spectrum.angular_frequencies, [0.0]
    )
    assert (result.start, result.stop) == (0, len(observations))
    np.testing.assert_allclose(result.electric, reference_e)
    np.testing.assert_allclose(result.magnetic, reference_b)
    assert observation_partition(10, rank=0, size=3) == slice(0, 4)
    assert observation_partition(10, rank=2, size=3) == slice(7, 10)


def test_surface_convergence_uses_combined_field_metric():
    wavelength = 0.8e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=8e-6, D=8e-6, mirror_type="HNAP")
    incident = LinearPolarisedSuperGaussian3D(w0=20e-6, wavelength=wavelength)
    result = surface_quadrature_convergence(
        np.zeros((1, 3)),
        mirror,
        incident,
        k,
        orders=((6, 12), (8, 16), (12, 24)),
        rtol=1e9,
    )
    assert result.converged
    assert len(result.levels) == 2
    assert np.isfinite(result.levels[-1].relative_combined_change)
    assert result.electric.shape == (1, 3)
    assert result.magnetic.shape == (1, 3)
