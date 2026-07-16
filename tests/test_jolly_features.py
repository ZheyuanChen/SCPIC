import numpy as np
import pytest
from scipy.integrate import trapezoid

from scpic import (
    ChromaticZernikePhase,
    FiniteRayleighTM01Beam3D,
    LinearPolarisedSuperGaussian3D,
    ParabolicMirror3D,
    ParaxialGaussLaguerreBeam3D,
    RadiallyPolarisedSuperGaussian3D,
    SuperGaussianSpectrum,
    TM01RadiallyPolarisedBeam3D,
    epoch_phase_diagnostics,
    propagate_broadband_3d,
)
from scpic.fields import C
from scpic.solvers import evaluate_SC_3D


def test_jolly_angular_dispersion_has_the_requested_edge_group_delay():
    carrier = 10.0
    pulse_front_tilt = 2.5e-15
    phase = ChromaticZernikePhase.jolly_angular_dispersion(
        1.0,
        pulse_front_tilt,
        carrier_angular_frequency=carrier,
    )
    points = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    np.testing.assert_array_equal(phase(points, carrier), 0.0)
    omega = 11.0
    expected_edge_phase = pulse_front_tilt * omega * (omega - carrier) / carrier
    np.testing.assert_allclose(
        phase(points, omega),
        [-expected_edge_phase, 0.0, expected_edge_phase],
    )

    step = 1e-4
    group_delay = (phase(points, carrier + step) - phase(points, carrier - step)) / (
        2 * step
    )
    np.testing.assert_allclose(
        group_delay,
        [-pulse_front_tilt, 0.0, pulse_front_tilt],
        rtol=1e-10,
        atol=1e-25,
    )


def test_jolly_chromatic_curvature_and_trefoil_follow_osa_modes():
    carrier = 12.0
    pulse_front_curvature = 4e-15
    curvature = ChromaticZernikePhase.jolly_chromatic_curvature(
        1.0,
        pulse_front_curvature,
        carrier_angular_frequency=carrier,
    )
    centre_and_edge = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    step = 1e-4
    group_delay = (
        curvature(centre_and_edge, carrier + step)
        - curvature(centre_and_edge, carrier - step)
    ) / (2 * step)
    np.testing.assert_allclose(
        group_delay,
        [0.0, pulse_front_curvature],
        rtol=1e-10,
        atol=1e-25,
    )

    trefoil_delay = 6e-15
    trefoil = ChromaticZernikePhase.jolly_chromatic_trefoil(
        1.0,
        trefoil_delay,
        carrier_angular_frequency=carrier,
    )
    angles = np.array([0.0, np.pi / 3, 2 * np.pi / 3])
    points = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    omega = carrier + 1.0
    common = trefoil_delay * omega * (omega - carrier) / (2 * carrier)
    np.testing.assert_allclose(trefoil(points, omega), [common, -common, common])


@pytest.mark.parametrize(
    "factory",
    [
        lambda phase: LinearPolarisedSuperGaussian3D(5.0, spatio_spectral_phase=phase),
        lambda phase: RadiallyPolarisedSuperGaussian3D(
            5.0, spatio_spectral_phase=phase
        ),
        lambda phase: TM01RadiallyPolarisedBeam3D(5.0, spatio_spectral_phase=phase),
        lambda phase: FiniteRayleighTM01Beam3D(5.0, spatio_spectral_phase=phase),
        lambda phase: ParaxialGaussLaguerreBeam3D(5.0, spatio_spectral_phase=phase),
    ],
)
def test_spatio_spectral_phase_multiplies_all_incident_vector_fields(factory):
    phase_value = 0.37

    def phase(points, angular_frequency):
        assert angular_frequency == pytest.approx(2.0 * C)
        return np.full(len(points), phase_value)

    points = np.array([[0.8, 0.2, 0.0], [-0.4, 1.0, 0.1]])
    phased = factory(phase)
    reference = factory(None)
    electric, magnetic = phased.fields(points, k=2.0)
    electric_reference, magnetic_reference = reference.fields(points, k=2.0)
    factor = np.exp(1j * phase_value)
    np.testing.assert_allclose(electric, factor * electric_reference)
    np.testing.assert_allclose(magnetic, factor * magnetic_reference)


def test_broadband_propagation_evaluates_chromatic_phase_at_each_frequency():
    seen_frequencies = []

    def phase(points, angular_frequency):
        seen_frequencies.append(angular_frequency)
        return 1e-3 * points[:, 0] * angular_frequency / C

    mirror = ParabolicMirror3D(f0=5e-6, D=4e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(4, 8)
    incident = LinearPolarisedSuperGaussian3D(
        w0=8e-6,
        spatio_spectral_phase=phase,
    )
    spectrum = SuperGaussianSpectrum.from_wavelength_bandwidth(
        central_wavelength=1e-6,
        wavelength_fwhm=40e-9,
        total_energy=1e-6,
        n_components=3,
    )
    result = propagate_broadband_3d(
        np.zeros((1, 3)),
        surface,
        incident,
        spectrum,
        [0.0],
    )

    np.testing.assert_allclose(seen_frequencies, spectrum.angular_frequencies)
    assert np.all(np.isfinite(result.electric))
    assert np.all(np.isfinite(result.magnetic))


def test_stratton_chu_maps_jolly_input_tilt_to_opposite_spectral_centroids():
    wavelength = 0.8e-6
    carrier = 2 * np.pi * C / wavelength
    mirror = ParabolicMirror3D(f0=8e-6, D=8e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(10, 20)
    chromatic_tilt = ChromaticZernikePhase.jolly_angular_dispersion(
        4e-6,
        3e-15,
        carrier_angular_frequency=carrier,
    )
    incident = LinearPolarisedSuperGaussian3D(
        w0=4e-6,
        spatial_order=2,
        spatio_spectral_phase=chromatic_tilt,
    )
    x = np.linspace(-1.5e-6, 1.5e-6, 101)
    observations = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    centroids = []
    for frequency_factor in (0.94, 1.0, 1.06):
        k = carrier * frequency_factor / C
        incident_fields = incident.fields(surface.points, k=k)
        electric, _ = evaluate_SC_3D(
            observations,
            surface,
            *incident_fields,
            k,
            chunk_size=32,
        )
        intensity = np.sum(np.abs(electric) ** 2, axis=1)
        centroids.append(trapezoid(x * intensity, x) / trapezoid(intensity, x))

    assert centroids[0] < -0.08e-6
    assert abs(centroids[1]) < 1e-10
    assert centroids[2] > 0.08e-6
    assert centroids[2] == pytest.approx(-centroids[0], rel=0.02)


def test_epoch_phase_diagnostics_detects_a_resolved_vortex():
    axis = np.array([-1.5, -0.5, 0.5, 1.5])
    x, y = np.meshgrid(axis, axis, indexing="ij")
    vortex = x + 1j * y
    diagnostics = epoch_phase_diagnostics(vortex)

    assert diagnostics.has_phase_singularity
    assert diagnostics.winding_cell_count == 1
    assert diagnostics.low_amplitude_fraction == pytest.approx(0.0)

    smooth = np.exp(1j * (0.2 * x - 0.1 * y))
    smooth_diagnostics = epoch_phase_diagnostics(smooth)
    assert not smooth_diagnostics.has_phase_singularity
    assert smooth_diagnostics.maximum_reliable_phase_step == pytest.approx(0.2)


def test_chromatic_phase_validation_is_explicit():
    with pytest.raises(ValueError, match="carrier_angular_frequency"):
        ChromaticZernikePhase(1.0, {}, carrier_angular_frequency=0.0)
    with pytest.raises(ValueError, match="azimuthal_index"):
        ChromaticZernikePhase.jolly_angular_dispersion(
            1.0,
            1e-15,
            carrier_angular_frequency=1.0,
            azimuthal_index=0,
        )

    invalid = ChromaticZernikePhase(
        1.0,
        {(1, 1): lambda omega: np.nan},
        carrier_angular_frequency=1.0,
    )
    with pytest.raises(ValueError, match="not finite"):
        invalid(np.array([[0.0, 0.0, 0.0]]), 1.0)
