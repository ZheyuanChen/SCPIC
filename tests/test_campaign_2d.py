import json

import numpy as np

from scpic import (
    GaussianPulseSpectrum,
    IncidentFieldTM,
    ParabolicMirror2D,
    evaluate_SC_2D,
    generate_epoch2d_oap_pulse,
    propagate_broadband_2d,
    reconstruct_complex_envelope,
)
from scpic.fields import C


def _fwhm(samples, values):
    half = 0.5 * np.max(values)
    above = np.flatnonzero(values >= half)

    def crossing(low, high):
        fraction = (half - values[low]) / (values[high] - values[low])
        return samples[low] + fraction * (samples[high] - samples[low])

    left = crossing(above[0] - 1, above[0])
    right = crossing(above[-1], above[-1] + 1)
    return right - left


def _small_f2_model():
    wavelength = 800e-9
    effective_focal_length = 40e-6
    incident_radius = effective_focal_length / 4
    mirror = ParabolicMirror2D(
        f0=effective_focal_length / 2,
        D=6 * incident_radius,
        mirror_type="OAP90",
    )
    incident = IncidentFieldTM(incident_radius, wavelength)
    return wavelength, mirror, incident


def test_gaussian_pulse_spectrum_reconstructs_requested_intensity_fwhm():
    duration = 30e-15
    spectrum = GaussianPulseSpectrum.from_intensity_fwhm(
        intensity_fwhm=duration,
        minimum_period=500e-15,
    )
    carrier = spectrum.central_angular_frequency
    times = np.linspace(-80e-15, 80e-15, 1281)
    coefficients = spectrum.component_coefficients(1.0)[:, None]
    envelope = reconstruct_complex_envelope(
        coefficients,
        spectrum.angular_frequencies,
        times,
        carrier,
    )[:, 0]

    assert len(spectrum.angular_frequencies) % 2 == 1
    assert spectrum.period > 500e-15
    assert np.isclose(
        spectrum.angular_frequencies[len(spectrum.angular_frequencies) // 2],
        carrier,
    )
    assert abs(_fwhm(times, np.abs(envelope) ** 2) / duration - 1) < 2e-3


def test_vectorised_2d_solver_matches_single_observation_chunks():
    wavelength, mirror, incident = _small_f2_model()
    surface = mirror.get_surface(200)
    by_incident = incident.B_y(surface[0], surface[1], surface[5])
    dby_incident = incident.dBy_dn(
        surface[0], surface[1], surface[2], surface[3], surface[5]
    )
    x = np.linspace(-1e-6, 2e-6, 8)
    z = np.linspace(-0.8e-6, 0.9e-6, 8)

    reference = evaluate_SC_2D(
        x,
        z,
        *surface[:5],
        by_incident,
        dby_incident,
        2 * np.pi / wavelength,
        chunk_size=1,
    )
    vectorised = evaluate_SC_2D(
        x,
        z,
        *surface[:5],
        by_incident,
        dby_incident,
        2 * np.pi / wavelength,
        chunk_size=7,
    )
    np.testing.assert_allclose(vectorised, reference, rtol=2e-14, atol=1e-15)


def test_broadband_2d_reference_defines_focus_duration_and_time():
    wavelength, mirror, incident = _small_f2_model()
    duration = 30e-15
    peak_time = 60e-15
    spectrum = GaussianPulseSpectrum.from_intensity_fwhm(
        central_wavelength=wavelength,
        intensity_fwhm=duration,
        n_components=31,
    )
    times = np.linspace(0.0, 120e-15, 241)
    coordinate = np.array([-10e-9, 0.0, 10e-9])
    serial = propagate_broadband_2d(
        coordinate,
        coordinate,
        mirror,
        incident,
        spectrum,
        times,
        num_surface_points=240,
        carrier_angular_frequency=spectrum.central_angular_frequency,
        envelope_peak_time=peak_time,
        reference_point=(0.0, 0.0),
        reference_step=10e-9,
        reference_normalisation="complex",
        workers=1,
    )
    threaded = propagate_broadband_2d(
        coordinate,
        coordinate,
        mirror,
        incident,
        spectrum,
        times,
        num_surface_points=240,
        carrier_angular_frequency=spectrum.central_angular_frequency,
        envelope_peak_time=peak_time,
        reference_point=(0.0, 0.0),
        reference_step=10e-9,
        reference_normalisation="complex",
        workers=2,
    )
    intensity = np.abs(serial.electric[:, 1, 1, 1]) ** 2

    assert abs(times[np.argmax(intensity)] - peak_time) <= 0.5e-15
    assert abs(_fwhm(times, intensity) / duration - 1) < 3e-3
    np.testing.assert_allclose(threaded.electric, serial.electric, rtol=1e-13)
    np.testing.assert_allclose(threaded.magnetic, serial.magnetic, rtol=1e-13)


def test_focus_reference_predicts_earlier_boundary_arrival():
    wavelength, mirror, incident = _small_f2_model()
    spectrum = GaussianPulseSpectrum.from_intensity_fwhm(
        central_wavelength=wavelength,
        intensity_fwhm=20e-15,
        n_components=31,
    )
    focus_distance = 4e-6
    focus_time = 70e-15
    x = focus_distance + np.array([-10e-9, 0.0, 10e-9])
    z = np.linspace(-0.4e-6, 0.4e-6, 9)
    times = np.linspace(30e-15, 90e-15, 241)
    result = propagate_broadband_2d(
        x,
        z,
        mirror,
        incident,
        spectrum,
        times,
        num_surface_points=240,
        carrier_angular_frequency=spectrum.central_angular_frequency,
        envelope_peak_time=focus_time,
        reference_point=(0.0, 0.0),
        reference_step=10e-9,
        reference_normalisation="complex",
    )
    on_axis = np.abs(result.electric[:, 1, len(z) // 2, 1]) ** 2
    expected = focus_time - focus_distance / C
    assert abs(times[np.argmax(on_axis)] - expected) < 1.0e-15


def test_campaign_generator_writes_epoch_shape_orientation_and_manifest(tmp_path):
    generated = generate_epoch2d_oap_pulse(
        tmp_path,
        intensity_fwhm=30e-15,
        focus_distance=4e-6,
        boundary_peak_time=50e-15,
        effective_focal_length=0.4e-3,
        time_end=200e-15,
        n_time=101,
        transverse_min=-4e-6,
        transverse_max=4e-6,
        n_transverse=41,
        n_surface=240,
        n_components=17,
        workers=2,
    )
    manifest = json.loads(generated.manifest_file.read_text())

    assert generated.shape == (101, 41)
    assert generated.amplitude_file.stat().st_size == 101 * 41 * 8
    assert generated.phase_file.stat().st_size == 101 * 41 * 8
    assert manifest["orientation"]["epoch_propagation"] == "+x from x_min"
    assert manifest["spectrum"]["reference_normalisation"].startswith("complex")
    assert manifest["profile_diagnostics"]["phase_winding_cell_count"] == 0
    assert manifest["configuration"]["phase_amplitude_floor"] == 1e-3
    assert (
        max(manifest["profile_diagnostics"]["maximum_stored_phase_step_by_axis_rad"])
        < np.pi
    )
    assert (
        abs(
            manifest["profile_diagnostics"]["boundary_peak_time_realised_s"] / 50e-15
            - 1
        )
        < 0.03
    )
