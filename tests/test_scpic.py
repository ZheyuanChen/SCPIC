import os

import numpy as np
import pytest
from scipy.optimize import curve_fit

from scpic.export import (
    epoch_amplitude_phase,
    export_epoch_profile,
    export_field_binary,
)
from scpic.fields import C, IncidentFieldTM, electric_from_magnetic_tm
from scpic.mirrors import ParabolicMirror2D
from scpic.solvers import evaluate_SC_2D


def test_mirror_normals_and_quadrature_weights():
    mirror = ParabolicMirror2D(f0=10e-6, D=15e-6, mirror_type="OAP90")
    x_m, _, nx, nz, dl, _ = mirror.get_surface(num_points=100)

    np.testing.assert_allclose(np.hypot(nx, nz), 1.0, rtol=1e-14)
    assert dl.shape == x_m.shape
    assert dl[0] < dl[1]
    assert dl[-1] < dl[-2]


def test_incident_field_amplitude_and_direction():
    field = IncidentFieldTM(w0=5e-6, wavelength=1e-6, E0=1.0)
    x_center = 10e-6
    By_center = field.B_y(x_center, 0.0, x_center=x_center)
    By_edge = field.B_y(x_center + field.w0, 0.0, x_center=x_center)

    assert np.isclose(abs(By_center), 1.0 / C)
    assert np.isclose(abs(By_edge / By_center), np.exp(-1))
    # With exp(-i omega t), exp(-i k z) propagates towards decreasing z.
    np.testing.assert_allclose(
        field.B_y(x_center, 0.25e-6, x_center=x_center) / By_center, -1j
    )


def test_electric_field_recovery_for_plane_wave():
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    x = np.linspace(-wavelength, wavelength, 1001)
    z = np.linspace(-0.1 * wavelength, 0.1 * wavelength, 3)
    By = np.exp(-1j * k * x[:, None]) * np.ones((1, z.size))

    Ex, Ez = electric_from_magnetic_tm(By, x, z, k)

    np.testing.assert_allclose(Ex, 0.0, atol=2e-7)
    np.testing.assert_allclose(Ez[2:-2], C * By[2:-2], rtol=3e-5)


def test_epoch_phase_conversion_reconstructs_physical_field():
    field = np.exp(1j * np.array([0.0, 0.4, -1.2, np.pi]))
    amplitude, phase, scale = epoch_amplitude_phase(field)

    for omega_t in np.linspace(-2 * np.pi, 2 * np.pi, 11):
        expected = np.real(field * np.exp(-1j * omega_t))
        actual = scale * amplitude * np.sin(omega_t + phase)
        np.testing.assert_allclose(actual, expected, atol=2e-15)


def test_epoch_phase_is_unwrapped_for_linear_interpolation():
    source_phase = np.linspace(-4 * np.pi, 4 * np.pi, 33)
    field = np.exp(1j * source_phase)
    amplitude, phase, scale = epoch_amplitude_phase(field)

    assert np.max(np.abs(np.diff(phase))) < np.pi
    omega_t = 0.37
    expected = np.real(field * np.exp(-1j * omega_t))
    actual = scale * amplitude * np.sin(omega_t + phase)
    np.testing.assert_allclose(actual, expected, atol=4e-15)

    _, wrapped, _ = epoch_amplitude_phase(field, unwrap_phase=False)
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_epoch_export_is_float64_headerless_and_c_ordered(tmp_path):
    field = np.array([[1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j], [1.0 - 1.0j, 0.5j, 0.0j]])
    result = export_epoch_profile(tmp_path, field)

    assert os.path.getsize(result.amplitude_file) == field.size * 8
    assert os.path.getsize(result.phase_file) == field.size * 8
    expected_amplitude, expected_phase, expected_scale = epoch_amplitude_phase(field)
    np.testing.assert_array_equal(
        np.fromfile(result.amplitude_file, dtype=np.float64),
        expected_amplitude.ravel(order="C"),
    )
    np.testing.assert_array_equal(
        np.fromfile(result.phase_file, dtype=np.float64),
        expected_phase.ravel(order="C"),
    )
    assert result.field_scale == expected_scale


def test_epoch_export_rejects_wrong_precision_and_undersized_scale(tmp_path):
    field = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    with pytest.raises(ValueError, match="smaller"):
        epoch_amplitude_phase(field, field_scale=1.0)
    with pytest.raises(ValueError, match="float64"):
        export_field_binary(tmp_path / "field", field, dtype=np.float32)
    with pytest.raises(ValueError, match="phase_reference"):
        epoch_amplitude_phase(field, phase_reference=np.nan)
    with pytest.raises(TypeError, match="unwrap_phase"):
        epoch_amplitude_phase(field, unwrap_phase="yes")


def test_paraxial_pec_reflector_matches_gaussian_waist():
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    f0 = 200e-6
    w_inc = 20e-6
    # Six beam radii suppress aperture truncation in this analytical test.
    mirror = ParabolicMirror2D(f0=f0, D=120e-6, mirror_type="OAP90")
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=2500)
    field = IncidentFieldTM(w0=w_inc, wavelength=wavelength, E0=1.0)
    By_inc = field.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = field.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)
    z = np.linspace(-15e-6, 15e-6, 161)

    By = evaluate_SC_2D(
        np.zeros_like(z),
        z,
        x_m,
        z_m,
        nx,
        nz,
        dl,
        By_inc,
        dBy_dn_inc,
        k,
    )

    def gaussian(coord, amplitude, waist):
        return amplitude * np.exp(-((coord / waist) ** 2))

    fitted, _ = curve_fit(gaussian, z, np.abs(By), p0=[np.max(np.abs(By)), 6e-6])
    numerical_waist = abs(fitted[1])
    theoretical_waist = 2 * wavelength * f0 / (np.pi * w_inc)
    assert abs(numerical_waist / theoretical_waist - 1.0) < 0.01


def test_generic_kirchhoff_model_requires_matching_derivative_shape():
    mirror = ParabolicMirror2D(f0=10e-6, D=5e-6)
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=20)
    field = IncidentFieldTM(w0=3e-6, wavelength=1e-6)
    By_inc = field.B_y(x_m, z_m, x_center=x_center)
    with pytest.raises(ValueError, match="dBy_dn_inc"):
        evaluate_SC_2D(
            np.array([0.0]),
            np.array([0.0]),
            x_m,
            z_m,
            nx,
            nz,
            dl,
            By_inc,
            np.ones(3),
            field.k,
            boundary_model="kirchhoff",
        )
