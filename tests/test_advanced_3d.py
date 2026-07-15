import importlib.util
import sys
import types

import numpy as np
import pytest

from scpic.diagnostics import maxwell_residuals
from scpic.fields import (
    C,
    LinearPolarisedSuperGaussian3D,
    RadiallyPolarisedSuperGaussian3D,
    TM01RadiallyPolarisedBeam3D,
    ZernikeWavefront,
)
from scpic.mirrors import ParabolicMirror3D
from scpic.solvers import evaluate_SC_3D


def test_osa_zernike_values_and_frequency_scaled_wavefront_phase():
    coefficient = 25e-9
    wavefront = ZernikeWavefront(
        pupil_radius=1.0,
        coefficients={(2, 0): coefficient},
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        wavefront(points),
        coefficient * np.sqrt(3) * np.array([-1.0, 1.0]),
    )

    field = LinearPolarisedSuperGaussian3D(
        w0=10.0,
        direction=(0.0, 0.0, -1.0),
        wavefront_opd=wavefront,
    )
    electric_k, _ = field.fields(points, k=2.0)
    electric_2k, _ = field.fields(points, k=4.0)
    phase_k = np.angle(electric_k[1, 0] / electric_k[0, 0])
    phase_2k = np.angle(electric_2k[1, 0] / electric_2k[0, 0])
    assert phase_2k == pytest.approx(2 * phase_k, abs=1e-15)

    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(12)
    radius = (radial_nodes + 1) / 2
    radial_weights = radial_weights / 2
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    radius_grid, theta_grid = np.meshgrid(radius, theta, indexing="ij")
    pupil_points = np.column_stack(
        (
            (radius_grid * np.cos(theta_grid)).ravel(),
            (radius_grid * np.sin(theta_grid)).ravel(),
            np.zeros(radius_grid.size),
        )
    )
    unit_mode = ZernikeWavefront(1.0, {(3, -1): 1.0})(pupil_points).reshape(
        radius_grid.shape
    )
    disk_mean_square = 2 * np.sum(
        radial_weights * radius * np.mean(unit_mode**2, axis=1)
    )
    assert disk_mean_square == pytest.approx(1.0, rel=2e-14)


def test_radial_polarisation_and_magnetic_direction():
    field = RadiallyPolarisedSuperGaussian3D(w0=10.0, wavelength=1.0)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    electric, magnetic = field.fields(points)

    assert electric[0, 0].real > 0
    assert electric[0, 0].imag == pytest.approx(0.0)
    assert electric[0, 1] == pytest.approx(0.0)
    assert electric[1, 0] == pytest.approx(0.0)
    assert electric[1, 1].real > 0
    assert electric[1, 1].imag == pytest.approx(0.0)
    np.testing.assert_allclose(magnetic, np.cross(field.direction, electric) / C)
    np.testing.assert_array_equal(electric[2], 0.0)


def test_tm01_matches_vallieres_incident_field_equations():
    wavelength = 800e-9
    k = 2 * np.pi / wavelength
    w0 = 107e-3
    amplitude = 3.5
    field = TM01RadiallyPolarisedBeam3D(
        w0=w0,
        wavelength=wavelength,
        direction=(0.0, 0.0, -1.0),
    )
    points = np.array([[w0 / np.sqrt(2), 0.0, 0.0], [0.0, 0.0, 0.0]])
    electric, magnetic = field.fields(points, amplitude=amplitude)

    radius = points[0, 0]
    factor = 2 * amplitude / (k * w0**2) * np.exp(-((radius / w0) ** 2))
    assert electric[0, 0] == pytest.approx(factor * radius)
    assert electric[0, 2] == pytest.approx(factor * 2j / k * (radius**2 / w0**2 - 1))
    assert electric[1, 0] == pytest.approx(0.0)
    assert electric[1, 1] == pytest.approx(0.0)
    assert electric[1, 2] == pytest.approx(-4j * amplitude / (k**2 * w0**2))
    np.testing.assert_allclose(magnetic[:, 1], -electric[:, 0] / C)
    np.testing.assert_allclose(magnetic[:, [0, 2]], 0.0, atol=1e-30)


def test_radial_hnap_focus_is_longitudinal_and_converged():
    wavelength = 0.8e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=8e-6, D=8e-6, mirror_type="HNAP")
    incident = RadiallyPolarisedSuperGaussian3D(
        w0=20e-6,
        wavelength=wavelength,
        spatial_order=16,
    )
    results = []
    for n_radial, n_azimuthal in ((12, 24), (20, 40)):
        surface = mirror.surface_quadrature(n_radial, n_azimuthal)
        electric_incident, magnetic_incident = incident.fields(surface.points)
        electric, magnetic = evaluate_SC_3D(
            np.zeros((1, 3)), surface, electric_incident, magnetic_incident, k
        )
        results.append((electric[0], magnetic[0]))

    np.testing.assert_allclose(results[0][0], results[1][0], rtol=2e-12, atol=3e-14)
    assert abs(results[1][0][0]) < 2e-14 * abs(results[1][0][2])
    assert abs(results[1][0][1]) < 2e-14 * abs(results[1][0][2])
    assert np.linalg.norm(results[1][1]) < 1e-13 * abs(results[1][0][2]) / C


def test_maxwell_residuals_for_sampled_plane_wave():
    wavelength = 1.0
    k = 2 * np.pi / wavelength
    axis = np.linspace(-0.25, 0.25, 31)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    phase = np.exp(1j * k * z)
    electric = np.zeros((*phase.shape, 3), dtype=complex)
    magnetic = np.zeros_like(electric)
    electric[..., 0] = phase
    magnetic[..., 1] = phase / C

    residuals = maxwell_residuals(electric, magnetic, (axis, axis, axis), k)
    assert residuals.divergence_e < 1e-13
    assert residuals.divergence_b < 1e-13
    assert residuals.faraday < 0.003
    assert residuals.ampere < 0.003


def test_stratton_chu_field_satisfies_maxwell_equations_near_focus():
    wavelength = 0.8e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=8e-6, D=8e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(16, 32)
    incident = LinearPolarisedSuperGaussian3D(
        w0=20e-6,
        wavelength=wavelength,
        spatial_order=16,
    )
    electric_incident, magnetic_incident = incident.fields(surface.points)
    axis = np.linspace(-0.2e-6, 0.2e-6, 11)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    observations = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    electric, magnetic = evaluate_SC_3D(
        observations,
        surface,
        electric_incident,
        magnetic_incident,
        k,
        chunk_size=32,
    )
    residuals = maxwell_residuals(
        electric.reshape(11, 11, 11, 3),
        magnetic.reshape(11, 11, 11, 3),
        (axis, axis, axis),
        k,
        trim=2,
    )

    assert residuals.divergence_e < 0.03
    assert residuals.divergence_b < 0.03
    assert residuals.faraday < 0.03
    assert residuals.ampere < 0.03


def test_general_integrand_matches_corrected_paraboloid_component_formula():
    """Cross-check against the component expansion used by Nielsen (2022)."""
    focal_length = 1.465
    wavelength = 1057e-9
    k = 2 * np.pi / wavelength
    x_source, y_source = 0.21, -0.17
    z_source = (x_source**2 + y_source**2) / (4 * focal_length) - focal_length
    source = np.array([x_source, y_source, z_source])
    observation = np.array([1.2e-6, -0.8e-6, 31e-6])
    source_to_observation = source - observation
    distance = np.linalg.norm(source_to_observation)

    # Projected-surface form of the general vector integrand, with the common
    # exp(ik(R-z_source)) factor removed and E_inc = x_hat.
    unnormalised_normal = np.array(
        [-x_source / (2 * focal_length), -y_source / (2 * focal_length), 1.0]
    )
    electric_incident = np.array([1.0, 0.0, 0.0])
    c_magnetic_incident = np.array([0.0, -1.0, 0.0])
    normal_cross_cb = np.cross(unnormalised_normal, c_magnetic_incident)
    gradient = source_to_observation * (1j * k * distance - 1) / distance**3
    electric_general = (
        1j * k * normal_cross_cb / distance
        + np.dot(unnormalised_normal, electric_incident) * gradient
    ) / (2 * np.pi)
    cb_general = np.cross(normal_cross_cb, gradient) / (2 * np.pi)

    dx, dy, dz = source_to_observation
    electric_components = np.array(
        [
            x_source * dx / (4 * np.pi * focal_length * distance**3)
            + 1j
            * k
            * (1 - x_source * dx / (2 * focal_length * distance))
            / (2 * np.pi * distance),
            x_source * dy / (4 * np.pi * focal_length * distance**3)
            - 1j * k * x_source * dy / (4 * np.pi * focal_length * distance**2),
            x_source * dz / (4 * np.pi * focal_length * distance**3)
            + 1j
            * k
            * x_source
            * (1 - dz / distance)
            / (4 * np.pi * focal_length * distance),
        ]
    )
    cb_components = np.array(
        [
            x_source * dy / (4 * np.pi * focal_length * distance**3)
            - 1j * k * x_source * dy / (4 * np.pi * focal_length * distance**2),
            -(x_source * dx / (2 * focal_length) - dz) / (2 * np.pi * distance**3)
            + 1j
            * k
            * (x_source * dx / (2 * focal_length) - dz)
            / (2 * np.pi * distance**2),
            -dy / (2 * np.pi * distance**3) + 1j * k * dy / (2 * np.pi * distance**2),
        ]
    )

    np.testing.assert_allclose(electric_general, electric_components, rtol=2e-15)
    np.testing.assert_allclose(cb_general, cb_components, rtol=2e-15)


def test_numpy_backend_is_reference_and_cupy_failure_is_explicit():
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=5e-6, D=4e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(6, 12)
    incident = LinearPolarisedSuperGaussian3D(w0=10e-6, wavelength=wavelength)
    electric_incident, magnetic_incident = incident.fields(surface.points)
    observation = np.array([[0.0, 0.0, 0.0], [0.1e-6, 0.2e-6, -0.1e-6]])

    default = evaluate_SC_3D(
        observation, surface, electric_incident, magnetic_incident, k
    )
    explicit = evaluate_SC_3D(
        observation,
        surface,
        electric_incident,
        magnetic_incident,
        k,
        backend="numpy",
    )
    np.testing.assert_array_equal(default[0], explicit[0])
    np.testing.assert_array_equal(default[1], explicit[1])
    with pytest.raises(ValueError, match="backend"):
        evaluate_SC_3D(
            observation,
            surface,
            electric_incident,
            magnetic_incident,
            k,
            backend="invalid",
        )
    if importlib.util.find_spec("cupy") is None:
        with pytest.raises(ImportError, match="CuPy"):
            evaluate_SC_3D(
                observation,
                surface,
                electric_incident,
                magnetic_incident,
                k,
                backend="cupy",
            )


def test_cupy_backend_algebra_with_numpy_array_module_stub(monkeypatch):
    """Exercise the optional backend without claiming CUDA runtime coverage."""
    cupy_stub = types.ModuleType("cupy")
    cupy_stub.asarray = np.asarray
    cupy_stub.asnumpy = np.asarray
    cupy_stub.cross = np.cross
    cupy_stub.sum = np.sum
    cupy_stub.any = np.any
    cupy_stub.exp = np.exp
    cupy_stub.linalg = np.linalg
    monkeypatch.setitem(sys.modules, "cupy", cupy_stub)

    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    mirror = ParabolicMirror3D(f0=5e-6, D=4e-6, mirror_type="HNAP")
    surface = mirror.surface_quadrature(6, 12)
    contour = mirror.contour_quadrature(16)
    incident = LinearPolarisedSuperGaussian3D(w0=10e-6, wavelength=wavelength)
    electric_incident, magnetic_incident = incident.fields(surface.points)
    _, magnetic_contour = incident.fields(contour.points)
    observation = np.array([[0.0, 0.0, 0.0], [0.1e-6, 0.2e-6, -0.1e-6]])

    reference = evaluate_SC_3D(
        observation,
        surface,
        electric_incident,
        magnetic_incident,
        k,
        contours=contour,
        B_inc_contours=(magnetic_contour,),
    )
    backend = evaluate_SC_3D(
        observation,
        surface,
        electric_incident,
        magnetic_incident,
        k,
        contours=contour,
        B_inc_contours=(magnetic_contour,),
        backend="cupy",
    )
    np.testing.assert_array_equal(reference[0], backend[0])
    np.testing.assert_array_equal(reference[1], backend[1])
