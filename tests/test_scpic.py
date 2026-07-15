# Change these lines at the top of tests/test_scpic.py:
import os
import numpy as np
import pytest
from scpic.mirrors import ParabolicMirror2D
from scpic.fields import IncidentFieldTM
from scpic.export import export_field_binary
def test_mirror_normal_vectors():
    """
    Verifies that the normal vectors generated for the 2D mirror
    are normalized to unit length: nx^2 + nz^2 = 1.
    """
    f0 = 10e-6  # 10 microns focal length
    D = 15e-6   # 15 microns diameter
    mirror = ParabolicMirror2D(f0=f0, D=D, mirror_type='OAP90')
    
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=100)
    
    # Calculate magnitude of normal vectors
    norm_magnitude = np.sqrt(nx**2 + nz**2)
    
    # Assert that all norms are extremely close to 1.0
    np.testing.assert_allclose(norm_magnitude, 1.0, rtol=1e-7)


def test_incident_field_amplitude():
    """
    Checks that the Gaussian transverse envelope behaves correctly:
    Maximum amplitude should be exactly at the center of the beam.
    """
    w0 = 5e-6
    wavelength = 1e-6
    E0 = 1.0
    field = IncidentFieldTM(w0=w0, wavelength=wavelength, E0=E0)
    
    # At the beam center x = x_center, z = 0, phase is 0, amplitude is B0
    x_center = 10e-6
    By_center = field.B_y(x_center, 0.0, x_center=x_center)
    expected_B0 = E0 / 299792458.0
    
    assert np.isclose(np.abs(By_center), expected_B0, rtol=1e-7)
    
    # Off-center should decay
    By_edge = field.B_y(x_center + w0, 0.0, x_center=x_center)
    assert np.abs(By_edge) < np.abs(By_center)


def test_binary_export_structure(tmp_path):
    """
    Tests that export_field_binary creates files of the correct size.
    Uses tmp_path (a built-in pytest fixture for temporary directory management).
    """
    # Create dummy complex spatial array (100 grid points)
    num_points = 100
    dummy_field = np.random.random(num_points) + 1j * np.random.random(num_points)
    
    filepath_base = os.path.join(tmp_path, "test_field")
    
    # Export as double precision (float64 = 8 bytes)
    export_field_binary(filepath_base, dummy_field, dtype=np.float64)
    
    amp_file = f"{filepath_base}_amp.bin"
    phase_file = f"{filepath_base}_phase.bin"
    
    assert os.path.exists(amp_file)
    assert os.path.exists(phase_file)
    
    # Expected file size = 100 points * 8 bytes/point = 800 bytes
    assert os.path.getsize(amp_file) == num_points * 8
    assert os.path.getsize(phase_file) == num_points * 8
    
    # Read files back and verify contents
    read_amp = np.fromfile(amp_file, dtype=np.float64)
    read_phase = np.fromfile(phase_file, dtype=np.float64)
    
    np.testing.assert_allclose(read_amp, np.abs(dummy_field))
    np.testing.assert_allclose(read_phase, np.angle(dummy_field))

    def test_paraxial_solver_accuracy():
    """
    Verifies that the Stratton-Chu solver matches the analytical 2D paraxial 
    Gaussian beam waist formula to within 2% error in the weak-focusing limit.
    """
    from scipy.optimize import curve_fit
    from scpic.solvers import evaluate_SC_2D
    
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    f0 = 50e-6
    D = 20e-6
    w_inc = 15e-6
    
    mirror = ParabolicMirror2D(f0=f0, D=D, mirror_type='OAP90')
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=1000)
    
    field = IncidentFieldTM(w0=w_inc, wavelength=wavelength, E0=1.0)
    By_inc = field.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = field.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)
    
    # 1D slice along transverse axis (z) at focal plane (x=0)
    z_vec = np.linspace(-4e-6, 4e-6, 100)
    x_vec = np.zeros_like(z_vec)
    
    By = evaluate_SC_2D(x_vec, z_vec, x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k)
    
    def gauss(z, amp, w):
        return amp * np.exp(-z**2 / w**2)
    
    popt, _ = curve_fit(gauss, z_vec, np.abs(By), p0=[np.max(np.abs(By)), 2e-6])
    w_num = abs(popt[1])
    w_theory = (2 * wavelength * f0) / (np.pi * w_inc)
    
    # Assert agreement within 2%
    assert abs(w_num - w_theory) / w_theory < 0.02