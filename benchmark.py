"""Compare the 2D OAP90 solver with paraxial and tightly focused beams."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scpic.mirrors import ParabolicMirror2D
from scpic.fields import IncidentFieldTM
from scpic.solvers import evaluate_SC_2D


def gaussian_1d(z, amp, z0, w):
    """Gaussian field-amplitude profile with 1/e radius ``w``."""
    return amp * np.exp(-((z - z0) ** 2) / w**2)


def run_benchmark(f0, D, w_inc, label):
    """Return focal field amplitudes for one 2D OAP90 configuration."""
    wavelength = 1e-6
    k = 2 * np.pi / wavelength
    c = 299792458.0

    mirror = ParabolicMirror2D(f0=f0, D=D, mirror_type="OAP90")
    x_m, z_m, nx, nz, dl, x_center = mirror.get_surface(num_points=1500)

    field = IncidentFieldTM(w0=w_inc, wavelength=wavelength, E0=1.0)
    By_inc = field.B_y(x_m, z_m, x_center=x_center)
    dBy_dn_inc = field.dBy_dn(x_m, z_m, nx, nz, x_center=x_center)

    # The reflected chief ray propagates along -x, so z spans the focal-plane
    # transverse profile.
    Nz = 300
    z_vec = np.linspace(-6e-6, 6e-6, Nz)
    x_vec = np.zeros_like(z_vec)

    By = evaluate_SC_2D(x_vec, z_vec, x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k)

    # E_z requires dB_y/dx, which cannot be recovered from the focal line
    # alone. Evaluate two additional planes for a centred difference.
    dx = 1e-8
    By_plus = evaluate_SC_2D(
        x_vec + dx, z_vec, x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k
    )
    By_minus = evaluate_SC_2D(
        x_vec - dx, z_vec, x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k
    )

    dBy_dx = (By_plus - By_minus) / (2 * dx)
    dBy_dz = np.gradient(By, z_vec)

    Ex = -(1j * c / k) * dBy_dz
    Ez = (1j * c / k) * dBy_dx

    return z_vec, np.abs(By), np.abs(Ex), np.abs(Ez)


def main():
    """Run both comparisons and write the normalised field plot."""
    print("--- 1. Running Paraxial Benchmark (High f-number) ---")
    # The aperture is six beam radii wide so truncation does not dominate the
    # analytical Gaussian-waist comparison.
    f0_parax = 200e-6
    D_parax = 120e-6
    w_inc_parax = 20e-6

    z_p, By_p, Ex_p, Ez_p = run_benchmark(f0_parax, D_parax, w_inc_parax, "Paraxial")

    # Fit the dominant transverse electric field, rather than B_y, so this
    # measures the quantity eventually exported to EPOCH.
    popt, _ = curve_fit(gaussian_1d, z_p, Ez_p, p0=[np.max(Ez_p), 0.0, 6e-6])
    w_numerical = abs(popt[2])
    w_theoretical = (2 * 1e-6 * f0_parax) / (np.pi * w_inc_parax)

    print(f"Theoretical Paraxial Waist: {w_theoretical*1e6:.4f} μm")
    print(f"Numerical (Fitted) Waist:   {w_numerical*1e6:.4f} μm")
    print(
        f"Waist Agreement Error:      {abs(w_numerical - w_theoretical)/w_theoretical * 100:.2f}%"
    )

    print("\n--- 2. Running Tight-Focusing Benchmark (High NA) ---")
    # Here f_eff=2*f0=20 um and the wide pupil gives appreciable vector
    # corrections, including longitudinal E_x.
    f0_tight = 10e-6
    D_tight = 20e-6
    w_inc_tight = 8e-6

    z_t, By_t, Ex_t, Ez_t = run_benchmark(f0_tight, D_tight, w_inc_tight, "Tight Focus")

    peak_transverse = np.max(Ez_t)
    peak_longitudinal = np.max(Ex_t)
    ratio = (peak_longitudinal / peak_transverse) * 100

    print(f"Peak Transverse Field (Ez): {peak_transverse:.2e} V/m")
    print(f"Peak Longitudinal Field (Ex): {peak_longitudinal:.2e} V/m")
    print(f"Longitudinal-to-Transverse Ratio: {ratio:.2f}%")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(z_p * 1e6, Ez_p / np.max(Ez_p), "b-", label="Transverse (Ez)")
    plt.plot(z_p * 1e6, Ex_p / np.max(Ez_p), "r--", label="Longitudinal (Ex)")
    plt.title("Paraxial Regime (Weak Focusing)")
    plt.xlabel("Transverse Coordinate z (μm)")
    plt.ylabel("Normalised amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(z_t * 1e6, Ez_t / np.max(Ez_t), "b-", label="Transverse (Ez)")
    plt.plot(z_t * 1e6, Ex_t / np.max(Ez_t), "r--", label="Longitudinal (Ex)")
    plt.title("Tight-Focusing Regime (High NA)")
    plt.xlabel("Transverse Coordinate z (μm)")
    plt.ylabel("Normalised amplitude")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    print("\nSaved benchmark visualisation to 'benchmark_results.png'.")


if __name__ == "__main__":
    main()
