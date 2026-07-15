import numpy as np

C = 299_792_458.0


class IncidentFieldTM:
    """Monochromatic Gaussian beam incident along ``-z``.

    Complex fields use the phasor convention
    ``physical_field = Re[phasor * exp(-1j * omega * t)]`` throughout
    SCPIC.
    """

    def __init__(self, w0, wavelength, E0=1.0):
        if w0 <= 0 or wavelength <= 0:
            raise ValueError("w0 and wavelength must be positive")
        self.w0 = w0
        self.k = 2 * np.pi / wavelength
        self.c = C
        self.B0 = E0 / C

    def B_y(self, x, z, x_center=0.0):
        """Return the incident magnetic-field phasor travelling along ``-z``."""
        envelope = np.exp(-(((x - x_center) / self.w0) ** 2))
        phase = np.exp(-1j * self.k * z)
        return self.B0 * envelope * phase

    def dBy_dn(self, x, z, nx, nz, x_center=0.0):
        """Directional derivative of By along the mirror normal"""
        envelope = np.exp(-(((x - x_center) / self.w0) ** 2))
        phase = np.exp(-1j * self.k * z)

        dB_dx = self.B0 * (-2 * (x - x_center) / self.w0**2) * envelope * phase
        dB_dz = self.B0 * (-1j * self.k) * envelope * phase

        return dB_dx * nx + dB_dz * nz


def electric_from_magnetic_tm(By, x, z, k, *, edge_order=2):
    """Recover ``Ex`` and ``Ez`` from a regularly sampled ``By`` phasor.

    ``By`` must have shape ``(len(x), len(z))``.  For the package's
    ``exp(-i omega t)`` convention, Ampere's law gives
    ``Ex = -i c/k dBy/dz`` and ``Ez = i c/k dBy/dx``.  The caller supplies
    the wavenumber ``k`` explicitly.
    """
    By = np.asarray(By)
    x = np.asarray(x)
    z = np.asarray(z)
    if k <= 0:
        raise ValueError("k must be positive")
    if By.shape != (x.size, z.size):
        raise ValueError("By must have shape (len(x), len(z))")
    if x.size < edge_order + 1 or z.size < edge_order + 1:
        raise ValueError("x and z do not contain enough points for the gradient")
    dBy_dx, dBy_dz = np.gradient(By, x, z, edge_order=edge_order)
    Ex = -(1j * C / k) * dBy_dz
    Ez = (1j * C / k) * dBy_dx
    return Ex, Ez
