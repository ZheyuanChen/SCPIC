import numpy as np
from scipy.special import gamma

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


class LinearPolarisedSuperGaussian3D:
    """Vector super-Gaussian plane wave incident on a 3D reflector.

    The complex phasor convention is ``Re(E exp(-i omega t))``.  The
    default field travels along ``-z``, is polarised along ``+x``, and has
    ``B = propagation_direction x E / c``.
    """

    def __init__(
        self,
        w0,
        wavelength=None,
        spatial_order=16,
        E0=1.0,
        direction=(0.0, 0.0, -1.0),
        polarisation=(1.0, 0.0, 0.0),
        centre=(0.0, 0.0, 0.0),
    ):
        if w0 <= 0:
            raise ValueError("w0 must be positive")
        if wavelength is not None and wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if spatial_order <= 0:
            raise ValueError("spatial_order must be positive")

        direction = np.asarray(direction, dtype=float)
        polarisation = np.asarray(polarisation, dtype=float)
        centre = np.asarray(centre, dtype=float)
        if (
            direction.shape != (3,)
            or polarisation.shape != (3,)
            or centre.shape != (3,)
        ):
            raise ValueError("direction, polarisation, and centre must be 3-vectors")
        if np.linalg.norm(direction) == 0 or np.linalg.norm(polarisation) == 0:
            raise ValueError("direction and polarisation must be non-zero")
        direction = direction / np.linalg.norm(direction)
        polarisation = polarisation / np.linalg.norm(polarisation)
        if not np.isclose(np.dot(direction, polarisation), 0.0, atol=1e-12):
            raise ValueError("polarisation must be perpendicular to direction")

        self.w0 = float(w0)
        self.wavelength = wavelength
        self.k = None if wavelength is None else 2 * np.pi / wavelength
        self.spatial_order = float(spatial_order)
        self.E0 = complex(E0)
        self.direction = direction
        self.polarisation = polarisation
        self.centre = centre

    @classmethod
    def from_intensity_fwhm(cls, intensity_fwhm, **kwargs):
        """Construct from the full-width intensity FWHM used in the paper."""
        spatial_order = kwargs.get("spatial_order", 16)
        if intensity_fwhm <= 0:
            raise ValueError("intensity_fwhm must be positive")
        w0 = intensity_fwhm / (2 * (np.log(2) / 2) ** (1 / spatial_order))
        return cls(w0=w0, **kwargs)

    @property
    def effective_area(self):
        """Integral of the squared transverse envelope over an infinite plane."""
        p = self.spatial_order
        return 2 * np.pi * self.w0**2 * 2 ** (-2 / p) * gamma(2 / p) / p

    def fields(self, points, *, k=None, amplitude=None, spectral_phase=0.0):
        """Evaluate incident electric and magnetic phasors at ``points``."""
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        if k is None:
            k = self.k
        if k is None or k <= 0:
            raise ValueError("a positive wavenumber is required")
        if amplitude is None:
            amplitude = self.E0

        relative = points - self.centre
        longitudinal = relative @ self.direction
        transverse = relative - longitudinal[:, None] * self.direction
        radius = np.linalg.norm(transverse, axis=1)
        envelope = np.exp(-((radius / self.w0) ** self.spatial_order))
        phase = np.exp(1j * (k * longitudinal + spectral_phase))
        scalar = amplitude * envelope * phase
        electric = scalar[:, None] * self.polarisation
        magnetic = np.cross(self.direction, electric) / C
        return electric, magnetic


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
