from math import factorial

import numpy as np
from scipy.special import eval_laguerre, gamma

C = 299_792_458.0


def _validate_zernike_mode(mode):
    if not isinstance(mode, tuple) or len(mode) != 2:
        raise ValueError("Zernike modes must be (n, m) tuples")
    n, m = mode
    if (
        not isinstance(n, (int, np.integer))
        or not isinstance(m, (int, np.integer))
        or n < 0
        or abs(m) > n
        or (n - abs(m)) % 2
    ):
        raise ValueError(f"invalid Zernike mode {(n, m)}")
    return int(n), int(m)


def _zernike_radial_polynomial(n, m, radius):
    radial = np.zeros_like(radius)
    for index in range((n - m) // 2 + 1):
        coefficient = (
            (-1) ** index
            * factorial(n - index)
            / (
                factorial(index)
                * factorial((n + m) // 2 - index)
                * factorial((n - m) // 2 - index)
            )
        )
        radial += coefficient * radius ** (n - 2 * index)
    return radial


def _zernike_mode(n, m, radius, theta):
    absolute_m = abs(m)
    radial = _zernike_radial_polynomial(n, absolute_m, radius)
    normalisation = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
    if m > 0:
        angular = np.cos(absolute_m * theta)
    elif m < 0:
        angular = np.sin(absolute_m * theta)
    else:
        angular = 1.0
    return normalisation * radial * angular


def _pupil_coordinates(points, centre, axis_u, axis_v, pupil_radius, outside):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    relative = points - centre
    u = relative @ axis_u / pupil_radius
    v = relative @ axis_v / pupil_radius
    radius = np.hypot(u, v)
    outside_mask = radius > 1.0 + 1e-12
    if outside == "raise" and np.any(outside_mask):
        raise ValueError("wavefront points lie outside the Zernike pupil")
    return radius, np.arctan2(v, u), outside_mask


def _validated_pupil_geometry(
    pupil_radius,
    centre,
    axis_u,
    axis_v,
    outside,
):
    if pupil_radius <= 0:
        raise ValueError("pupil_radius must be positive")
    if outside not in {"raise", "zero", "extrapolate"}:
        raise ValueError("outside must be 'raise', 'zero', or 'extrapolate'")

    centre = np.asarray(centre, dtype=float)
    axis_u = np.asarray(axis_u, dtype=float)
    axis_v = np.asarray(axis_v, dtype=float)
    if centre.shape != (3,) or axis_u.shape != (3,) or axis_v.shape != (3,):
        raise ValueError("centre, axis_u, and axis_v must be 3-vectors")
    if np.linalg.norm(axis_u) == 0 or np.linalg.norm(axis_v) == 0:
        raise ValueError("pupil axes must be non-zero")
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    if not np.isclose(np.dot(axis_u, axis_v), 0.0, atol=1e-12):
        raise ValueError("pupil axes must be perpendicular")
    return float(pupil_radius), centre, axis_u, axis_v, outside


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


class ZernikeWavefront:
    """Optical-path-difference map expanded in OSA/ANSI Zernike modes.

    Coefficients are supplied as ``{(n, m): value_in_metres}``.  The modes
    are orthonormal on the unit disk, so each non-piston coefficient is its
    RMS optical-path-difference contribution.  Positive ``m`` uses cosine
    azimuthal dependence and negative ``m`` uses sine dependence.
    """

    def __init__(
        self,
        pupil_radius,
        coefficients,
        *,
        centre=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        outside="raise",
    ):
        pupil_radius, centre, axis_u, axis_v, outside = _validated_pupil_geometry(
            pupil_radius,
            centre,
            axis_u,
            axis_v,
            outside,
        )

        validated = {}
        for mode, coefficient in dict(coefficients).items():
            n, m = _validate_zernike_mode(mode)
            coefficient = float(coefficient)
            if not np.isfinite(coefficient):
                raise ValueError("Zernike coefficients must be finite")
            validated[(n, m)] = coefficient

        self.pupil_radius = pupil_radius
        self.coefficients = validated
        self.centre = centre
        self.axis_u = axis_u
        self.axis_v = axis_v
        self.outside = outside

    @staticmethod
    def _radial_polynomial(n, m, radius):
        return _zernike_radial_polynomial(n, m, radius)

    def opd(self, points):
        """Return optical path difference in metres at three-dimensional points."""
        radius, theta, outside = _pupil_coordinates(
            points,
            self.centre,
            self.axis_u,
            self.axis_v,
            self.pupil_radius,
            self.outside,
        )
        result = np.zeros(len(points), dtype=float)
        for (n, m), coefficient in self.coefficients.items():
            result += coefficient * _zernike_mode(n, m, radius, theta)
        if self.outside == "zero":
            result[outside] = 0.0
        return result

    __call__ = opd


class ChromaticZernikePhase:
    """Position--frequency phase expanded in OSA/ANSI Zernike modes.

    ``coefficients`` maps ``(n, m)`` modes to either a finite phase in radians
    or a callable ``coefficient(angular_frequency)`` returning radians.  This
    represents a genuinely non-separable spatio-spectral phase and is distinct
    from :class:`ZernikeWavefront`, whose coefficients are fixed optical path
    differences in metres.
    """

    def __init__(
        self,
        pupil_radius,
        coefficients,
        *,
        carrier_angular_frequency,
        centre=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        outside="raise",
    ):
        carrier = float(carrier_angular_frequency)
        if not np.isfinite(carrier) or carrier <= 0:
            raise ValueError("carrier_angular_frequency must be positive and finite")
        pupil_radius, centre, axis_u, axis_v, outside = _validated_pupil_geometry(
            pupil_radius,
            centre,
            axis_u,
            axis_v,
            outside,
        )
        validated = {}
        for mode, coefficient in dict(coefficients).items():
            mode = _validate_zernike_mode(mode)
            if callable(coefficient):
                validated[mode] = coefficient
            else:
                coefficient = float(coefficient)
                if not np.isfinite(coefficient):
                    raise ValueError("chromatic Zernike coefficients must be finite")
                validated[mode] = coefficient

        self.pupil_radius = pupil_radius
        self.coefficients = validated
        self.carrier_angular_frequency = carrier
        self.centre = centre
        self.axis_u = axis_u
        self.axis_v = axis_v
        self.outside = outside

    @classmethod
    def jolly_angular_dispersion(
        cls,
        pupil_radius,
        pulse_front_tilt,
        *,
        carrier_angular_frequency,
        azimuthal_index=1,
        **kwargs,
    ):
        """Construct Eq. (44) of Jolly et al. (2025).

        ``pulse_front_tilt`` is the group delay in seconds at one pupil radius.
        ``azimuthal_index=1`` uses the cosine x-tilt mode and ``-1`` uses the
        sine y-tilt mode.
        """
        if azimuthal_index not in {-1, 1}:
            raise ValueError("azimuthal_index must be -1 or 1")
        delay = float(pulse_front_tilt)
        if not np.isfinite(delay):
            raise ValueError("pulse_front_tilt must be finite")
        carrier = float(carrier_angular_frequency)

        def coefficient(omega):
            return delay * omega * (omega - carrier) / (2 * carrier)

        return cls(
            pupil_radius,
            {(1, azimuthal_index): coefficient},
            carrier_angular_frequency=carrier,
            **kwargs,
        )

    @classmethod
    def jolly_chromatic_curvature(
        cls,
        pupil_radius,
        pulse_front_curvature,
        *,
        carrier_angular_frequency,
        include_piston=True,
        **kwargs,
    ):
        """Construct the chromatic defocus phase in Eq. (45)."""
        delay = float(pulse_front_curvature)
        if not np.isfinite(delay):
            raise ValueError("pulse_front_curvature must be finite")
        if not isinstance(include_piston, (bool, np.bool_)):
            raise TypeError("include_piston must be boolean")
        carrier = float(carrier_angular_frequency)

        def common(omega):
            return delay * omega * (omega - carrier) / (2 * carrier)

        coefficients = {(2, 0): lambda omega: common(omega) / np.sqrt(3)}
        if include_piston:
            coefficients[(0, 0)] = common
        return cls(
            pupil_radius,
            coefficients,
            carrier_angular_frequency=carrier,
            **kwargs,
        )

    @classmethod
    def jolly_chromatic_trefoil(
        cls,
        pupil_radius,
        characteristic_delay,
        *,
        carrier_angular_frequency,
        azimuthal_index=3,
        **kwargs,
    ):
        """Construct the chromatic trefoil phase in Eq. (46)."""
        if azimuthal_index not in {-3, 3}:
            raise ValueError("azimuthal_index must be -3 or 3")
        delay = float(characteristic_delay)
        if not np.isfinite(delay):
            raise ValueError("characteristic_delay must be finite")
        carrier = float(carrier_angular_frequency)

        def coefficient(omega):
            return delay * omega * (omega - carrier) / (2 * carrier * np.sqrt(8))

        return cls(
            pupil_radius,
            {(3, azimuthal_index): coefficient},
            carrier_angular_frequency=carrier,
            **kwargs,
        )

    def phase(self, points, angular_frequency):
        """Return the spatio-spectral phase in radians."""
        omega = float(angular_frequency)
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("angular_frequency must be positive and finite")
        radius, theta, outside = _pupil_coordinates(
            points,
            self.centre,
            self.axis_u,
            self.axis_v,
            self.pupil_radius,
            self.outside,
        )
        result = np.zeros(len(points), dtype=float)
        for (n, m), coefficient in self.coefficients.items():
            value = coefficient(omega) if callable(coefficient) else coefficient
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(
                    f"chromatic Zernike coefficient {(n, m)} is not finite"
                )
            result += value * _zernike_mode(n, m, radius, theta)
        if self.outside == "zero":
            result[outside] = 0.0
        return result

    __call__ = phase


class _SuperGaussian3D:
    """Shared spatial and spectral handling for collimated vector beams."""

    def __init__(
        self,
        w0,
        wavelength=None,
        spatial_order=16,
        E0=1.0,
        direction=(0.0, 0.0, -1.0),
        centre=(0.0, 0.0, 0.0),
        wavefront_opd=None,
        spatio_spectral_phase=None,
    ):
        if w0 <= 0:
            raise ValueError("w0 must be positive")
        if wavelength is not None and wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if spatial_order <= 0:
            raise ValueError("spatial_order must be positive")

        direction = np.asarray(direction, dtype=float)
        centre = np.asarray(centre, dtype=float)
        if direction.shape != (3,) or centre.shape != (3,):
            raise ValueError("direction and centre must be 3-vectors")
        if np.linalg.norm(direction) == 0:
            raise ValueError("direction must be non-zero")
        if wavefront_opd is not None and not callable(wavefront_opd):
            raise TypeError("wavefront_opd must be callable")
        if spatio_spectral_phase is not None and not callable(spatio_spectral_phase):
            raise TypeError("spatio_spectral_phase must be callable")

        self.w0 = float(w0)
        self.wavelength = wavelength
        self.k = None if wavelength is None else 2 * np.pi / wavelength
        self.spatial_order = float(spatial_order)
        self.E0 = complex(E0)
        self.direction = direction / np.linalg.norm(direction)
        self.centre = centre
        self.wavefront_opd = wavefront_opd
        self.spatio_spectral_phase = spatio_spectral_phase

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

    def _polarisation_vectors(self, transverse, radius):
        raise NotImplementedError

    def _phase_correction(self, points, k):
        correction = np.zeros(len(points), dtype=float)
        if self.wavefront_opd is not None:
            opd = np.asarray(self.wavefront_opd(points), dtype=float)
            if opd.shape != (len(points),) or not np.all(np.isfinite(opd)):
                raise ValueError("wavefront_opd must return one finite value per point")
            correction += k * opd
        if self.spatio_spectral_phase is not None:
            phase = np.asarray(
                self.spatio_spectral_phase(points, k * C),
                dtype=float,
            )
            if phase.shape != (len(points),) or not np.all(np.isfinite(phase)):
                raise ValueError(
                    "spatio_spectral_phase must return one finite value per point"
                )
            correction += phase
        return correction

    def fields(self, points, *, k=None, amplitude=None, spectral_phase=0.0):
        """Evaluate incident electric and magnetic phasors at ``points``.

        ``wavefront_opd(points)`` is interpreted in metres and contributes
        ``+k * OPD`` to the phasor phase.  This keeps a measured wavefront
        frequency-aware during broadband propagation.
        ``spatio_spectral_phase(points, k*c)`` instead contributes phase
        directly in radians and may vary arbitrarily with position and
        frequency.
        """
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
        phase_argument = (
            k * longitudinal + spectral_phase + self._phase_correction(points, k)
        )
        scalar = amplitude * envelope * np.exp(1j * phase_argument)
        electric = scalar[:, None] * self._polarisation_vectors(transverse, radius)
        magnetic = np.cross(self.direction, electric) / C
        return electric, magnetic


class LinearPolarisedSuperGaussian3D(_SuperGaussian3D):
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
        wavefront_opd=None,
        spatio_spectral_phase=None,
    ):
        polarisation = np.asarray(polarisation, dtype=float)
        if polarisation.shape != (3,):
            raise ValueError("polarisation must be a 3-vector")
        if np.linalg.norm(polarisation) == 0:
            raise ValueError("polarisation must be non-zero")
        polarisation = polarisation / np.linalg.norm(polarisation)
        super().__init__(
            w0=w0,
            wavelength=wavelength,
            spatial_order=spatial_order,
            E0=E0,
            direction=direction,
            centre=centre,
            wavefront_opd=wavefront_opd,
            spatio_spectral_phase=spatio_spectral_phase,
        )
        if not np.isclose(np.dot(self.direction, polarisation), 0.0, atol=1e-12):
            raise ValueError("polarisation must be perpendicular to direction")
        self.polarisation = polarisation

    def _polarisation_vectors(self, transverse, radius):
        return np.broadcast_to(self.polarisation, transverse.shape)


class RadiallyPolarisedSuperGaussian3D(_SuperGaussian3D):
    """Collimated radially polarised super-Gaussian beam.

    The electric direction is the local transverse radial unit vector.  It is
    set to zero at the single undefined point on the beam axis; this has zero
    measure in surface and energy integrals and represents the regular centre
    of a physical radially polarised mode.
    """

    def _polarisation_vectors(self, transverse, radius):
        vectors = np.zeros_like(transverse)
        nonzero = radius > 0
        vectors[nonzero] = transverse[nonzero] / radius[nonzero, None]
        return vectors


class TM01RadiallyPolarisedBeam3D(_SuperGaussian3D):
    """Paper-accurate collimated TM01 incident field.

    This implements Eqs. (16)--(17) of Vallières et al. (2023), generalised
    from propagation along ``-z`` to an arbitrary ``direction``.  Unlike
    :class:`RadiallyPolarisedSuperGaussian3D`, the transverse electric field
    vanishes continuously on axis and the small longitudinal component makes
    the incident field divergence-free.

    ``E0`` is the spectral coefficient denoted ``E0,n`` in the paper, rather
    than the peak radial electric field.
    """

    def __init__(
        self,
        w0,
        wavelength=None,
        E0=1.0,
        direction=(0.0, 0.0, -1.0),
        centre=(0.0, 0.0, 0.0),
        wavefront_opd=None,
        spatio_spectral_phase=None,
    ):
        super().__init__(
            w0=w0,
            wavelength=wavelength,
            spatial_order=2,
            E0=E0,
            direction=direction,
            centre=centre,
            wavefront_opd=wavefront_opd,
            spatio_spectral_phase=spatio_spectral_phase,
        )

    def effective_area(self, k=None):
        """Return the longitudinal-flux area relative to ``E0``.

        Integrating the Poynting flux of Eqs. (16)--(17) over an infinite
        transverse plane gives ``pi/k**2``.  It is frequency-dependent for a
        broadband pulse.
        """
        if k is None:
            k = self.k
        if k is None:
            raise ValueError("a positive wavenumber is required")
        k = np.asarray(k, dtype=float)
        if np.any(~np.isfinite(k)) or np.any(k <= 0):
            raise ValueError("a positive wavenumber is required")
        return np.pi / k**2

    def fields(self, points, *, k=None, amplitude=None, spectral_phase=0.0):
        """Evaluate the TM01 electric and magnetic phasors at ``points``."""
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
        radius_squared = np.sum(transverse**2, axis=1)
        phase_argument = (
            k * longitudinal + spectral_phase + self._phase_correction(points, k)
        )

        scalar = (
            2
            * amplitude
            / (k * self.w0**2)
            * np.exp(-radius_squared / self.w0**2 + 1j * phase_argument)
        )
        longitudinal_term = (-2j / k * (radius_squared / self.w0**2 - 1))[
            :, None
        ] * self.direction
        electric = scalar[:, None] * (transverse + longitudinal_term)
        magnetic = np.cross(self.direction, electric) / C
        return electric, magnetic


class FiniteRayleighTM01Beam3D(TM01RadiallyPolarisedBeam3D):
    """Finite-Rayleigh-range radial Gaussian from Dumont et al. (2017).

    The beam waist is centred at ``centre`` and the signed propagation
    coordinate increases along ``direction``.  This retains the complex
    ``q`` dependence, wavefront curvature and longitudinal electric field of
    Eqs. (7a)--(7c) in Dumont's supplementary material.  At the waist it is
    exactly :class:`TM01RadiallyPolarisedBeam3D` with the same amplitude.
    """

    def fields(self, points, *, k=None, amplitude=None, spectral_phase=0.0):
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
        radius_squared = np.sum(transverse**2, axis=1)
        rayleigh_range = k * self.w0**2 / 2
        q = 1 / (1 - 1j * longitudinal / rayleigh_range)
        phase_argument = (
            k * longitudinal + spectral_phase + self._phase_correction(points, k)
        )

        common = amplitude * np.exp(
            -q * radius_squared / self.w0**2 + 1j * phase_argument
        )
        transverse_coefficient = 2 * q**2 / (k * self.w0**2)
        longitudinal_coefficient = (
            4j / (k**2 * self.w0**2) * (q**2 - q**3 * radius_squared / self.w0**2)
        )
        electric = common[:, None] * (
            transverse_coefficient[:, None] * transverse
            + longitudinal_coefficient[:, None] * self.direction
        )
        magnetic = np.cross(self.direction, electric) / C
        return electric, magnetic


class ParaxialGaussLaguerreBeam3D:
    """Axisymmetric finite-distance Gauss--Laguerre incident beam.

    This implements Eq. (9) of the Dumont et al. supplementary material as
    a linearly polarised paraxial field. ``mode_coefficients`` maps the
    non-negative radial mode number ``n`` to a complex coefficient ``c_n``.
    It is intended for a well-collimated upstream field; longitudinal
    corrections of order the divergence angle are not added.
    """

    def __init__(
        self,
        w0,
        mode_coefficients=(1.0,),
        *,
        wavelength=None,
        E0=1.0,
        direction=(0.0, 0.0, -1.0),
        polarisation=(1.0, 0.0, 0.0),
        centre=(0.0, 0.0, 0.0),
        wavefront_opd=None,
        spatio_spectral_phase=None,
    ):
        if w0 <= 0:
            raise ValueError("w0 must be positive")
        if wavelength is not None and wavelength <= 0:
            raise ValueError("wavelength must be positive")
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
        if wavefront_opd is not None and not callable(wavefront_opd):
            raise TypeError("wavefront_opd must be callable")
        if spatio_spectral_phase is not None and not callable(spatio_spectral_phase):
            raise TypeError("spatio_spectral_phase must be callable")

        if isinstance(mode_coefficients, dict):
            items = mode_coefficients.items()
        else:
            items = enumerate(mode_coefficients)
        modes = {}
        for mode, coefficient in items:
            if not isinstance(mode, (int, np.integer)) or mode < 0:
                raise ValueError(
                    "Gauss--Laguerre mode numbers must be non-negative integers"
                )
            coefficient = complex(coefficient)
            if not np.isfinite(coefficient.real) or not np.isfinite(coefficient.imag):
                raise ValueError("mode coefficients must be finite")
            if coefficient != 0:
                modes[int(mode)] = coefficient
        if not modes:
            raise ValueError("at least one mode coefficient must be non-zero")

        self.w0 = float(w0)
        self.wavelength = wavelength
        self.k = None if wavelength is None else 2 * np.pi / wavelength
        self.E0 = complex(E0)
        self.direction = direction
        self.polarisation = polarisation
        self.centre = centre
        self.mode_coefficients = modes
        self.wavefront_opd = wavefront_opd
        self.spatio_spectral_phase = spatio_spectral_phase

    @property
    def effective_area(self):
        """Return the conserved transverse-flux area relative to ``E0``."""
        modal_norm = sum(abs(value) ** 2 for value in self.mode_coefficients.values())
        return np.pi * self.w0**2 / 2 * modal_norm

    def fields(self, points, *, k=None, amplitude=None, spectral_phase=0.0):
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
        radius_squared = np.sum(transverse**2, axis=1)
        rayleigh_range = k * self.w0**2 / 2
        width = self.w0 * np.sqrt(1 + (longitudinal / rayleigh_range) ** 2)
        argument = 2 * radius_squared / width**2
        modal_sum = np.zeros(len(points), dtype=complex)
        for mode, coefficient in self.mode_coefficients.items():
            modal_sum += coefficient * eval_laguerre(mode, argument)

        curvature_phase = (
            k
            * radius_squared
            * longitudinal
            / (2 * (longitudinal**2 + rayleigh_range**2))
        )
        gouy_phase = np.arctan(longitudinal / rayleigh_range)
        phase_argument = (
            k * longitudinal + curvature_phase - gouy_phase + spectral_phase
        )
        if self.wavefront_opd is not None:
            opd = np.asarray(self.wavefront_opd(points), dtype=float)
            if opd.shape != (len(points),) or not np.all(np.isfinite(opd)):
                raise ValueError("wavefront_opd must return one finite value per point")
            phase_argument = phase_argument + k * opd
        if self.spatio_spectral_phase is not None:
            phase = np.asarray(
                self.spatio_spectral_phase(points, k * C),
                dtype=float,
            )
            if phase.shape != (len(points),) or not np.all(np.isfinite(phase)):
                raise ValueError(
                    "spatio_spectral_phase must return one finite value per point"
                )
            phase_argument = phase_argument + phase
        scalar = (
            amplitude
            * self.w0
            / width
            * modal_sum
            * np.exp(-radius_squared / width**2 + 1j * phase_argument)
        )
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
