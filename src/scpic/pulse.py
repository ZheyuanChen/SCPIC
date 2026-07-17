"""Discrete broadband pulse construction following Vallières and Dumont."""

from dataclasses import dataclass, replace

import numpy as np

from .fields import C

EPSILON_0 = 8.854_187_8128e-12


def _phase_array(phase, angular_frequencies):
    if callable(phase):
        phase = phase(angular_frequencies)
    values = np.asarray(phase, dtype=float)
    if values.ndim == 0:
        values = np.full_like(angular_frequencies, float(values))
    if values.shape != angular_frequencies.shape or np.any(~np.isfinite(values)):
        raise ValueError("spectral_phase must be finite and match the spectrum")
    return values


def _validated_spectrum_arrays(angular_frequencies, relative_energy_density, phase):
    frequencies = np.asarray(angular_frequencies, dtype=float)
    density = np.asarray(relative_energy_density, dtype=float)
    if frequencies.ndim != 1 or len(frequencies) < 2:
        raise ValueError("angular_frequencies must contain at least two samples")
    if density.shape != frequencies.shape:
        raise ValueError("relative_energy_density must match angular_frequencies")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
        raise ValueError("angular_frequencies must be positive and finite")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("angular_frequencies must be strictly increasing")
    spacing = np.diff(frequencies)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=0.0):
        raise ValueError("the discrete Fourier spectrum must be uniformly sampled")
    if np.any(~np.isfinite(density)) or np.any(density < 0) or not np.any(density > 0):
        raise ValueError(
            "relative_energy_density must be finite, non-negative and non-zero"
        )
    phases = _phase_array(phase, frequencies)
    frequencies = frequencies.copy()
    density = density.copy()
    phases = phases.copy()
    for values in (frequencies, density, phases):
        values.setflags(write=False)
    return frequencies, density, phases, float(spacing[0])


class _DiscreteSpectrumMethods:
    """Shared energy normalisation for uniformly sampled positive frequencies."""

    @property
    def period(self):
        """Period of the discrete representation, ``2*pi/delta_omega``."""
        return 2 * np.pi / self.delta_omega

    @property
    def nyquist_timestep(self):
        """Largest time step resolving the highest optical frequency."""
        return np.pi / self.angular_frequencies[-1]

    def envelope_nyquist_timestep(self, carrier_angular_frequency):
        """Largest time step resolving the spectrum relative to a carrier."""
        carrier = float(carrier_angular_frequency)
        if not np.isfinite(carrier) or carrier <= 0:
            raise ValueError("carrier_angular_frequency must be positive and finite")
        maximum_detuning = np.max(np.abs(self.angular_frequencies - carrier))
        if maximum_detuning == 0:
            return np.inf
        return float(np.pi / maximum_detuning)

    def with_spectral_phase(self, spectral_phase):
        """Return a copy carrying ``phi(omega)`` in radians.

        ``spectral_phase`` may be a scalar, an array matching the frequency
        grid, or a callable evaluated on the angular-frequency array.
        """
        return replace(
            self,
            spectral_phase=_phase_array(spectral_phase, self.angular_frequencies),
        )

    def component_amplitudes(self, effective_area):
        """Return non-negative incident electric-field amplitudes.

        Each component has time-averaged Poynting flux
        ``2*eps0*c*|E_n|**2`` under the package's positive-frequency
        convention. ``effective_area`` may be scalar or frequency-dependent.
        """
        effective_area = np.asarray(effective_area, dtype=float)
        if effective_area.ndim > 1 or (
            effective_area.ndim == 1
            and effective_area.shape != self.angular_frequencies.shape
        ):
            raise ValueError("effective_area must be scalar or match the spectrum")
        if np.any(~np.isfinite(effective_area)) or np.any(effective_area <= 0):
            raise ValueError("effective_area must be positive and finite")
        common_normalisation = self.total_energy / (
            self.period * 2 * EPSILON_0 * C * np.sum(self.relative_energy_density)
        )
        return np.sqrt(
            common_normalisation * self.relative_energy_density / effective_area
        )

    def component_coefficients(self, effective_area):
        """Return complex component amplitudes including spectral phase."""
        return self.component_amplitudes(effective_area) * np.exp(
            1j * self.spectral_phase
        )

    def recovered_energy(self, amplitudes, effective_area):
        amplitudes = np.asarray(amplitudes)
        if amplitudes.shape != self.angular_frequencies.shape:
            raise ValueError("amplitudes must match the spectrum")
        effective_area = np.asarray(effective_area, dtype=float)
        if effective_area.ndim > 1 or (
            effective_area.ndim == 1
            and effective_area.shape != self.angular_frequencies.shape
        ):
            raise ValueError("effective_area must be scalar or match the spectrum")
        if np.any(~np.isfinite(effective_area)) or np.any(effective_area <= 0):
            raise ValueError("effective_area must be positive and finite")
        return float(
            self.period
            * 2
            * EPSILON_0
            * C
            * np.sum(effective_area * np.abs(amplitudes) ** 2)
        )

    def validate_time_samples(self, times):
        """Reject a time grid that aliases the carrier or spans one period."""
        self._validate_time_samples(times, self.nyquist_timestep)

    def validate_envelope_time_samples(self, times, carrier_angular_frequency):
        """Validate a time grid for a carrier-referenced complex envelope."""
        self._validate_time_samples(
            times,
            self.envelope_nyquist_timestep(carrier_angular_frequency),
        )

    def _validate_time_samples(self, times, maximum_timestep):
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or len(times) < 1 or np.any(~np.isfinite(times)):
            raise ValueError("times must be a finite one-dimensional array")
        if len(times) > 1:
            if np.any(np.diff(times) <= 0):
                raise ValueError("times must be strictly increasing")
            if np.max(np.diff(times)) > maximum_timestep * (1 + 1e-12):
                raise ValueError("time spacing does not resolve the represented field")
            if times[-1] - times[0] >= self.period:
                raise ValueError("time span must be shorter than the discrete period")


@dataclass(frozen=True)
class SampledSpectrum(_DiscreteSpectrumMethods):
    """Measured or tabulated spectrum resampled on a uniform omega grid."""

    angular_frequencies: np.ndarray
    relative_energy_density: np.ndarray
    delta_omega: float
    total_energy: float
    spectral_phase: object = 0.0

    def __post_init__(self):
        frequencies, density, phases, spacing = _validated_spectrum_arrays(
            self.angular_frequencies,
            self.relative_energy_density,
            self.spectral_phase,
        )
        if not np.isfinite(self.total_energy) or self.total_energy <= 0:
            raise ValueError("total_energy must be positive and finite")
        if not np.isclose(self.delta_omega, spacing, rtol=1e-10):
            raise ValueError("delta_omega does not match the frequency grid")
        object.__setattr__(self, "angular_frequencies", frequencies)
        object.__setattr__(self, "relative_energy_density", density)
        object.__setattr__(self, "spectral_phase", phases)
        object.__setattr__(self, "delta_omega", spacing)

    @classmethod
    def from_angular_frequency_samples(
        cls,
        angular_frequencies,
        energy_density,
        *,
        total_energy,
        spectral_phase=0.0,
        n_components=None,
    ):
        """Interpolate tabulated ``dE/domega`` onto a uniform omega grid."""
        frequencies = np.asarray(angular_frequencies, dtype=float)
        density = np.asarray(energy_density, dtype=float)
        if (
            frequencies.ndim != 1
            or density.shape != frequencies.shape
            or len(frequencies) < 2
        ):
            raise ValueError("frequency and density samples must be matching 1D arrays")
        if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0):
            raise ValueError("sample frequencies must be positive and finite")
        if (
            np.any(~np.isfinite(density))
            or np.any(density < 0)
            or not np.any(density > 0)
        ):
            raise ValueError(
                "sample energy density must be finite, non-negative and non-zero"
            )
        order = np.argsort(frequencies)
        frequencies = frequencies[order]
        density = density[order]
        if np.any(np.diff(frequencies) <= 0):
            raise ValueError("sample frequencies must be unique")
        phases = _phase_array(
            spectral_phase, np.asarray(angular_frequencies, dtype=float)
        )[order]
        if n_components is None:
            n_components = len(frequencies)
        if not isinstance(n_components, (int, np.integer)) or n_components < 2:
            raise ValueError("n_components must be at least two")
        uniform = np.linspace(frequencies[0], frequencies[-1], n_components)
        interpolated_density = np.interp(uniform, frequencies, density)
        interpolated_phase = np.interp(uniform, frequencies, np.unwrap(phases))
        interpolated_density /= np.max(interpolated_density)
        return cls(
            angular_frequencies=uniform,
            relative_energy_density=interpolated_density,
            delta_omega=float(uniform[1] - uniform[0]),
            total_energy=float(total_energy),
            spectral_phase=interpolated_phase,
        )

    @classmethod
    def from_wavelength_samples(
        cls,
        wavelengths,
        energy_density_per_wavelength,
        *,
        total_energy,
        spectral_phase=0.0,
        n_components=None,
    ):
        """Convert measured ``dE/dlambda`` to ``dE/domega`` exactly.

        The Jacobian ``|dlambda/domega| = 2*pi*c/omega**2`` is applied before
        interpolation onto the uniform angular-frequency grid required by the
        discrete Fourier reconstruction.
        """
        wavelengths = np.asarray(wavelengths, dtype=float)
        density_lambda = np.asarray(energy_density_per_wavelength, dtype=float)
        if wavelengths.ndim != 1 or density_lambda.shape != wavelengths.shape:
            raise ValueError(
                "wavelength and density samples must be matching 1D arrays"
            )
        if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
            raise ValueError("wavelength samples must be positive and finite")
        frequencies = 2 * np.pi * C / wavelengths
        density_omega = density_lambda * (2 * np.pi * C / frequencies**2)
        return cls.from_angular_frequency_samples(
            frequencies,
            density_omega,
            total_energy=total_energy,
            spectral_phase=spectral_phase,
            n_components=n_components,
        )


@dataclass(frozen=True)
class SuperGaussianSpectrum(_DiscreteSpectrumMethods):
    """Uniformly sampled, energy-normalised super-Gaussian spectrum."""

    angular_frequencies: np.ndarray
    relative_energy_density: np.ndarray
    delta_omega: float
    total_energy: float
    central_wavelength: float
    wavelength_fwhm: float
    spectral_order: float
    spectral_phase: object = 0.0
    conversion: str = "narrowband"

    def __post_init__(self):
        frequencies, density, phases, spacing = _validated_spectrum_arrays(
            self.angular_frequencies,
            self.relative_energy_density,
            self.spectral_phase,
        )
        if not np.isfinite(self.total_energy) or self.total_energy <= 0:
            raise ValueError("total_energy must be positive and finite")
        if (
            not np.isfinite(self.central_wavelength)
            or not np.isfinite(self.wavelength_fwhm)
            or not np.isfinite(self.spectral_order)
            or self.central_wavelength <= 0
            or self.wavelength_fwhm <= 0
            or self.spectral_order <= 0
        ):
            raise ValueError("spectrum wavelength metadata and order must be positive")
        if not np.isclose(self.delta_omega, spacing, rtol=1e-10):
            raise ValueError("delta_omega does not match the frequency grid")
        if self.conversion not in {"narrowband", "exact_wavelength_density"}:
            raise ValueError("unsupported wavelength-to-frequency conversion")
        object.__setattr__(self, "angular_frequencies", frequencies)
        object.__setattr__(self, "relative_energy_density", density)
        object.__setattr__(self, "spectral_phase", phases)
        object.__setattr__(self, "delta_omega", spacing)

    @classmethod
    def from_wavelength_bandwidth(
        cls,
        *,
        central_wavelength=800e-9,
        wavelength_fwhm=90e-9,
        spectral_order=7,
        total_energy=20.0,
        n_components=31,
        span_fwhm=2.0,
        conversion="narrowband",
        spectral_phase=0.0,
    ):
        """Construct a super-Gaussian specified in wavelength units.

        ``conversion='narrowband'`` preserves the Vallières benchmark:
        the wavelength FWHM is linearly converted around the carrier and the
        super-Gaussian is formed in angular frequency.  The
        ``'exact_wavelength_density'`` option instead defines ``dE/dlambda``
        in wavelength and applies the exact Jacobian to obtain ``dE/domega``.
        """
        if central_wavelength <= 0 or wavelength_fwhm <= 0:
            raise ValueError("wavelengths must be positive")
        if spectral_order <= 0 or total_energy <= 0:
            raise ValueError("spectral_order and total_energy must be positive")
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        if span_fwhm <= 0:
            raise ValueError("span_fwhm must be positive")

        omega0 = 2 * np.pi * C / central_wavelength
        if conversion == "narrowband":
            omega_fwhm = 2 * np.pi * C * wavelength_fwhm / central_wavelength**2
            offsets = np.linspace(
                -span_fwhm * omega_fwhm,
                span_fwhm * omega_fwhm,
                n_components,
            )
            frequencies = omega0 + offsets
            scale = omega_fwhm / (2 * np.log(2) ** (1 / spectral_order))
            density = np.exp(-np.abs(offsets / scale) ** spectral_order)
        elif conversion == "exact_wavelength_density":
            wavelength_min = central_wavelength - span_fwhm * wavelength_fwhm
            wavelength_max = central_wavelength + span_fwhm * wavelength_fwhm
            if wavelength_min <= 0:
                raise ValueError(
                    "the exact wavelength interval includes non-positive values"
                )
            frequencies = np.linspace(
                2 * np.pi * C / wavelength_max,
                2 * np.pi * C / wavelength_min,
                n_components,
            )
            wavelengths = 2 * np.pi * C / frequencies
            scale = wavelength_fwhm / (2 * np.log(2) ** (1 / spectral_order))
            density_lambda = np.exp(
                -np.abs((wavelengths - central_wavelength) / scale) ** spectral_order
            )
            density = density_lambda * (2 * np.pi * C / frequencies**2)
            density /= np.max(density)
        else:
            raise ValueError(
                "conversion must be 'narrowband' or 'exact_wavelength_density'"
            )
        if np.any(frequencies <= 0):
            raise ValueError("the requested spectrum includes non-positive frequencies")
        phases = _phase_array(spectral_phase, frequencies)
        return cls(
            angular_frequencies=frequencies,
            relative_energy_density=density,
            delta_omega=float(frequencies[1] - frequencies[0]),
            total_energy=float(total_energy),
            central_wavelength=float(central_wavelength),
            wavelength_fwhm=float(wavelength_fwhm),
            spectral_order=float(spectral_order),
            spectral_phase=phases,
            conversion=conversion,
        )


@dataclass(frozen=True)
class GaussianPulseSpectrum(_DiscreteSpectrumMethods):
    """Transform-limited Gaussian pulse specified by intensity FWHM.

    The positive-frequency energy density is Gaussian in angular-frequency
    detuning.  Its square-root component amplitudes reconstruct a field
    envelope whose intensity FWHM is ``intensity_fwhm``.  ``span_fwhm`` is
    the retained half-range measured in full spectral FWHM values.

    ``minimum_period`` is especially useful for EPOCH files: when
    ``n_components`` is omitted, the frequency spacing is chosen so the
    discrete representation does not repeat within that time.  The returned
    count is always odd, keeping the carrier as an explicit component.
    """

    angular_frequencies: np.ndarray
    relative_energy_density: np.ndarray
    delta_omega: float
    total_energy: float
    central_wavelength: float
    intensity_fwhm: float
    span_fwhm: float
    spectral_phase: object = 0.0

    def __post_init__(self):
        frequencies, density, phases, spacing = _validated_spectrum_arrays(
            self.angular_frequencies,
            self.relative_energy_density,
            self.spectral_phase,
        )
        metadata = (
            self.total_energy,
            self.central_wavelength,
            self.intensity_fwhm,
            self.span_fwhm,
        )
        if any(not np.isfinite(value) or value <= 0 for value in metadata):
            raise ValueError("Gaussian pulse metadata must be positive and finite")
        if not np.isclose(self.delta_omega, spacing, rtol=1e-10):
            raise ValueError("delta_omega does not match the frequency grid")
        carrier = 2 * np.pi * C / self.central_wavelength
        centre_index = len(frequencies) // 2
        if len(frequencies) % 2 != 1 or not np.isclose(
            frequencies[centre_index], carrier, rtol=1e-12, atol=0.0
        ):
            raise ValueError(
                "the frequency grid must contain the carrier at its centre"
            )
        object.__setattr__(self, "angular_frequencies", frequencies)
        object.__setattr__(self, "relative_energy_density", density)
        object.__setattr__(self, "spectral_phase", phases)
        object.__setattr__(self, "delta_omega", spacing)

    @property
    def central_angular_frequency(self):
        """Carrier angular frequency in radians per second."""
        return 2 * np.pi * C / self.central_wavelength

    @property
    def angular_frequency_fwhm(self):
        """Intensity-spectrum FWHM, ``4*ln(2)/intensity_fwhm``."""
        return 4 * np.log(2) / self.intensity_fwhm

    @classmethod
    def from_intensity_fwhm(
        cls,
        *,
        central_wavelength=800e-9,
        intensity_fwhm=30e-15,
        total_energy=1.0,
        span_fwhm=2.0,
        n_components=None,
        minimum_period=None,
        spectral_phase=0.0,
    ):
        """Construct a transform-limited Gaussian temporal spectrum.

        ``total_energy`` sets only the common component scale.  A normalised
        EPOCH profile may therefore use any positive value, while an
        energy-calibrated three-dimensional calculation should retain the
        physical pulse energy and effective area consistently.
        """
        values = (central_wavelength, intensity_fwhm, total_energy, span_fwhm)
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError("pulse parameters must be positive and finite")
        if minimum_period is not None:
            minimum_period = float(minimum_period)
            if not np.isfinite(minimum_period) or minimum_period <= 0:
                raise ValueError("minimum_period must be positive and finite")

        carrier = 2 * np.pi * C / central_wavelength
        frequency_fwhm = 4 * np.log(2) / intensity_fwhm
        half_span = span_fwhm * frequency_fwhm
        if carrier - half_span <= 0:
            raise ValueError("the retained Gaussian spectrum reaches zero frequency")

        if n_components is None:
            if minimum_period is None:
                n_components = 65
            else:
                maximum_spacing = 2 * np.pi / minimum_period
                n_intervals = int(np.ceil(2 * half_span / maximum_spacing))
                n_intervals = max(2, n_intervals)
                if n_intervals % 2:
                    n_intervals += 1
                n_components = n_intervals + 1
        if (
            not isinstance(n_components, (int, np.integer))
            or n_components < 3
            or n_components % 2 != 1
        ):
            raise ValueError("n_components must be an odd integer of at least three")

        offsets = np.linspace(-half_span, half_span, int(n_components))
        frequencies = carrier + offsets
        density = np.exp(-4 * np.log(2) * (offsets / frequency_fwhm) ** 2)
        phases = _phase_array(spectral_phase, frequencies)
        result = cls(
            angular_frequencies=frequencies,
            relative_energy_density=density,
            delta_omega=float(frequencies[1] - frequencies[0]),
            total_energy=float(total_energy),
            central_wavelength=float(central_wavelength),
            intensity_fwhm=float(intensity_fwhm),
            span_fwhm=float(span_fwhm),
            spectral_phase=phases,
        )
        if minimum_period is not None and result.period <= minimum_period:
            raise ValueError(
                "the selected component count does not meet minimum_period"
            )
        return result


def reconstruct_analytic_signal(components, angular_frequencies, times):
    """Reconstruct ``2*sum(E_n exp(-i omega_n t))`` from positive frequencies."""
    components = np.asarray(components, dtype=complex)
    frequencies = np.asarray(angular_frequencies, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if components.ndim < 1 or components.shape[0] != len(frequencies):
        raise ValueError("the first component axis must match angular_frequencies")
    phase = np.exp(-1j * times[:, None] * frequencies[None, :])
    return 2 * np.tensordot(phase, components, axes=(1, 0))


def reconstruct_complex_envelope(
    components,
    angular_frequencies,
    times,
    carrier_angular_frequency,
):
    """Reconstruct the envelope relative to a selected carrier frequency.

    The returned field is
    ``2*sum(E_n exp[-i*(omega_n-omega_carrier)*t])``.  Multiplying it by
    ``exp(-i*omega_carrier*t)`` recovers the analytic signal.  This is the
    correct quantity to split into amplitude and phase for EPOCH-mod's
    spatiotemporal reader because EPOCH supplies the carrier oscillation
    internally.
    """
    components = np.asarray(components, dtype=complex)
    frequencies = np.asarray(angular_frequencies, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    carrier = float(carrier_angular_frequency)
    if components.ndim < 1 or components.shape[0] != len(frequencies):
        raise ValueError("the first component axis must match angular_frequencies")
    if not np.isfinite(carrier) or carrier <= 0:
        raise ValueError("carrier_angular_frequency must be positive and finite")
    phase = np.exp(-1j * times[:, None] * (frequencies[None, :] - carrier))
    return 2 * np.tensordot(phase, components, axes=(1, 0))


def electric_intensity(analytic_electric_field):
    """Return ``epsilon0*c*|E_tilde|**2/2`` in W/m2."""
    field = np.asarray(analytic_electric_field)
    if field.shape[-1] != 3:
        raise ValueError("the final field axis must have length 3")
    return 0.5 * EPSILON_0 * C * np.sum(np.abs(field) ** 2, axis=-1)
