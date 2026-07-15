"""Discrete broadband pulse construction following Vallières et al. (2023)."""

from dataclasses import dataclass

import numpy as np

from .fields import C

EPSILON_0 = 8.854_187_8128e-12


@dataclass(frozen=True)
class SuperGaussianSpectrum:
    """Uniformly sampled, energy-normalised super-Gaussian spectrum."""

    angular_frequencies: np.ndarray
    relative_energy_density: np.ndarray
    delta_omega: float
    total_energy: float
    central_wavelength: float
    wavelength_fwhm: float
    spectral_order: float

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
    ):
        """Construct the paper's spectrum using its narrow-band conversion.

        The wavelength FWHM is converted to angular frequency with
        ``Delta omega = 2*pi*c*Delta lambda/lambda0**2``, as in the paper.
        ``span_fwhm`` is the half-width of the sampled interval in FWHM units.
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
        omega_fwhm = 2 * np.pi * C * wavelength_fwhm / central_wavelength**2
        offsets = np.linspace(
            -span_fwhm * omega_fwhm,
            span_fwhm * omega_fwhm,
            n_components,
        )
        frequencies = omega0 + offsets
        if np.any(frequencies <= 0):
            raise ValueError("the requested spectrum includes non-positive frequencies")
        delta_omega = frequencies[1] - frequencies[0]
        scale = omega_fwhm / (2 * np.log(2) ** (1 / spectral_order))
        density = np.exp(-np.abs(offsets / scale) ** spectral_order)
        return cls(
            angular_frequencies=frequencies,
            relative_energy_density=density,
            delta_omega=float(delta_omega),
            total_energy=float(total_energy),
            central_wavelength=float(central_wavelength),
            wavelength_fwhm=float(wavelength_fwhm),
            spectral_order=float(spectral_order),
        )

    @property
    def period(self):
        """Period of the discrete Fourier representation, ``2*pi/Delta omega``."""
        return 2 * np.pi / self.delta_omega

    def component_amplitudes(self, effective_area):
        """Return peak incident electric-field phasors for each frequency.

        With the paper's positive-frequency convention, each component has
        time-averaged Poynting flux ``2*eps0*c*|E_n|**2``. ``effective_area``
        may be one scalar or one value per frequency. The returned amplitudes
        therefore recover ``total_energy`` over ``period``.
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


def reconstruct_analytic_signal(components, angular_frequencies, times):
    """Reconstruct ``2*sum(E_n exp(-i omega_n t))`` from positive frequencies."""
    components = np.asarray(components, dtype=complex)
    frequencies = np.asarray(angular_frequencies, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if components.ndim < 1 or components.shape[0] != len(frequencies):
        raise ValueError("the first component axis must match angular_frequencies")
    phase = np.exp(-1j * times[:, None] * frequencies[None, :])
    return 2 * np.tensordot(phase, components, axes=(1, 0))


def electric_intensity(analytic_electric_field):
    """Return ``epsilon0*c*|E_tilde|**2/2`` in W/m2."""
    field = np.asarray(analytic_electric_field)
    if field.shape[-1] != 3:
        raise ValueError("the final field axis must have length 3")
    return 0.5 * EPSILON_0 * C * np.sum(np.abs(field) ** 2, axis=-1)
