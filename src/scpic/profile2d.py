"""Broadband two-dimensional TM propagation and EPOCH profile generation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from .export import epoch_phase_diagnostics, export_epoch_profile
from .fields import C, IncidentFieldTM, electric_from_magnetic_tm
from .mirrors import ParabolicMirror2D
from .pulse import (
    GaussianPulseSpectrum,
    reconstruct_analytic_signal,
    reconstruct_complex_envelope,
)
from .solvers import evaluate_SC_2D


@dataclass(frozen=True)
class BroadbandPropagation2DResult:
    """A broadband TM field reconstructed on a rectilinear ``(x, z)`` grid.

    ``electric`` has shape ``(n_t, n_x, n_z, 2)`` with components
    ``(E_x, E_z)``. ``magnetic`` has shape ``(n_t, n_x, n_z)`` and stores
    ``B_y``. Under the campaign mapping ``x_epoch = focus_x - x_scpic`` and
    ``y_epoch = z_scpic``, the corresponding EPOCH fields are
    ``E_x_epoch=-E_x``, ``E_y_epoch=E_z`` and ``B_z_epoch=B_y``.
    """

    x: np.ndarray
    z: np.ndarray
    times: np.ndarray
    angular_frequencies: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray


@dataclass(frozen=True)
class Epoch2DPulseGeneration:
    """Files and summary metadata produced by :func:`generate_epoch2d_oap_pulse`."""

    amplitude_file: Path
    phase_file: Path
    manifest_file: Path
    shape: tuple[int, int]
    generation_seconds: float


def _frequency_values(values, wavenumbers, name):
    """Broadcast a scalar or evaluate frequency-dependent input on ``k``."""
    if callable(values):
        values = values(wavenumbers)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full_like(wavenumbers, float(values))
    if values.shape != wavenumbers.shape or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite and match the frequency grid")
    return values


def _validated_axis(values, name, *, minimum_size=3):
    """Return a finite, strictly increasing one-dimensional coordinate axis."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 1
        or len(values) < minimum_size
        or np.any(~np.isfinite(values))
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError(
            f"{name} must be a finite, strictly increasing one-dimensional "
            f"array with at least {minimum_size} values"
        )
    return values


def _reference_grid(point, step):
    """Build the 3-by-3 stencil used to recover the reference ``E_z``."""
    point = np.asarray(point, dtype=float)
    step = float(step)
    if point.shape != (2,) or np.any(~np.isfinite(point)):
        raise ValueError("reference_point must contain finite (x, z) coordinates")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("reference_step must be positive and finite")
    return (
        point[0] + np.array([-step, 0.0, step]),
        point[1] + np.array([-step, 0.0, step]),
    )


def _solve_unit_frequency(
    k,
    *,
    x,
    z,
    mirror_surface,
    incident,
    boundary_model,
    solver_chunk_size,
    reference_grid,
):
    """Propagate a unit incident pupil at one wavenumber.

    The optional reference stencil is evaluated separately from the output
    grid. This allows focal normalisation even when the requested output is an
    upstream EPOCH boundary.
    """
    x_m, z_m, nx, nz, dl, x_center = mirror_surface
    by_incident = incident.B_y(x_m, z_m, x_center=x_center, k=k)
    dby_incident = incident.dBy_dn(
        x_m,
        z_m,
        nx,
        nz,
        x_center=x_center,
        k=k,
    )
    X, Z = np.meshgrid(x, z, indexing="ij")
    magnetic = evaluate_SC_2D(
        X.ravel(),
        Z.ravel(),
        x_m,
        z_m,
        nx,
        nz,
        dl,
        by_incident,
        dby_incident,
        k,
        boundary_model=boundary_model,
        chunk_size=solver_chunk_size,
    ).reshape(X.shape)
    electric_x, electric_z = electric_from_magnetic_tm(magnetic, x, z, k)
    electric = np.stack((electric_x, electric_z), axis=-1)

    reference_value = None
    if reference_grid is not None:
        x_reference, z_reference = reference_grid
        X_reference, Z_reference = np.meshgrid(x_reference, z_reference, indexing="ij")
        magnetic_reference = evaluate_SC_2D(
            X_reference.ravel(),
            Z_reference.ravel(),
            x_m,
            z_m,
            nx,
            nz,
            dl,
            by_incident,
            dby_incident,
            k,
            boundary_model=boundary_model,
            chunk_size=solver_chunk_size,
        ).reshape(X_reference.shape)
        _, electric_z_reference = electric_from_magnetic_tm(
            magnetic_reference,
            x_reference,
            z_reference,
            k,
        )
        reference_value = electric_z_reference[1, 1]
    return electric, magnetic, reference_value


def propagate_broadband_2d(
    x,
    z,
    mirror,
    incident,
    spectrum,
    times,
    *,
    num_surface_points=3000,
    effective_area=1.0,
    propagation_phase=0.0,
    carrier_angular_frequency=None,
    envelope_peak_time=None,
    reference_point=None,
    reference_step=None,
    reference_normalisation="none",
    boundary_model="pec_physical_optics",
    solver_chunk_size=64,
    workers=1,
):
    """Propagate a pulsed 2D TM field frequency by frequency.

    The output grid is rectilinear because recovering both electric components
    from ``B_y`` requires spatial derivatives. The input spectrum supplies the
    desired positive-frequency component amplitudes.

    ``reference_normalisation='complex'`` makes the reference-point ``E_z``
    spectrum exactly equal to the requested spectrum. This is the appropriate
    mode when a transform-limited duration is defined at the focus and the
    required upstream spectrum is to be inferred. ``'phase'`` removes only the
    reference phase, retaining the mirror's spectral amplitude response.
    ``'none'`` propagates the supplied upstream spectrum unchanged.

    ``envelope_peak_time`` adds the linear spectral phase required for the
    reference pulse to peak at that time. With a focus reference, propagation
    then determines the earlier boundary arrival without a manually imposed
    delay. ``workers`` parallelises independent frequencies with shared-memory
    threads; set the Slurm request as one task with matching CPUs per task.
    """
    x = _validated_axis(x, "x")
    z = _validated_axis(z, "z")
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if times.ndim != 1 or np.any(~np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    if not isinstance(mirror, ParabolicMirror2D):
        raise TypeError("mirror must be a ParabolicMirror2D")
    if not isinstance(incident, IncidentFieldTM):
        raise TypeError("incident must be an IncidentFieldTM")
    if not isinstance(num_surface_points, (int, np.integer)) or num_surface_points < 2:
        raise ValueError("num_surface_points must be an integer of at least two")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if reference_normalisation not in {"none", "phase", "complex"}:
        raise ValueError(
            "reference_normalisation must be 'none', 'phase', or 'complex'"
        )
    if reference_normalisation != "none" and reference_point is None:
        raise ValueError("reference_point is required for reference normalisation")
    if reference_point is not None and reference_step is None:
        raise ValueError("reference_step is required with reference_point")
    if envelope_peak_time is not None:
        envelope_peak_time = float(envelope_peak_time)
        if not np.isfinite(envelope_peak_time):
            raise ValueError("envelope_peak_time must be finite")
        if carrier_angular_frequency is None:
            raise ValueError("envelope_peak_time requires a carrier frequency")

    if carrier_angular_frequency is not None:
        carrier_angular_frequency = float(carrier_angular_frequency)
        if not np.isfinite(carrier_angular_frequency) or carrier_angular_frequency <= 0:
            raise ValueError("carrier_angular_frequency must be positive and finite")

    mirror_surface = mirror.get_surface(int(num_surface_points))
    reference_grid = (
        None
        if reference_point is None
        else _reference_grid(reference_point, reference_step)
    )
    angular_frequencies = np.asarray(spectrum.angular_frequencies, dtype=float)
    wavenumbers = angular_frequencies / C
    component_coefficients = spectrum.component_coefficients(effective_area)
    additional_phase = _frequency_values(
        propagation_phase, wavenumbers, "propagation_phase"
    )

    def solve(k):
        return _solve_unit_frequency(
            k,
            x=x,
            z=z,
            mirror_surface=mirror_surface,
            incident=incident,
            boundary_model=boundary_model,
            solver_chunk_size=solver_chunk_size,
            reference_grid=reference_grid,
        )

    if workers == 1:
        solved = [solve(k) for k in wavenumbers]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            solved = list(executor.map(solve, wavenumbers))

    electric_components = np.stack([item[0] for item in solved])
    magnetic_components = np.stack([item[1] for item in solved])
    references = [item[2] for item in solved]
    scales = component_coefficients * np.exp(1j * additional_phase)
    if reference_normalisation != "none":
        references = np.asarray(references, dtype=complex)
        threshold = np.finfo(float).eps * np.max(np.abs(references))
        if np.any(np.abs(references) <= threshold):
            raise ValueError("reference E_z is zero or numerically unresolved")
        if reference_normalisation == "phase":
            # Remove the focal spectral phase but retain the mirror's
            # frequency-dependent gain.
            scales *= np.exp(-1j * np.angle(references))
        else:
            # Invert the complete complex response. After multiplication the
            # reference E_z coefficients equal component_coefficients exactly.
            scales /= references
    if envelope_peak_time is not None:
        # With exp[-i(omega-omega_c)t], a positive spectral slope delays the
        # carrier-referenced envelope to envelope_peak_time.
        scales *= np.exp(
            1j * (angular_frequencies - carrier_angular_frequency) * envelope_peak_time
        )
    electric_components *= scales[:, None, None, None]
    magnetic_components *= scales[:, None, None]

    if carrier_angular_frequency is None:
        reconstruct = reconstruct_analytic_signal
        reconstruction_options = {}
    else:
        reconstruct = reconstruct_complex_envelope
        reconstruction_options = {
            "carrier_angular_frequency": carrier_angular_frequency
        }
    return BroadbandPropagation2DResult(
        x=x.copy(),
        z=z.copy(),
        times=times.copy(),
        angular_frequencies=angular_frequencies.copy(),
        electric=reconstruct(
            electric_components,
            angular_frequencies,
            times,
            **reconstruction_options,
        ),
        magnetic=reconstruct(
            magnetic_components,
            angular_frequencies,
            times,
            **reconstruction_options,
        ),
    )


def _fwhm(samples, values):
    """Measure FWHM using linear interpolation at the outer crossings."""
    samples = np.asarray(samples, dtype=float)
    values = np.asarray(values, dtype=float)
    if samples.ndim != 1 or values.shape != samples.shape:
        raise ValueError("FWHM samples and values must be matching arrays")
    if np.any(~np.isfinite(values)) or np.max(values) <= 0:
        raise ValueError("FWHM values must be finite and non-zero")
    above = values >= 0.5 * np.max(values)
    indices = np.flatnonzero(above)
    if len(indices) < 2 or indices[0] == 0 or indices[-1] == len(values) - 1:
        return None

    def crossing(low, high):
        target = 0.5 * np.max(values)
        fraction = (target - values[low]) / (values[high] - values[low])
        return samples[low] + fraction * (samples[high] - samples[low])

    left = crossing(indices[0] - 1, indices[0])
    right = crossing(indices[-1], indices[-1] + 1)
    return float(right - left)


def _sha256(path):
    """Return the hexadecimal SHA-256 digest of a generated profile file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_epoch2d_oap_pulse(
    directory,
    *,
    central_wavelength=800e-9,
    intensity_fwhm=30e-15,
    focus_distance=24e-6,
    boundary_peak_time=143.31487238414786e-15,
    f_number=2.0,
    effective_focal_length=50.8e-3,
    aperture_radius_in_beam_waists=3.0,
    time_start=0.0,
    time_end=400e-15,
    n_time=801,
    transverse_min=-20e-6,
    transverse_max=20e-6,
    n_transverse=1601,
    n_surface=6000,
    n_components=None,
    spectrum_span_fwhm=2.0,
    derivative_step=None,
    workers=1,
    solver_chunk_size=64,
    phase_reference=0.0,
    phase_amplitude_floor=1e-3,
    file_prefix="laser",
):
    """Generate a campaign-ready ``(time, transverse)`` EPOCH2D profile.

    The OAP90 result propagates along SCPIC ``-x``. The exported tangential
    component uses the right-handed coordinate transformation
    ``x_epoch=focus_x-x_scpic`` and ``y_epoch=z_scpic``; it therefore injects
    along EPOCH ``+x`` from ``x_min`` without conjugating the phasor.

    The Gaussian duration is imposed at the intended focus through complex
    reference normalisation. The boundary arrival is a prediction of the
    frequency-resolved propagation rather than a separately imposed temporal
    shift. Output amplitudes are normalised shapes; production field strength
    must still be calibrated in EPOCH with ``intensity_w_cm2``.
    """
    positive = {
        "central_wavelength": central_wavelength,
        "intensity_fwhm": intensity_fwhm,
        "focus_distance": focus_distance,
        "f_number": f_number,
        "effective_focal_length": effective_focal_length,
        "aperture_radius_in_beam_waists": aperture_radius_in_beam_waists,
    }
    if any(not np.isfinite(value) or value <= 0 for value in positive.values()):
        raise ValueError("wavelength, duration, geometry and aperture must be positive")
    if not time_start < boundary_peak_time < time_end:
        raise ValueError("boundary_peak_time must lie inside the output time window")
    if not time_start < time_end or not transverse_min < transverse_max:
        raise ValueError("time and transverse bounds must be increasing")
    for name, value, minimum in (
        ("n_time", n_time, 3),
        ("n_transverse", n_transverse, 3),
        ("n_surface", n_surface, 2),
    ):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if derivative_step is None:
        derivative_step = central_wavelength / 100
    derivative_step = float(derivative_step)
    if not np.isfinite(derivative_step) or derivative_step <= 0:
        raise ValueError("derivative_step must be positive and finite")

    start = time.perf_counter()
    times = np.linspace(time_start, time_end, int(n_time))
    transverse = np.linspace(transverse_min, transverse_max, int(n_transverse))
    carrier = 2 * np.pi * C / central_wavelength
    focus_time = boundary_peak_time + focus_distance / C
    # Keep the first periodic replica outside the requested EPOCH time window.
    minimum_period = 1.25 * (time_end - time_start)
    spectrum = GaussianPulseSpectrum.from_intensity_fwhm(
        central_wavelength=central_wavelength,
        intensity_fwhm=intensity_fwhm,
        total_energy=1.0,
        span_fwhm=spectrum_span_fwhm,
        n_components=n_components,
        minimum_period=minimum_period,
    )
    spectrum.validate_envelope_time_samples(times, carrier)

    # The campaign f-number is defined with the Gaussian 1/e field diameter,
    # not with the wider numerical integration aperture.
    incident_radius = effective_focal_length / (2 * f_number)
    mirror = ParabolicMirror2D(
        f0=effective_focal_length / 2,
        D=2 * aperture_radius_in_beam_waists * incident_radius,
        mirror_type="OAP90",
    )
    incident = IncidentFieldTM(
        w0=incident_radius,
        wavelength=central_wavelength,
        E0=1.0,
    )
    # Three longitudinal planes are required because E_z is recovered from
    # dB_y/dx. Only the centre plane is passed to EPOCH.
    x = focus_distance + derivative_step * np.array([-1.0, 0.0, 1.0])
    propagation = propagate_broadband_2d(
        x,
        transverse,
        mirror,
        incident,
        spectrum,
        times,
        num_surface_points=int(n_surface),
        carrier_angular_frequency=carrier,
        envelope_peak_time=focus_time,
        reference_point=(0.0, 0.0),
        reference_step=derivative_step,
        reference_normalisation="complex",
        solver_chunk_size=solver_chunk_size,
        workers=workers,
    )
    # This monochromatic focal solve is independent of the broadband inverse
    # normalisation and provides a direct optical reference for the manifest.
    focus_x = derivative_step * np.array([-1.0, 0.0, 1.0])
    focus_electric, _, _ = _solve_unit_frequency(
        carrier / C,
        x=focus_x,
        z=transverse,
        mirror_surface=mirror.get_surface(int(n_surface)),
        incident=incident,
        boundary_model="pec_physical_optics",
        solver_chunk_size=solver_chunk_size,
        reference_grid=None,
    )
    carrier_focus_intensity = np.abs(focus_electric[1, :, 1]) ** 2
    carrier_focus_intensity_fwhm = _fwhm(transverse, carrier_focus_intensity)
    carrier_focus_waist = (
        None
        if carrier_focus_intensity_fwhm is None
        else carrier_focus_intensity_fwhm / np.sqrt(2 * np.log(2))
    )
    carrier_focus_longitudinal_ratio = float(
        np.max(np.abs(focus_electric[1, :, 0]))
        / np.max(np.abs(focus_electric[1, :, 1]))
    )
    # Native E_z becomes tangential E_y after mapping the -x SCPIC solution to
    # a +x EPOCH laser at x_min.
    epoch_envelope = propagation.electric[:, 1, :, 1]
    diagnostics = epoch_phase_diagnostics(epoch_envelope)
    directory = Path(directory)
    exported = export_epoch_profile(
        directory,
        epoch_envelope,
        phase_reference=phase_reference,
        phase_amplitude_floor=phase_amplitude_floor,
        amplitude_filename=f"{file_prefix}_amplitude.dat",
        phase_filename=f"{file_prefix}_phase.dat",
    )
    # Check the values as written, since EPOCH interpolates this unwrapped
    # float64 phase rather than the original complex envelope.
    stored_phase = np.fromfile(exported.phase_file, dtype=np.float64).reshape(
        exported.shape
    )
    stored_phase_steps = [
        float(np.max(np.abs(np.diff(stored_phase, axis=axis))))
        for axis in range(stored_phase.ndim)
    ]

    intensity = np.abs(epoch_envelope) ** 2
    centre_index = int(np.argmin(np.abs(transverse)))
    peak_time_index = int(np.argmax(intensity[:, centre_index]))
    boundary_duration = _fwhm(times, intensity[:, centre_index])
    boundary_width = _fwhm(transverse, intensity[peak_time_index])
    peak_transverse = np.max(np.abs(propagation.electric[peak_time_index, 1, :, 1]))
    peak_longitudinal = np.max(np.abs(propagation.electric[peak_time_index, 1, :, 0]))
    focus_reference_peak = float(
        2 * np.abs(np.sum(spectrum.component_coefficients(1.0)))
    )
    elapsed = time.perf_counter() - start
    try:
        package_version = importlib.metadata.version("scpic")
    except importlib.metadata.PackageNotFoundError:
        package_version = "local/uninstalled"

    files = (exported.amplitude_file, exported.phase_file)
    manifest = {
        "generator": "scpic.generate_epoch2d_oap_pulse",
        "scpic_version": package_version,
        "profile_model": (
            "frequency-resolved 2D TM physical-optics OAP90; Gaussian focus "
            "spectrum; transformed to EPOCH +x slab geometry"
        ),
        "shape": list(exported.shape),
        "dtype": "native float64",
        "configuration": {
            "central_wavelength_m": central_wavelength,
            "intensity_fwhm_at_focus_s": intensity_fwhm,
            "focus_distance_m": focus_distance,
            "boundary_peak_time_requested_s": boundary_peak_time,
            "focus_peak_time_s": focus_time,
            "f_number_by_1e_field_diameter": f_number,
            "effective_focal_length_m": effective_focal_length,
            "incident_1e_field_radius_m": incident_radius,
            "computational_aperture_diameter_m": mirror.D,
            "aperture_radius_in_beam_waists": aperture_radius_in_beam_waists,
            "time_start_s": time_start,
            "time_end_s": time_end,
            "n_time": int(n_time),
            "transverse_min_m": transverse_min,
            "transverse_max_m": transverse_max,
            "n_transverse": int(n_transverse),
            "n_surface": int(n_surface),
            "derivative_step_m": derivative_step,
            "solver_chunk_size": int(solver_chunk_size),
            "frequency_workers": int(workers),
            "phase_reference_rad": phase_reference,
            "phase_amplitude_floor": phase_amplitude_floor,
        },
        "spectrum": {
            "n_components": len(spectrum.angular_frequencies),
            "span_fwhm": spectrum.span_fwhm,
            "angular_frequency_fwhm_rad_per_s": spectrum.angular_frequency_fwhm,
            "delta_omega_rad_per_s": spectrum.delta_omega,
            "period_s": spectrum.period,
            "reference_normalisation": "complex E_z at SCPIC focus (0, 0)",
        },
        "orientation": {
            "scpic_propagation": "-x",
            "epoch_propagation": "+x from x_min",
            "coordinate_map": "x_epoch=focus_x-x_scpic; y_epoch=z_scpic",
            "component_map": "E_y_epoch=E_z_scpic; E_x_epoch=-E_x_scpic; B_z_epoch=B_y_scpic",
        },
        "profile_diagnostics": {
            "boundary_peak_time_realised_s": float(times[peak_time_index]),
            "boundary_intensity_fwhm_s": boundary_duration,
            "boundary_intensity_fwhm_transverse_m": boundary_width,
            "boundary_peak_longitudinal_to_transverse_ratio": float(
                peak_longitudinal / peak_transverse
            ),
            "paraxial_focus_1e_field_waist_m": (
                2 * central_wavelength * f_number / np.pi
            ),
            "low_amplitude_fraction": diagnostics.low_amplitude_fraction,
            "maximum_reliable_phase_step_rad": (
                diagnostics.maximum_reliable_phase_step
            ),
            "phase_winding_cell_count": diagnostics.winding_cell_count,
            "phase_diagnostic_amplitude_floor": 1e-6,
            "maximum_stored_phase_step_by_axis_rad": stored_phase_steps,
            "arbitrary_field_scale": exported.field_scale,
        },
        "direct_scpic_reference": {
            "carrier_focus_intensity_fwhm_m": carrier_focus_intensity_fwhm,
            "carrier_focus_1e_field_waist_m": carrier_focus_waist,
            "carrier_focus_peak_longitudinal_to_transverse_ratio": (
                carrier_focus_longitudinal_ratio
            ),
            "focus_to_boundary_peak_field_gain": (
                focus_reference_peak / exported.field_scale
            ),
            "qualification": (
                "Carrier-frequency 2D TM reference. The EPOCH simple_laser "
                "boundary cannot impose its longitudinal component directly."
            ),
        },
        "normalisation_warning": (
            "The files carry normalised shape only. Do not reuse a LASY "
            "focus-to-boundary intensity correction without an EPOCH vacuum "
            "calibration of this SCPIC profile family."
        ),
        "generation_seconds": elapsed,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in files
        },
    }
    manifest_file = directory / f"{file_prefix}_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Epoch2DPulseGeneration(
        amplitude_file=exported.amplitude_file,
        phase_file=exported.phase_file,
        manifest_file=manifest_file,
        shape=exported.shape,
        generation_seconds=elapsed,
    )
