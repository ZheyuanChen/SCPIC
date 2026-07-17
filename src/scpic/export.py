"""Export complex SCPIC phasors for EPOCH-mod custom laser injection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class EpochProfileExport:
    """Metadata needed to configure the matching EPOCH laser block."""

    amplitude_file: Path
    phase_file: Path
    shape: tuple[int, ...]
    field_scale: float


@dataclass(frozen=True)
class EpochPhaseDiagnostics:
    """Phase-continuity indicators for an EPOCH amplitude/phase profile."""

    low_amplitude_fraction: float
    maximum_reliable_phase_step: float
    winding_cell_count: int

    @property
    def has_phase_singularity(self):
        """Whether a reliable two-dimensional cell contains phase winding."""
        return self.winding_cell_count > 0


def _wrapped_phase_difference(difference):
    return np.angle(np.exp(1j * difference))


def epoch_phase_diagnostics(field, *, amplitude_floor=1e-6):
    """Diagnose branch-cut and phase-singularity risks before EPOCH export.

    ``amplitude_floor`` is relative to the peak field magnitude.  Adjacent
    phase steps and winding cells touching less reliable samples are excluded,
    because phase is physically undefined at zero amplitude.  A nonzero
    ``winding_cell_count`` indicates that no globally continuous scalar phase
    exists on at least one two-dimensional slice of the stored array.
    """
    field = np.asarray(field)
    if field.ndim not in {1, 2, 3}:
        raise ValueError("an EPOCH profile must be a 1D, 2D, or 3D array")
    if field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("field must be non-empty and finite")
    amplitude_floor = float(amplitude_floor)
    if not np.isfinite(amplitude_floor) or amplitude_floor < 0 or amplitude_floor >= 1:
        raise ValueError("amplitude_floor must satisfy 0 <= value < 1")

    magnitude = np.abs(field)
    peak = float(np.max(magnitude))
    if peak == 0:
        raise ValueError("field must contain at least one non-zero value")
    reliable = magnitude > amplitude_floor * peak
    phase = np.angle(field)
    maximum_step = 0.0
    for axis in range(field.ndim):
        low = [slice(None)] * field.ndim
        high = [slice(None)] * field.ndim
        low[axis] = slice(None, -1)
        high[axis] = slice(1, None)
        valid = reliable[tuple(low)] & reliable[tuple(high)]
        if np.any(valid):
            differences = _wrapped_phase_difference(
                phase[tuple(high)] - phase[tuple(low)]
            )
            maximum_step = max(maximum_step, float(np.max(np.abs(differences[valid]))))

    winding_cell_count = 0
    for first_axis in range(field.ndim):
        for second_axis in range(first_axis + 1, field.ndim):
            corners = []
            for first_high, second_high in (
                (False, False),
                (True, False),
                (True, True),
                (False, True),
            ):
                selection = [slice(None)] * field.ndim
                selection[first_axis] = (
                    slice(1, None) if first_high else slice(None, -1)
                )
                selection[second_axis] = (
                    slice(1, None) if second_high else slice(None, -1)
                )
                corners.append(tuple(selection))
            valid = np.logical_and.reduce([reliable[corner] for corner in corners])
            if not np.any(valid):
                continue
            circulation = sum(
                _wrapped_phase_difference(
                    phase[corners[(index + 1) % 4]] - phase[corners[index]]
                )
                for index in range(4)
            )
            winding_cell_count += int(
                np.count_nonzero(valid & (np.abs(circulation) > np.pi))
            )

    return EpochPhaseDiagnostics(
        low_amplitude_fraction=float(1.0 - np.mean(reliable)),
        maximum_reliable_phase_step=maximum_step,
        winding_cell_count=winding_cell_count,
    )


def epoch_amplitude_phase(
    field,
    *,
    field_scale=None,
    phase_reference=0.0,
    unwrap_phase=True,
    phase_amplitude_floor=None,
):
    """Convert an SCPIC phasor to normalised EPOCH amplitude and phase.

    SCPIC uses ``Re[F exp(-i omega t)]`` while EPOCH-mod injects
    ``A sin(omega t + phase)``.  Consequently the phase written to EPOCH is
    ``pi/2 - (angle(F) - phase_reference)``.  ``field_scale`` is the EPOCH
    laser's peak electric-field amplitude; if omitted, the peak magnitude of
    ``field`` is used.  Returned amplitudes are therefore in ``[0, 1]``.

    For a static profile, ``field`` is a monochromatic complex phasor.  For a
    spatiotemporal profile, it must be the complex envelope relative to the
    carrier configured in the EPOCH laser block, not a full analytic signal:
    EPOCH adds ``omega*t`` internally.

    EPOCH linearly interpolates phase as an ordinary real number.  Therefore
    the phase is unwrapped along every array axis by default; a wrapped
    ``[-pi, pi)`` representation would create false interpolation ramps at
    each branch cut.  Set ``unwrap_phase=False`` only when the caller will
    handle phase continuity separately.  Use
    :func:`scpic.epoch_phase_diagnostics` before exporting a field that may
    contain a physical phase singularity; no scalar unwrapping can remove its
    topological branch cut. If ``phase_amplitude_floor`` is supplied, phase
    below that fraction of peak amplitude is replaced by the nearest reliable
    sample before unwrapping. Phase is undefined there, and this reduces false
    interpolation ramps caused by harmless numerical tail noise. A physical
    phase singularity can still require an explicit branch cut; inspect
    :func:`scpic.epoch_phase_diagnostics` and the stored phase steps.
    """
    field = np.asarray(field)
    if field.ndim not in {1, 2, 3}:
        raise ValueError("an EPOCH profile must be a 1D, 2D, or 3D array")
    if field.size == 0 or not np.all(np.isfinite(field)):
        raise ValueError("field must be non-empty and finite")

    phase_reference = float(phase_reference)
    if not np.isfinite(phase_reference):
        raise ValueError("phase_reference must be finite")
    if not isinstance(unwrap_phase, (bool, np.bool_)):
        raise TypeError("unwrap_phase must be boolean")
    if phase_amplitude_floor is not None:
        phase_amplitude_floor = float(phase_amplitude_floor)
        if (
            not np.isfinite(phase_amplitude_floor)
            or phase_amplitude_floor < 0
            or phase_amplitude_floor >= 1
        ):
            raise ValueError("phase_amplitude_floor must satisfy 0 <= value < 1")

    magnitude = np.abs(field)
    peak = float(np.max(magnitude))
    if field_scale is None:
        field_scale = peak
    field_scale = float(field_scale)
    if not np.isfinite(field_scale) or field_scale <= 0:
        raise ValueError("field_scale must be finite and positive")
    if peak > field_scale * (1.0 + 10 * np.finfo(float).eps):
        raise ValueError("field_scale is smaller than the field's peak magnitude")

    amplitude = magnitude / field_scale
    phase = np.pi / 2 - (np.angle(field) - phase_reference)
    if phase_amplitude_floor is not None:
        reliable = magnitude > phase_amplitude_floor * peak
        if np.any(~reliable):
            nearest = distance_transform_edt(
                ~reliable,
                return_distances=False,
                return_indices=True,
            )
            phase = np.where(reliable, phase, phase[tuple(nearest)])
    if unwrap_phase:
        for axis in range(phase.ndim):
            phase = np.unwrap(phase, axis=axis)
    else:
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
    if phase_amplitude_floor is None:
        phase = np.where(magnitude == 0, 0.0, phase)
    return amplitude.astype(np.float64), phase.astype(np.float64), field_scale


def export_epoch_profile(
    directory,
    field,
    *,
    field_scale=None,
    phase_reference=0.0,
    unwrap_phase=True,
    phase_amplitude_floor=None,
    amplitude_filename="amplitude.dat",
    phase_filename="phase.dat",
):
    """Write a headerless EPOCH-mod profile pair as native float64 streams.

    Arrays are written in NumPy C order.  This directly matches EPOCH-mod's
    Fortran arrays when the NumPy axes are ordered as documented there:
    ``(n_y,)`` for 2D static data, ``(n_t, n_y)`` for 2D spatiotemporal
    data, ``(n_tr2, n_tr1)`` for 3D static data, and
    ``(n_t, n_tr2, n_tr1)`` for 3D spatiotemporal data.

    A time-dependent ``field`` must be a carrier-referenced complex envelope.
    Use :func:`scpic.reconstruct_complex_envelope` or pass
    ``carrier_angular_frequency`` to
    :func:`scpic.iter_broadband_field_chunks`; do not export the full analytic
    signal because EPOCH supplies the carrier oscillation itself.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    amplitude, phase, field_scale = epoch_amplitude_phase(
        field,
        field_scale=field_scale,
        phase_reference=phase_reference,
        unwrap_phase=unwrap_phase,
        phase_amplitude_floor=phase_amplitude_floor,
    )
    amplitude_file = directory / amplitude_filename
    phase_file = directory / phase_filename
    np.ascontiguousarray(amplitude, dtype=np.float64).tofile(amplitude_file)
    np.ascontiguousarray(phase, dtype=np.float64).tofile(phase_file)
    return EpochProfileExport(
        amplitude_file=amplitude_file,
        phase_file=phase_file,
        shape=amplitude.shape,
        field_scale=field_scale,
    )


def export_field_binary(filepath_base, field_array, dtype=np.float64):
    """Backward-compatible wrapper for the old ``*_amp.bin`` interface.

    EPOCH-mod only accepts 64-bit values, so other dtypes are rejected.  New
    code should use :func:`export_epoch_profile`, which also returns the
    electric-field scale required by the deck.
    """
    if np.dtype(dtype) != np.dtype(np.float64):
        raise ValueError("epoch_dev requires native float64 profile data")
    base = Path(filepath_base)
    return export_epoch_profile(
        base.parent,
        field_array,
        amplitude_filename=f"{base.name}_amp.bin",
        phase_filename=f"{base.name}_phase.bin",
    )


def export_all_fields(*args, **kwargs):
    """Reject the obsolete volume-field export with an actionable message."""
    raise RuntimeError(
        "EPOCH's custom laser reader accepts one transverse electric-field "
        "profile, not Ex/Ez/By volume dumps; use export_epoch_profile()"
    )
