"""Export complex SCPIC phasors for EPOCH-mod custom laser injection."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EpochProfileExport:
    """Metadata needed to configure the matching EPOCH laser block."""

    amplitude_file: Path
    phase_file: Path
    shape: tuple[int, ...]
    field_scale: float


def epoch_amplitude_phase(
    field,
    *,
    field_scale=None,
    phase_reference=0.0,
    unwrap_phase=True,
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
    handle phase continuity separately.
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
    if unwrap_phase:
        for axis in range(phase.ndim):
            phase = np.unwrap(phase, axis=axis)
    else:
        phase = (phase + np.pi) % (2 * np.pi) - np.pi
    phase = np.where(magnitude == 0, 0.0, phase)
    return amplitude.astype(np.float64), phase.astype(np.float64), field_scale


def export_epoch_profile(
    directory,
    field,
    *,
    field_scale=None,
    phase_reference=0.0,
    unwrap_phase=True,
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
