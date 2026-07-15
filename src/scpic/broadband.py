"""Memory-bounded and observation-domain-parallel broadband propagation."""

from dataclasses import dataclass

import numpy as np

from .fields import C
from .mirrors import ContourQuadrature3D
from .pulse import reconstruct_analytic_signal
from .solvers import evaluate_SC_3D


@dataclass(frozen=True)
class BroadbandFieldChunk:
    """One globally indexed observation chunk reconstructed at requested times."""

    start: int
    stop: int
    electric: np.ndarray
    magnetic: np.ndarray


@dataclass(frozen=True)
class BroadbandPropagationResult:
    """Contiguous local result; under MPI each rank owns a different slice."""

    start: int
    stop: int
    electric: np.ndarray
    magnetic: np.ndarray


def observation_partition(n_observations, rank=0, size=1):
    """Return the balanced contiguous observation slice owned by one rank."""
    if not isinstance(n_observations, (int, np.integer)) or n_observations < 0:
        raise ValueError("n_observations must be a non-negative integer")
    if (
        not isinstance(rank, (int, np.integer))
        or not isinstance(size, (int, np.integer))
        or size < 1
        or rank < 0
        or rank >= size
    ):
        raise ValueError("rank and size must satisfy 0 <= rank < size")
    quotient, remainder = divmod(int(n_observations), int(size))
    start = rank * quotient + min(rank, remainder)
    stop = start + quotient + (1 if rank < remainder else 0)
    return slice(start, stop)


def _frequency_values(values, wavenumbers, name):
    if callable(values):
        values = values(wavenumbers)
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full_like(wavenumbers, float(values))
    if values.shape != wavenumbers.shape or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite and match the frequency grid")
    return values


def _effective_area(incident, spectrum, effective_area):
    if effective_area is not None:
        return effective_area
    try:
        area = incident.effective_area
    except AttributeError as error:
        raise ValueError(
            "effective_area is required for this incident field"
        ) from error
    if callable(area):
        return area(spectrum.angular_frequencies / C)
    return area


def iter_broadband_field_chunks(
    observation_points,
    surface,
    incident,
    spectrum,
    times,
    *,
    effective_area=None,
    propagation_phase=0.0,
    observation_chunk_size=64,
    contours=(),
    communicator=None,
    solver_options=None,
):
    """Yield reconstructed field chunks without storing a global spectrum.

    ``propagation_phase`` is an additional phase in radians, supplied as a
    scalar, one value per wavenumber, or a callable of the wavenumber array.
    It is useful for removing a common paraboloid optical path.  Spectral
    phase carried by ``spectrum`` is already included in its complex component
    coefficients.

    If an mpi4py-like ``communicator`` is supplied, observations are divided
    contiguously using ``Get_rank()`` and ``Get_size()``.  No MPI dependency is
    imported and no collective operation is performed: each rank can write
    its yielded globally indexed chunks independently.
    """
    observation_points = np.asarray(observation_points, dtype=float)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    if observation_points.ndim != 2 or observation_points.shape[1] != 3:
        raise ValueError("observation_points must have shape (n, 3)")
    if times.ndim != 1 or np.any(~np.isfinite(times)):
        raise ValueError("times must be a finite one-dimensional array")
    if (
        not isinstance(observation_chunk_size, (int, np.integer))
        or observation_chunk_size < 1
    ):
        raise ValueError("observation_chunk_size must be a positive integer")
    solver_options = {} if solver_options is None else dict(solver_options)

    if communicator is None:
        rank, size = 0, 1
    else:
        try:
            rank = communicator.Get_rank()
            size = communicator.Get_size()
        except AttributeError as error:
            raise TypeError(
                "communicator must provide Get_rank() and Get_size()"
            ) from error
    local = observation_partition(len(observation_points), rank, size)
    wavenumbers = np.asarray(spectrum.angular_frequencies, dtype=float) / C
    phase = _frequency_values(propagation_phase, wavenumbers, "propagation_phase")
    area = _effective_area(incident, spectrum, effective_area)
    coefficients = spectrum.component_coefficients(area)

    if isinstance(contours, ContourQuadrature3D):
        contours = (contours,)
    else:
        contours = tuple(contours)
    incident_surface_fields = []
    incident_contour_fields = []
    for k, coefficient, component_phase in zip(wavenumbers, coefficients, phase):
        incident_surface_fields.append(
            incident.fields(
                surface.points,
                k=k,
                amplitude=coefficient,
                spectral_phase=component_phase,
            )
        )
        incident_contour_fields.append(
            tuple(
                incident.fields(
                    contour.points,
                    k=k,
                    amplitude=coefficient,
                    spectral_phase=component_phase,
                )[1]
                for contour in contours
            )
        )

    for start in range(local.start, local.stop, observation_chunk_size):
        stop = min(start + observation_chunk_size, local.stop)
        points = observation_points[start:stop]
        electric_components = np.empty(
            (len(wavenumbers), len(points), 3), dtype=complex
        )
        magnetic_components = np.empty_like(electric_components)
        for index, k in enumerate(wavenumbers):
            electric_incident, magnetic_incident = incident_surface_fields[index]
            electric_components[index], magnetic_components[index] = evaluate_SC_3D(
                points,
                surface,
                electric_incident,
                magnetic_incident,
                k,
                contours=contours,
                B_inc_contours=incident_contour_fields[index],
                **solver_options,
            )
        yield BroadbandFieldChunk(
            start=start,
            stop=stop,
            electric=reconstruct_analytic_signal(
                electric_components, spectrum.angular_frequencies, times
            ),
            magnetic=reconstruct_analytic_signal(
                magnetic_components, spectrum.angular_frequencies, times
            ),
        )


def propagate_broadband_3d(*args, **kwargs):
    """Collect local chunks from :func:`iter_broadband_field_chunks`.

    In serial the result spans every observation. Under MPI it contains only
    the contiguous slice owned by the current rank, identified by ``start``
    and ``stop``. Use the iterator directly when even that local result should
    be streamed to disk rather than retained.
    """
    chunks = list(iter_broadband_field_chunks(*args, **kwargs))
    if chunks:
        start, stop = chunks[0].start, chunks[-1].stop
        electric = np.concatenate([chunk.electric for chunk in chunks], axis=1)
        magnetic = np.concatenate([chunk.magnetic for chunk in chunks], axis=1)
    else:
        times = np.atleast_1d(np.asarray(args[4] if len(args) > 4 else kwargs["times"]))
        observation_points = np.asarray(
            args[0] if args else kwargs["observation_points"]
        )
        communicator = kwargs.get("communicator")
        if communicator is None:
            local = observation_partition(len(observation_points))
        else:
            local = observation_partition(
                len(observation_points),
                communicator.Get_rank(),
                communicator.Get_size(),
            )
        start, stop = local.start, local.stop
        electric = np.empty((len(times), 0, 3), dtype=complex)
        magnetic = np.empty_like(electric)
    return BroadbandPropagationResult(
        start=start,
        stop=stop,
        electric=electric,
        magnetic=magnetic,
    )
