"""Field-level diagnostics for monochromatic complex phasors."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid

from .fields import C

EPSILON_0 = 8.854_187_8128e-12
MU_0 = 1 / (EPSILON_0 * C**2)


@dataclass(frozen=True)
class MaxwellResiduals:
    """Dimensionless RMS residuals of the four source-free Maxwell equations."""

    divergence_e: float
    divergence_b: float
    faraday: float
    ampere: float


def _rms(values):
    """Root-mean-square magnitude for real or complex arrays."""
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _normalised_rms(residual, scale):
    """Normalise an RMS residual, retaining an explicit zero-scale result."""
    scale = float(scale)
    value = _rms(residual)
    if scale == 0:
        return 0.0 if value == 0 else np.inf
    return value / scale


def maxwell_residuals(electric, magnetic, coordinates, k, *, edge_order=2, trim=1):
    """Evaluate source-free Maxwell residuals on a regular Cartesian grid.

    ``electric`` and ``magnetic`` must have shape ``(nx, ny, nz, 3)`` and
    use SCPIC's ``exp(-i omega t)`` convention. ``coordinates`` is ``(x,y,z)``,
    with each entry a one-dimensional coordinate array.  Returned residuals
    are normalised by ``k * rms(E)`` or ``k * rms(B)`` as appropriate.
    ``trim`` excludes finite-difference boundary cells from the RMS measure.
    """
    electric = np.asarray(electric, dtype=complex)
    magnetic = np.asarray(magnetic, dtype=complex)
    if (
        electric.shape != magnetic.shape
        or electric.ndim != 4
        or electric.shape[-1] != 3
    ):
        raise ValueError(
            "electric and magnetic must have matching shape (nx, ny, nz, 3)"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    if edge_order not in {1, 2}:
        raise ValueError("edge_order must be 1 or 2")
    if not isinstance(trim, (int, np.integer)) or trim < 0:
        raise ValueError("trim must be a non-negative integer")
    if len(coordinates) != 3:
        raise ValueError("coordinates must contain x, y, and z arrays")

    axes = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
    for dimension, (axis, size) in enumerate(zip(axes, electric.shape[:3])):
        if axis.ndim != 1 or len(axis) != size:
            raise ValueError(
                f"coordinate axis {dimension} does not match the field grid"
            )
        if len(axis) < edge_order + 1 or np.any(np.diff(axis) <= 0):
            raise ValueError(
                "coordinate axes must be increasing and sufficiently sampled"
            )
        if 2 * trim >= size:
            raise ValueError("trim removes the complete field grid")

    derivatives_e = [
        np.gradient(electric[..., component], *axes, edge_order=edge_order)
        for component in range(3)
    ]
    derivatives_b = [
        np.gradient(magnetic[..., component], *axes, edge_order=edge_order)
        for component in range(3)
    ]

    divergence_e = sum(derivatives_e[component][component] for component in range(3))
    divergence_b = sum(derivatives_b[component][component] for component in range(3))
    curl_e = np.stack(
        (
            derivatives_e[2][1] - derivatives_e[1][2],
            derivatives_e[0][2] - derivatives_e[2][0],
            derivatives_e[1][0] - derivatives_e[0][1],
        ),
        axis=-1,
    )
    curl_b = np.stack(
        (
            derivatives_b[2][1] - derivatives_b[1][2],
            derivatives_b[0][2] - derivatives_b[2][0],
            derivatives_b[1][0] - derivatives_b[0][1],
        ),
        axis=-1,
    )

    omega = k * C
    faraday = curl_e - 1j * omega * magnetic
    ampere = curl_b + 1j * omega / C**2 * electric
    region = tuple(slice(trim, -trim) if trim else slice(None) for _ in range(3))
    vector_region = (*region, slice(None))
    electric_scale = k * _rms(electric[vector_region])
    magnetic_scale = k * _rms(magnetic[vector_region])

    return MaxwellResiduals(
        divergence_e=_normalised_rms(divergence_e[region], electric_scale),
        divergence_b=_normalised_rms(divergence_b[region], magnetic_scale),
        faraday=_normalised_rms(faraday[vector_region], electric_scale),
        ampere=_normalised_rms(ampere[vector_region], magnetic_scale),
    )


def time_domain_maxwell_residuals(
    electric,
    magnetic,
    times,
    coordinates,
    reference_angular_frequency,
    *,
    edge_order=2,
    trim=1,
    time_trim=1,
):
    """Evaluate source-free Maxwell residuals for a sampled broadband field.

    Fields must have shape ``(nt, nx, ny, nz, 3)``.  Real physical fields and
    complex analytic fields are both accepted.  The reference frequency is
    used only to make the four RMS residuals dimensionless.
    """
    electric = np.asarray(electric)
    magnetic = np.asarray(magnetic)
    times = np.asarray(times, dtype=float)
    if (
        electric.shape != magnetic.shape
        or electric.ndim != 5
        or electric.shape[-1] != 3
    ):
        raise ValueError(
            "electric and magnetic must have matching shape (nt, nx, ny, nz, 3)"
        )
    if times.ndim != 1 or len(times) != electric.shape[0]:
        raise ValueError("times must match the leading field axis")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if reference_angular_frequency <= 0:
        raise ValueError("reference_angular_frequency must be positive")
    if edge_order not in {1, 2}:
        raise ValueError("edge_order must be 1 or 2")
    if not isinstance(trim, (int, np.integer)) or trim < 0:
        raise ValueError("trim must be a non-negative integer")
    if not isinstance(time_trim, (int, np.integer)) or time_trim < 0:
        raise ValueError("time_trim must be a non-negative integer")
    if len(times) < edge_order + 1 or 2 * time_trim >= len(times):
        raise ValueError("time grid is too short for the requested derivative and trim")
    if len(coordinates) != 3:
        raise ValueError("coordinates must contain x, y, and z arrays")

    axes = tuple(np.asarray(axis, dtype=float) for axis in coordinates)
    for dimension, (axis, size) in enumerate(zip(axes, electric.shape[1:4])):
        if axis.ndim != 1 or len(axis) != size:
            raise ValueError(
                f"coordinate axis {dimension} does not match the field grid"
            )
        if len(axis) < edge_order + 1 or np.any(np.diff(axis) <= 0):
            raise ValueError(
                "coordinate axes must be increasing and sufficiently sampled"
            )
        if 2 * trim >= size:
            raise ValueError("trim removes the complete field grid")

    derivatives_e = [
        np.gradient(
            electric[..., component],
            *axes,
            axis=(1, 2, 3),
            edge_order=edge_order,
        )
        for component in range(3)
    ]
    derivatives_b = [
        np.gradient(
            magnetic[..., component],
            *axes,
            axis=(1, 2, 3),
            edge_order=edge_order,
        )
        for component in range(3)
    ]
    divergence_e = sum(derivatives_e[component][component] for component in range(3))
    divergence_b = sum(derivatives_b[component][component] for component in range(3))
    curl_e = np.stack(
        (
            derivatives_e[2][1] - derivatives_e[1][2],
            derivatives_e[0][2] - derivatives_e[2][0],
            derivatives_e[1][0] - derivatives_e[0][1],
        ),
        axis=-1,
    )
    curl_b = np.stack(
        (
            derivatives_b[2][1] - derivatives_b[1][2],
            derivatives_b[0][2] - derivatives_b[2][0],
            derivatives_b[1][0] - derivatives_b[0][1],
        ),
        axis=-1,
    )
    d_e_dt = np.gradient(electric, times, axis=0, edge_order=edge_order)
    d_b_dt = np.gradient(magnetic, times, axis=0, edge_order=edge_order)
    faraday = curl_e + d_b_dt
    ampere = curl_b - d_e_dt / C**2

    spatial_region = tuple(
        slice(trim, -trim) if trim else slice(None) for _ in range(3)
    )
    time_region = slice(time_trim, -time_trim) if time_trim else slice(None)
    region = (time_region, *spatial_region)
    vector_region = (*region, slice(None))
    k_reference = reference_angular_frequency / C
    electric_scale = k_reference * _rms(electric[vector_region])
    magnetic_scale = k_reference * _rms(magnetic[vector_region])
    return MaxwellResiduals(
        divergence_e=_normalised_rms(divergence_e[region], electric_scale),
        divergence_b=_normalised_rms(divergence_b[region], magnetic_scale),
        faraday=_normalised_rms(faraday[vector_region], electric_scale),
        ampere=_normalised_rms(ampere[vector_region], magnetic_scale),
    )


def electromagnetic_energy_density(electric, magnetic, *, cycle_averaged=True):
    """Return electromagnetic energy density in J/m3.

    Complex phasors or analytic-signal envelopes should use the default
    cycle-averaged expression.  Set ``cycle_averaged=False`` for sampled real
    physical fields at an instant in time.
    """
    electric = np.asarray(electric)
    magnetic = np.asarray(magnetic)
    if electric.shape != magnetic.shape or electric.shape[-1] != 3:
        raise ValueError("electric and magnetic must have matching final vector axes")
    if cycle_averaged:
        electric_square = np.sum(np.abs(electric) ** 2, axis=-1)
        magnetic_square = np.sum(np.abs(magnetic) ** 2, axis=-1)
        factor = 0.25
    else:
        if np.iscomplexobj(electric) or np.iscomplexobj(magnetic):
            raise ValueError("instantaneous energy requires real physical fields")
        electric_square = np.sum(electric**2, axis=-1)
        magnetic_square = np.sum(magnetic**2, axis=-1)
        factor = 0.5
    return factor * (EPSILON_0 * electric_square + magnetic_square / MU_0)


def _integrate_rectilinear(values, coordinates):
    """Integrate scalar data over matching rectilinear coordinate axes."""
    result = np.asarray(values)
    if result.ndim != len(coordinates):
        raise ValueError(
            "the number of coordinate arrays must match the data dimensions"
        )
    for dimension, coordinate in reversed(list(enumerate(coordinates))):
        coordinate = np.asarray(coordinate, dtype=float)
        if coordinate.ndim != 1 or len(coordinate) != result.shape[dimension]:
            raise ValueError("coordinate arrays must match the data shape")
        if np.any(np.diff(coordinate) <= 0):
            raise ValueError("coordinate arrays must be strictly increasing")
        result = trapezoid(result, x=coordinate, axis=dimension)
    return float(result)


def integrated_field_energy(electric, magnetic, coordinates, *, cycle_averaged=True):
    """Integrate electromagnetic energy over a rectilinear volume."""
    density = electromagnetic_energy_density(
        electric, magnetic, cycle_averaged=cycle_averaged
    )
    return _integrate_rectilinear(density, coordinates)


def integrated_poynting_flux(
    electric,
    magnetic,
    coordinates,
    normal,
    *,
    cycle_averaged=True,
):
    """Integrate signed Poynting flux through a rectilinear plane."""
    electric = np.asarray(electric)
    magnetic = np.asarray(magnetic)
    if electric.shape != magnetic.shape or electric.shape[-1] != 3:
        raise ValueError("electric and magnetic must have matching final vector axes")
    normal = np.asarray(normal, dtype=float)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or np.linalg.norm(normal) == 0
    ):
        raise ValueError("normal must be a finite non-zero 3-vector")
    normal = normal / np.linalg.norm(normal)
    if cycle_averaged:
        poynting = 0.5 / MU_0 * np.real(np.cross(electric, np.conjugate(magnetic)))
    else:
        if np.iscomplexobj(electric) or np.iscomplexobj(magnetic):
            raise ValueError(
                "instantaneous Poynting flux requires real physical fields"
            )
        poynting = np.cross(electric, magnetic) / MU_0
    normal_flux = np.sum(poynting * normal, axis=-1)
    return _integrate_rectilinear(normal_flux, coordinates)


def relative_energy_error(reference_energy, measured_energy):
    """Return signed fractional energy error relative to a positive reference."""
    if not np.isfinite(reference_energy) or reference_energy <= 0:
        raise ValueError("reference_energy must be positive and finite")
    if not np.isfinite(measured_energy) or measured_energy < 0:
        raise ValueError("measured_energy must be non-negative and finite")
    return float(measured_energy / reference_energy - 1)
