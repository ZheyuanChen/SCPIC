"""Field-level diagnostics for monochromatic complex phasors."""

from dataclasses import dataclass

import numpy as np

from .fields import C


@dataclass(frozen=True)
class MaxwellResiduals:
    """Dimensionless RMS residuals of the four source-free Maxwell equations."""

    divergence_e: float
    divergence_b: float
    faraday: float
    ampere: float


def _rms(values):
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _normalised_rms(residual, scale):
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
