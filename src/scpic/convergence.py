"""Convergence helpers for complete Stratton--Chu observation grids."""

from dataclasses import dataclass

import numpy as np

from .fields import C
from .solvers import evaluate_SC_3D


@dataclass(frozen=True)
class QuadratureConvergenceLevel:
    """Surface orders and field changes measured at one refinement level."""

    n_radial: int
    n_azimuthal: int
    relative_electric_change: float
    relative_magnetic_change: float
    relative_combined_change: float


@dataclass(frozen=True)
class QuadratureConvergenceResult:
    """Final fields and successive changes from an adaptive quadrature run."""

    levels: tuple
    electric: np.ndarray
    magnetic: np.ndarray
    converged: bool


def _relative_change(current, previous):
    denominator = np.linalg.norm(current)
    numerator = np.linalg.norm(current - previous)
    if denominator == 0:
        return 0.0 if numerator == 0 else np.inf
    return float(numerator / denominator)


def surface_quadrature_convergence(
    observation_points,
    mirror,
    incident,
    k,
    *,
    orders=((12, 24), (18, 36), (24, 48), (32, 64)),
    rtol=1e-3,
    amplitude=None,
    spectral_phase=0.0,
    include_contours=False,
    solver_options=None,
):
    """Refine mirror quadrature until a complete observation grid converges.

    The physically scaled combined metric compares ``(E, cB)`` between
    successive levels.  This avoids a misleading magnetic relative error for
    symmetry points at which the magnetic field vanishes.  For broadband
    work, run this check at the shortest material frequency and on the actual
    EPOCH injection plane.
    """
    observation_points = np.asarray(observation_points, dtype=float)
    if observation_points.ndim != 2 or observation_points.shape[1] != 3:
        raise ValueError("observation_points must have shape (n, 3)")
    if k <= 0 or rtol <= 0:
        raise ValueError("k and rtol must be positive")
    validated_orders = []
    for order in orders:
        if len(order) != 2:
            raise ValueError("each quadrature order must be (n_radial, n_azimuthal)")
        radial, azimuthal = order
        if (
            not isinstance(radial, (int, np.integer))
            or not isinstance(azimuthal, (int, np.integer))
            or radial < 1
            or azimuthal < 1
        ):
            raise ValueError("quadrature orders must be positive integers")
        validated_orders.append((int(radial), int(azimuthal)))
    if len(validated_orders) < 2:
        raise ValueError("at least two quadrature levels are required")
    solver_options = {} if solver_options is None else dict(solver_options)

    previous_electric = None
    previous_magnetic = None
    levels = []
    converged = False
    electric = magnetic = None
    for radial, azimuthal in validated_orders:
        surface = mirror.surface_quadrature(radial, azimuthal)
        electric_incident, magnetic_incident = incident.fields(
            surface.points,
            k=k,
            amplitude=amplitude,
            spectral_phase=spectral_phase,
        )
        contours = ()
        contour_magnetic = ()
        if include_contours:
            rims = ["outer"]
            if mirror.inner_diameter > 0:
                rims.append("inner")
            contours = tuple(
                mirror.contour_quadrature(max(azimuthal, 8), rim=rim) for rim in rims
            )
            contour_magnetic = tuple(
                incident.fields(
                    contour.points,
                    k=k,
                    amplitude=amplitude,
                    spectral_phase=spectral_phase,
                )[1]
                for contour in contours
            )
        electric, magnetic = evaluate_SC_3D(
            observation_points,
            surface,
            electric_incident,
            magnetic_incident,
            k,
            contours=contours,
            B_inc_contours=contour_magnetic,
            **solver_options,
        )
        if previous_electric is None:
            electric_change = magnetic_change = combined_change = np.nan
        else:
            electric_change = _relative_change(electric, previous_electric)
            magnetic_change = _relative_change(magnetic, previous_magnetic)
            current_combined = np.concatenate((electric.ravel(), C * magnetic.ravel()))
            previous_combined = np.concatenate(
                (previous_electric.ravel(), C * previous_magnetic.ravel())
            )
            combined_change = _relative_change(current_combined, previous_combined)
        levels.append(
            QuadratureConvergenceLevel(
                n_radial=radial,
                n_azimuthal=azimuthal,
                relative_electric_change=electric_change,
                relative_magnetic_change=magnetic_change,
                relative_combined_change=combined_change,
            )
        )
        if previous_electric is not None and combined_change <= rtol:
            converged = True
            break
        previous_electric = electric
        previous_magnetic = magnetic

    return QuadratureConvergenceResult(
        levels=tuple(levels),
        electric=electric,
        magnetic=magnetic,
        converged=converged,
    )
