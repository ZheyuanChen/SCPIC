import numpy as np
from scipy.special import hankel1

from .fields import C
from .mirrors import ContourQuadrature3D, SurfaceQuadrature3D


def evaluate_SC_2D(
    x_obs,
    z_obs,
    x_m,
    z_m,
    nx,
    nz,
    dl,
    By_inc,
    dBy_dn_inc,
    k,
    *,
    boundary_model="pec_physical_optics",
):
    """
    Evaluate the 2D boundary integral for a reflected TM field.

    The default ``pec_physical_optics`` model applies the physical-optics
    boundary values for a perfectly conducting mirror: the tangential
    magnetic field is twice the incident value and its normal derivative is
    zero.  ``kirchhoff`` retains the generic boundary representation
    ``By*dG/dn - G*dBy/dn`` for callers that already know both total boundary
    values; incident-field values alone must not be used with that model.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    z_obs = np.asarray(z_obs, dtype=float)
    if x_obs.shape != z_obs.shape or x_obs.ndim != 1:
        raise ValueError(
            "x_obs and z_obs must be one-dimensional arrays of equal shape"
        )
    surface = [np.asarray(v) for v in (x_m, z_m, nx, nz, By_inc)]
    if len({v.shape for v in surface}) != 1 or surface[0].ndim != 1:
        raise ValueError(
            "all mirror-surface arrays must be one-dimensional and equal-sized"
        )
    if np.ndim(dl) > 0 and np.asarray(dl).shape != surface[0].shape:
        raise ValueError(
            "dl must be scalar or have the same shape as the mirror arrays"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    if boundary_model not in {"pec_physical_optics", "kirchhoff"}:
        raise ValueError("unknown boundary_model")
    if boundary_model == "kirchhoff":
        dBy_dn_inc = np.asarray(dBy_dn_inc)
        if dBy_dn_inc.shape != surface[0].shape:
            raise ValueError("dBy_dn_inc must match the mirror arrays")

    By_obs = np.zeros(x_obs.shape, dtype=complex)

    for i in range(len(x_obs)):
        dx = x_obs[i] - x_m
        dz = z_obs[i] - z_m
        R = np.sqrt(dx**2 + dz**2)

        # Prevent singularity if observation point is exactly on the mirror
        if np.any(R == 0):
            raise ValueError(
                "observation points must not lie on the integration surface"
            )

        # 2D Green's Function
        G = 0.25j * hankel1(0, k * R)

        # Derivative of Green's function with respect to normal n
        # dH0(x)/dx = -H1(x)
        dG_dR = -0.25j * k * hankel1(1, k * R)

        # dot product of grad_m(R) and normal
        dR_dn = -(dx * nx + dz * nz) / R
        dG_dn = dG_dR * dR_dn

        if boundary_model == "pec_physical_optics":
            integrand = 2.0 * By_inc * dG_dn
        else:
            integrand = By_inc * dG_dn - G * dBy_dn_inc
        By_obs[i] = np.sum(integrand * dl)

    return By_obs


def _validate_vector_field(values, n_points, name):
    values = np.asarray(values, dtype=complex)
    if values.shape != (n_points, 3):
        raise ValueError(f"{name} must have shape ({n_points}, 3)")
    return values


def _source_gradient_green(observation_points, source_points, k):
    displacement = observation_points[:, None, :] - source_points[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance == 0):
        raise ValueError("observation points must not lie on the integration surface")
    exponential = np.exp(1j * k * distance)
    green = exponential / distance
    radial_derivative = exponential * (1j * k * distance - 1) / distance**2
    gradient_source = (
        -(displacement / distance[:, :, None]) * radial_derivative[:, :, None]
    )
    return green, gradient_source


def _select_array_backend(backend):
    if backend == "numpy":
        return np, np.asarray
    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as error:
            raise ImportError(
                "backend='cupy' requires a CuPy build matching the installed CUDA runtime"
            ) from error
        return cp, cp.asnumpy
    raise ValueError("backend must be 'numpy' or 'cupy'")


def _source_gradient_green_backend(observation_points, source_points, k, xp):
    displacement = observation_points[:, None, :] - source_points[None, :, :]
    distance = xp.linalg.norm(displacement, axis=2)
    singular = xp.any(distance == 0)
    if hasattr(singular, "item"):
        singular = singular.item()
    if singular:
        raise ValueError("observation points must not lie on the integration surface")
    exponential = xp.exp(1j * k * distance)
    green = exponential / distance
    radial_derivative = exponential * (1j * k * distance - 1) / distance**2
    gradient_source = (
        -(displacement / distance[:, :, None]) * radial_derivative[:, :, None]
    )
    return green, gradient_source


def evaluate_SC_3D(
    observation_points,
    surface,
    E_inc,
    B_inc,
    k,
    *,
    chunk_size=64,
    contours=(),
    B_inc_contours=(),
    backend="numpy",
):
    """Evaluate the physical-optics Stratton--Chu reflected field in 3D.

    This is the open-surface formula used by Vallières et al.  The surface
    contribution is always included.  Optional oriented contour data adds
    the electric-field rim term retained in Dumont et al. (2017).

    Parameters use SI units.  Because the papers write the formula with
    ``c = 1``, the electric-field terms containing an incident magnetic
    field include the explicit factor ``c`` here.  ``backend='cupy'`` moves
    each observation chunk and the fixed surface data to a CUDA device, but
    still returns NumPy arrays and leaves ``'numpy'`` as the reference path.
    """
    observation_points = np.asarray(observation_points, dtype=float)
    if observation_points.ndim != 2 or observation_points.shape[1] != 3:
        raise ValueError("observation_points must have shape (n, 3)")
    if not isinstance(surface, SurfaceQuadrature3D):
        raise TypeError("surface must be a SurfaceQuadrature3D")
    if k <= 0:
        raise ValueError("k must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    xp, to_numpy = _select_array_backend(backend)

    n_surface = len(surface.points)
    E_inc = _validate_vector_field(E_inc, n_surface, "E_inc")
    B_inc = _validate_vector_field(B_inc, n_surface, "B_inc")
    if surface.normals.shape != (n_surface, 3):
        raise ValueError("surface normals must have shape (n, 3)")
    if surface.weights.shape != (n_surface,):
        raise ValueError("surface weights must have shape (n,)")

    if isinstance(contours, ContourQuadrature3D):
        contours = (contours,)
    if len(contours) != len(B_inc_contours):
        raise ValueError("each contour needs corresponding incident B fields")

    validated_contours = []
    for contour, B_contour in zip(contours, B_inc_contours):
        if not isinstance(contour, ContourQuadrature3D):
            raise TypeError("contours must contain ContourQuadrature3D objects")
        n_contour = len(contour.points)
        B_contour = _validate_vector_field(B_contour, n_contour, "B_inc_contour")
        if contour.normals.shape != (n_contour, 3) or contour.d_ell.shape != (
            n_contour,
            3,
        ):
            raise ValueError("contour normals and d_ell must have shape (n, 3)")
        validated_contours.append((contour, B_contour))

    n_observation = len(observation_points)
    electric = np.zeros((n_observation, 3), dtype=complex)
    magnetic = np.zeros((n_observation, 3), dtype=complex)
    surface_points_backend = xp.asarray(surface.points)
    surface_normals_backend = xp.asarray(surface.normals)
    surface_weights_backend = xp.asarray(surface.weights)
    E_inc_backend = xp.asarray(E_inc)
    B_inc_backend = xp.asarray(B_inc)
    normal_cross_B = xp.cross(surface_normals_backend, B_inc_backend)
    normal_dot_E = xp.sum(surface_normals_backend * E_inc_backend, axis=1)
    contour_backend = []
    for contour, B_contour in validated_contours:
        contour_points = xp.asarray(contour.points)
        contour_normals = xp.asarray(contour.normals)
        B_contour_backend = xp.asarray(B_contour)
        tangential_B = xp.cross(
            contour_normals, xp.cross(contour_normals, B_contour_backend)
        )
        line_scalar = xp.sum(tangential_B * xp.asarray(contour.d_ell), axis=1)
        contour_backend.append((contour_points, line_scalar))
    factor = 1 / (2 * np.pi)

    for start in range(0, n_observation, chunk_size):
        stop = min(start + chunk_size, n_observation)
        observation_backend = xp.asarray(observation_points[start:stop])
        green, gradient = _source_gradient_green_backend(
            observation_backend, surface_points_backend, k, xp
        )
        electric_integrand = (
            1j * k * C * normal_cross_B[None, :, :] * green[:, :, None]
            + normal_dot_E[None, :, None] * gradient
        )
        magnetic_integrand = xp.cross(normal_cross_B[None, :, :], gradient, axis=-1)
        electric_chunk = factor * xp.sum(
            electric_integrand * surface_weights_backend[None, :, None], axis=1
        )
        magnetic_chunk = factor * xp.sum(
            magnetic_integrand * surface_weights_backend[None, :, None], axis=1
        )

        for contour_points, line_scalar in contour_backend:
            _, contour_gradient = _source_gradient_green_backend(
                observation_backend, contour_points, k, xp
            )
            electric_chunk -= (
                C
                * factor
                / (1j * k)
                * xp.sum(contour_gradient * line_scalar[None, :, None], axis=1)
            )

        electric[start:stop] = to_numpy(electric_chunk)
        magnetic[start:stop] = to_numpy(magnetic_chunk)

    return electric, magnetic
