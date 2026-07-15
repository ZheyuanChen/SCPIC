import numpy as np
from scipy.special import hankel1


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
