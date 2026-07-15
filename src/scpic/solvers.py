import numpy as np
from scipy.special import hankel1

def evaluate_SC_2D(x_obs, z_obs, x_m, z_m, nx, nz, dl, By_inc, dBy_dn_inc, k):
    """
    Evaluates the 2D Stratton-Chu integral for the TM mode.
    Returns By at the observation points.
    """
    By_obs = np.zeros_like(x_obs, dtype=complex)
    
    for i in range(len(x_obs)):
        dx = x_obs[i] - x_m
        dz = z_obs[i] - z_m
        R = np.sqrt(dx**2 + dz**2)
        
        # Prevent singularity if observation point is exactly on the mirror
        R[R == 0] = 1e-12 
        
        # 2D Green's Function
        G = 0.25j * hankel1(0, k * R)
        
        # Derivative of Green's function with respect to normal n
        # dH0(x)/dx = -H1(x)
        dG_dR = -0.25j * k * hankel1(1, k * R)
        
        # dot product of grad_m(R) and normal
        dR_dn = -(dx * nx + dz * nz) / R
        dG_dn = dG_dR * dR_dn
        
        # SC Integral
        integrand = By_inc * dG_dn - G * dBy_dn_inc
        By_obs[i] = np.sum(integrand * dl)
        
    return By_obs