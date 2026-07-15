import numpy as np

class IncidentFieldTM:
    def __init__(self, w0, wavelength, E0=1.0):
        self.w0 = w0
        self.k = 2 * np.pi / wavelength
        self.c = 299792458.0
        self.B0 = E0 / self.c

    def B_y(self, x, z, x_center=0.0):
        """Magnetic field evaluating traveling in -z direction"""
        envelope = np.exp(-((x - x_center) / self.w0)**2)
        phase = np.exp(-1j * self.k * z)
        return self.B0 * envelope * phase
    
    def dBy_dn(self, x, z, nx, nz, x_center=0.0):
        """Directional derivative of By along the mirror normal"""
        envelope = np.exp(-((x - x_center) / self.w0)**2)
        phase = np.exp(-1j * self.k * z)
        
        dB_dx = self.B0 * (-2 * (x - x_center) / self.w0**2) * envelope * phase
        dB_dz = self.B0 * (-1j * self.k) * envelope * phase
        
        return dB_dx * nx + dB_dz * nz