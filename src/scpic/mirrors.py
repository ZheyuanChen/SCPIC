import numpy as np


class ParabolicMirror2D:
    def __init__(self, f0, D, mirror_type="OAP90", offset=0.0):
        """
        f0: Focal length
        D: Beam/Mirror diameter
        mirror_type: 'HNAP' (on-axis) or 'OAP90' (90-deg off-axis)
        """
        if f0 <= 0 or D <= 0:
            raise ValueError("f0 and D must be positive")
        if mirror_type not in {"OAP90", "HNAP", "offset"}:
            raise ValueError("mirror_type must be 'OAP90', 'HNAP', or 'offset'")
        self.f0 = f0
        self.D = D
        self.mirror_type = mirror_type
        self.offset = offset

    def get_surface(self, num_points=2000):
        if num_points < 2:
            raise ValueError("num_points must be at least 2")
        # The 90-degree reflection point is at x = 2*f0
        if self.mirror_type == "OAP90":
            x_center = 2 * self.f0
        elif self.mirror_type == "HNAP":
            x_center = 0.0
        else:  # explicitly requested generic offset segment
            x_center = self.offset

        # 1D array representing the mirror surface in x
        x_m = np.linspace(x_center - self.D / 2, x_center + self.D / 2, num_points)
        # Parabola equation: z = x^2 / (4f) - f
        z_m = (x_m**2) / (4 * self.f0) - self.f0

        # Calculate the normal vector n = (nx, nz) pointing INWARD towards the incoming beam
        dz_dx = x_m / (2 * self.f0)
        norm = np.sqrt(1 + dz_dx**2)
        nx = -dz_dx / norm
        nz = 1.0 / norm

        # Trapezoidal quadrature weights for dl = sqrt(1 + (dz/dx)^2) dx.
        # Returning one weight per surface point avoids double-weighting the
        # two endpoints in the boundary integral.
        weights = np.ones(num_points)
        weights[[0, -1]] = 0.5
        dl = norm * (x_m[1] - x_m[0]) * weights

        return x_m, z_m, nx, nz, dl, x_center
