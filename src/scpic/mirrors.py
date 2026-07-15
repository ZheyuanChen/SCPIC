from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceQuadrature3D:
    """Quadrature data for an open reflector surface."""

    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray
    projected_weights: np.ndarray
    aperture_coordinates: np.ndarray


@dataclass(frozen=True)
class ContourQuadrature3D:
    """Quadrature data for one oriented rim of an open reflector."""

    points: np.ndarray
    normals: np.ndarray
    d_ell: np.ndarray


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


class ParabolicMirror3D:
    """Circular or annular aperture cut from a parent paraboloid.

    The parent surface is ``z = (x**2 + y**2)/(4*f0) - f0`` and its focus
    is the origin.  Apertures are defined in the projected ``x-y`` plane.
    For ``OAP90`` the aperture centre is ``(2*f0, 0)``, so its chief ray
    travelling along ``-z`` is reflected along ``-x``.
    """

    _MIRROR_TYPES = {"OAP90", "HNAP", "TP", "offset"}

    def __init__(
        self,
        f0,
        D,
        mirror_type="OAP90",
        offset=(0.0, 0.0),
        inner_diameter=0.0,
    ):
        if f0 <= 0 or D <= 0:
            raise ValueError("f0 and D must be positive")
        if mirror_type not in self._MIRROR_TYPES:
            raise ValueError("mirror_type must be 'OAP90', 'HNAP', 'TP', or 'offset'")
        if inner_diameter < 0 or inner_diameter >= D:
            raise ValueError("inner_diameter must satisfy 0 <= inner_diameter < D")
        if mirror_type != "TP" and inner_diameter != 0:
            raise ValueError("inner_diameter is only supported for mirror_type='TP'")

        offset = np.asarray(offset, dtype=float)
        if offset.shape != (2,):
            raise ValueError("offset must contain exactly two coordinates")

        self.f0 = float(f0)
        self.D = float(D)
        self.mirror_type = mirror_type
        self.offset = tuple(offset)
        self.inner_diameter = float(inner_diameter)

    @classmethod
    def vallieres_oap90(cls):
        """Return the 220-mm, 115-mm-effective-focal-length paper geometry."""
        return cls(f0=57.5e-3, D=220e-3, mirror_type="OAP90")

    @property
    def aperture_centre(self):
        if self.mirror_type == "OAP90":
            return np.array([2 * self.f0, 0.0])
        if self.mirror_type in {"HNAP", "TP"}:
            return np.zeros(2)
        return np.asarray(self.offset)

    @property
    def effective_focal_length(self):
        """Distance from the OAP chief-ray point to the focus."""
        centre = self.aperture_centre
        point = self._points_from_xy(centre[None, :])[0]
        return float(np.linalg.norm(point))

    @property
    def projected_area(self):
        return np.pi * (self.D**2 - self.inner_diameter**2) / 4

    def _points_from_xy(self, xy):
        xy = np.asarray(xy, dtype=float)
        z = np.sum(xy**2, axis=-1) / (4 * self.f0) - self.f0
        return np.column_stack((xy, z))

    def _normals_from_xy(self, xy):
        xy = np.asarray(xy, dtype=float)
        unnormalised = np.column_stack(
            (-xy[:, 0] / (2 * self.f0), -xy[:, 1] / (2 * self.f0), np.ones(len(xy)))
        )
        return unnormalised / np.linalg.norm(unnormalised, axis=1)[:, None]

    def surface_quadrature(self, n_radial=40, n_azimuthal=80):
        """Return tensor-product Gauss--Legendre surface quadrature."""
        if n_radial < 1 or n_azimuthal < 2:
            raise ValueError("n_radial >= 1 and n_azimuthal >= 2 are required")

        radial_nodes, radial_weights = np.polynomial.legendre.leggauss(n_radial)
        azimuth_nodes, azimuth_weights = np.polynomial.legendre.leggauss(n_azimuthal)
        rho_min = self.inner_diameter / 2
        rho_max = self.D / 2
        rho = rho_min + (radial_nodes + 1) * (rho_max - rho_min) / 2
        w_rho = radial_weights * (rho_max - rho_min) / 2
        phi = np.pi * (azimuth_nodes + 1)
        w_phi = np.pi * azimuth_weights

        rho_grid, phi_grid = np.meshgrid(rho, phi, indexing="ij")
        wr_grid, wp_grid = np.meshgrid(w_rho, w_phi, indexing="ij")
        local_xy = np.column_stack(
            (
                (rho_grid * np.cos(phi_grid)).ravel(),
                (rho_grid * np.sin(phi_grid)).ravel(),
            )
        )
        xy = local_xy + self.aperture_centre
        points = self._points_from_xy(xy)
        normals = self._normals_from_xy(xy)
        projected_weights = (rho_grid * wr_grid * wp_grid).ravel()
        surface_jacobian = np.sqrt(
            1 + (xy[:, 0] / (2 * self.f0)) ** 2 + (xy[:, 1] / (2 * self.f0)) ** 2
        )

        return SurfaceQuadrature3D(
            points=points,
            normals=normals,
            weights=projected_weights * surface_jacobian,
            projected_weights=projected_weights,
            aperture_coordinates=local_xy,
        )

    def contour_quadrature(self, n_azimuthal=160, rim="outer"):
        """Return an oriented Gauss--Legendre quadrature for an aperture rim."""
        if n_azimuthal < 2:
            raise ValueError("n_azimuthal must be at least 2")
        if rim not in {"outer", "inner"}:
            raise ValueError("rim must be 'outer' or 'inner'")
        if rim == "inner" and self.inner_diameter == 0:
            raise ValueError("this mirror has no inner rim")

        nodes, weights = np.polynomial.legendre.leggauss(n_azimuthal)
        phi = np.pi * (nodes + 1)
        w_phi = np.pi * weights
        radius = self.D / 2 if rim == "outer" else self.inner_diameter / 2
        orientation = 1.0 if rim == "outer" else -1.0
        local_xy = radius * np.column_stack((np.cos(phi), np.sin(phi)))
        xy = local_xy + self.aperture_centre
        points = self._points_from_xy(xy)
        normals = self._normals_from_xy(xy)
        tangent = np.column_stack(
            (
                -radius * np.sin(phi),
                radius * np.cos(phi),
                radius
                * (-xy[:, 0] * np.sin(phi) + xy[:, 1] * np.cos(phi))
                / (2 * self.f0),
            )
        )
        return ContourQuadrature3D(
            points=points,
            normals=normals,
            d_ell=orientation * tangent * w_phi[:, None],
        )
