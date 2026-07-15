"""Stratton--Chu preprocessing for EPOCH laser injection."""

from .export import epoch_amplitude_phase, export_epoch_profile
from .fields import (
    IncidentFieldTM,
    LinearPolarisedSuperGaussian3D,
    electric_from_magnetic_tm,
)
from .mirrors import (
    ContourQuadrature3D,
    ParabolicMirror2D,
    ParabolicMirror3D,
    SurfaceQuadrature3D,
)
from .pulse import (
    SuperGaussianSpectrum,
    electric_intensity,
    reconstruct_analytic_signal,
)
from .solvers import evaluate_SC_2D, evaluate_SC_3D

__all__ = [
    "IncidentFieldTM",
    "LinearPolarisedSuperGaussian3D",
    "ParabolicMirror2D",
    "ParabolicMirror3D",
    "SurfaceQuadrature3D",
    "ContourQuadrature3D",
    "SuperGaussianSpectrum",
    "electric_from_magnetic_tm",
    "electric_intensity",
    "epoch_amplitude_phase",
    "evaluate_SC_2D",
    "evaluate_SC_3D",
    "export_epoch_profile",
    "reconstruct_analytic_signal",
]
