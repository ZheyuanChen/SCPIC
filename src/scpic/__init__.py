"""Stratton--Chu preprocessing for EPOCH laser injection."""

from .broadband import (
    BroadbandFieldChunk,
    BroadbandPropagationResult,
    iter_broadband_field_chunks,
    observation_partition,
    propagate_broadband_3d,
)
from .convergence import (
    QuadratureConvergenceLevel,
    QuadratureConvergenceResult,
    surface_quadrature_convergence,
)
from .diagnostics import (
    MaxwellResiduals,
    electromagnetic_energy_density,
    integrated_field_energy,
    integrated_poynting_flux,
    maxwell_residuals,
    relative_energy_error,
    time_domain_maxwell_residuals,
)
from .export import epoch_amplitude_phase, export_epoch_profile
from .fields import (
    IncidentFieldTM,
    FiniteRayleighTM01Beam3D,
    LinearPolarisedSuperGaussian3D,
    ParaxialGaussLaguerreBeam3D,
    RadiallyPolarisedSuperGaussian3D,
    TM01RadiallyPolarisedBeam3D,
    ZernikeWavefront,
    electric_from_magnetic_tm,
)
from .mirrors import (
    ContourQuadrature3D,
    ParabolicMirror2D,
    ParabolicMirror3D,
    SurfaceQuadrature3D,
)
from .pulse import (
    SampledSpectrum,
    SuperGaussianSpectrum,
    electric_intensity,
    reconstruct_analytic_signal,
    reconstruct_complex_envelope,
)
from .solvers import evaluate_SC_2D, evaluate_SC_3D

__all__ = [
    "IncidentFieldTM",
    "FiniteRayleighTM01Beam3D",
    "LinearPolarisedSuperGaussian3D",
    "ParaxialGaussLaguerreBeam3D",
    "RadiallyPolarisedSuperGaussian3D",
    "TM01RadiallyPolarisedBeam3D",
    "ZernikeWavefront",
    "MaxwellResiduals",
    "BroadbandFieldChunk",
    "BroadbandPropagationResult",
    "QuadratureConvergenceLevel",
    "QuadratureConvergenceResult",
    "ParabolicMirror2D",
    "ParabolicMirror3D",
    "SurfaceQuadrature3D",
    "ContourQuadrature3D",
    "SuperGaussianSpectrum",
    "SampledSpectrum",
    "electric_from_magnetic_tm",
    "electric_intensity",
    "electromagnetic_energy_density",
    "integrated_field_energy",
    "integrated_poynting_flux",
    "iter_broadband_field_chunks",
    "maxwell_residuals",
    "observation_partition",
    "propagate_broadband_3d",
    "relative_energy_error",
    "surface_quadrature_convergence",
    "time_domain_maxwell_residuals",
    "epoch_amplitude_phase",
    "evaluate_SC_2D",
    "evaluate_SC_3D",
    "export_epoch_profile",
    "reconstruct_analytic_signal",
    "reconstruct_complex_envelope",
]
