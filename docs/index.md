# SCPIC documentation

SCPIC is a preprocessing package for propagating an upstream laser field from
a parabolic reflector to a focal region or EPOCH injection plane. The
three-dimensional solver uses the physical-optics Stratton--Chu formulation;
the separate two-dimensional solver is a TM reduction intended for EPOCH2D.

## Where to start

| Goal | Document |
|---|---|
| Generate a first monochromatic or broadband profile | [User guide](user_guide.md) |
| Generate a pulsed f/2-style EPOCH2D profile | [User guide, EPOCH export](user_guide.md#10-epoch-export) |
| Decide whether a result is sufficiently converged | [Validation and acceptance](validation.md) |
| Look up classes, functions, array shapes, and return values | [API reference](api_reference.md) |
| Understand the implemented equations and paper benchmarks | [Three-dimensional methodology](methodology_3d.md) |
| Model chromatic aberrations and pulse-front couplings | [Space-time couplings](space_time_couplings.md) |
| Run the local modified-EPOCH test cases | [EPOCH validation guide](../epoch_tests/README.md) |
| Review related implementations and literature | [Literature review](literature_review.md) |
| See intentionally deferred development | [Roadmap](roadmap.md) |
| Follow the implementation history | [Developer log](../dev_log.md) |

## Supported workflow

```text
mirror geometry
      +
incident spatial field, polarisation, spectrum, phase and OPD
      |
      v
Stratton--Chu surface integration
      |
      +--> focal profiles and volume diagnostics
      |
      +--> complex field on an EPOCH injection line or plane
                     |
                     v
        EPOCH amplitude/phase conversion
                     |
                     v
           local EPOCH propagation test
```

The supported optical model is a perfect-conductor physical-optics reflector.
Finite coating throughput may be represented by adjusting the incident energy.
A general frequency-dependent complex coating response is not currently a
first-class API.

## Release status

The package is ready for controlled scientific use when the convergence and
EPOCH validation steps in [validation.md](validation.md) are followed. It is
not a black-box guarantee that EPOCH's `simple_laser` boundary reproduces every
high-numerical-aperture field component. In particular, that boundary does not
directly impose the longitudinal electric field.

The present evidence includes:

- a 2D paraxial Gaussian waist benchmark;
- focus-defined pulsed 2D TM profile and group-delay regressions;
- all six Vallières linear/TM01 and HNAP/OAP90/TP benchmarks;
- independent surface and spectral convergence studies;
- frequency-domain and time-domain Maxwell residual tests;
- energy-normalisation, energy-density, and Poynting-flux tests;
- non-separable chromatic Zernike phase and Jolly STC regressions;
- direct comparison with the local EPOCH-mod file reader;
- three EPOCH2D and three EPOCH3D workstation-scale propagation cases.

CUDA execution, multi-rank campaign I/O, and a large focused-volume energy
closure calculation still require validation on suitable hardware.
