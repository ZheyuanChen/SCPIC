# Roadmap and deferred features

The current package contains the core optical workflow required to generate
and validate 2D or 3D profiles for the modified EPOCH reader. The items below
are worthwhile, but none is a prerequisite for using the supported
perfect-conductor workflow with the documented checks.

## Priority 1: full-vector EPOCH injection

### Interior current-sheet or antenna injector

This is the most important remaining scientific feature.

EPOCH's `simple_laser` boundary accepts transverse electric profiles and
constructs the associated magnetic field. It cannot impose the longitudinal
field of a near-unity-NA focus and its characteristic approximation can alter
large-angle content.

A complete injector should:

- impose both tangential electric and magnetic data on an interior plane;
- reproduce a one-way field without a counter-propagating pulse;
- support two transverse dimensions and time dependence;
- preserve EPOCH domain decomposition and guard-cell handling;
- include energy accounting and vacuum propagation regressions;
- be tested against analytic plane waves before a Stratton--Chu profile.

This belongs partly in the EPOCH modification rather than solely in SCPIC.

## Priority 2: measured optical data adapters

The core already accepts arbitrary OPD callables and sampled spectra. Useful
adapters would depend on actual instrument outputs:

- Phasics SID4 or OAsys wavefront exports;
- generic regular-grid and scattered-point OPD files;
- pupil masks and invalid-pixel treatment;
- phase unwrapping and wavelength metadata;
- conversion of reflective surface metrology to OPD;
- measured near-field amplitude maps;
- complex frequency-dependent coating reflectivity.

These should be implemented from representative files with known units and
coordinate conventions, not from guessed formats.

## Priority 3: campaign configuration and provenance

A declarative campaign runner could read a versioned TOML or YAML file and
write:

- profile files;
- a complete metadata record;
- convergence results;
- diagnostic summaries;
- runtime and memory estimates;
- EPOCH deck fragments.

This would reduce manual mistakes across parameter scans. It should be added
only after the campaign's stable set of inputs and outputs is known.

## Priority 4: hardware and scaling validation

### CUDA

The CuPy algebra is unit-tested through an array-module stub, but production
CUDA execution still needs:

- numerical comparison with NumPy across representative geometries;
- chunk-size and surface-size tuning;
- memory-limit handling;
- timing over frequency count and observation count;
- a documented CUDA/CuPy compatibility matrix.

### MPI

Observation partitioning is deterministic and does not require mpi4py at
import time. Remaining validation includes:

- multi-rank numerical comparison with serial output;
- rank-local file assembly;
- failure-safe metadata and restart behaviour;
- scaling on Viking or another cluster;
- avoiding shared-filesystem bottlenecks.

## Priority 5: general reflectors and upstream propagation

The present geometry is a paraboloid with circular or annular projected
apertures. Possible extensions are:

- triangulated or parametric arbitrary reflector surfaces;
- multiple coherent reflectors;
- frequency-dependent surface response;
- imported complex near-field data on a non-parabolic surface;
- frequency-dependent beam centre, width, direction, and polarisation;
- position-dependent spectral amplitude and filtering.

These broaden the scientific model and should be benchmarked independently.
Position--frequency phase, including Jolly chromatic tilt, defocus and
trefoil, is now implemented through `ChromaticZernikePhase`.

## Lower-priority engineering improvements

- automated documentation generation from docstrings;
- packaged command-line entry points;
- Zarr or HDF5 campaign writers;
- resumable on-disk spectral caches;
- profiling and optional compiled CPU kernels;
- continuous integration across supported Python versions;
- release packaging, versioning, citation metadata, and an explicit licence.

The licence and project-name collision with Popov's earlier PIC code should be
resolved before public distribution beyond the current repository.

## Features deliberately not planned as silent defaults

- multiplying TM01 results by an empirical Vallières correction factor;
- assuming a universal mirror quadrature order;
- inferring coating loss from mirror material alone;
- discarding the longitudinal field without reporting it;
- treating a paraxial Strehl ratio as an exact near-unity-NA intensity ratio;
- automatically extrapolating measured wavefronts outside their valid pupil.
