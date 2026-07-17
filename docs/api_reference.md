# Public API reference

This reference summarises the objects exported from `scpic`. Detailed
mathematics and workflows are in [methodology_3d.md](methodology_3d.md) and
[user_guide.md](user_guide.md).

## Geometry

### `ParabolicMirror2D(f0, D, mirror_type="OAP90", offset=0.0)`

Two-dimensional parabolic segment.

- `get_surface(num_points=2000)` returns
  `(x, z, nx, nz, dl, x_center)`.
- Supported types: `"OAP90"`, `"HNAP"`, and `"offset"`.

### `ParabolicMirror3D(f0, D, mirror_type="OAP90", offset=(0, 0), inner_diameter=0)`

Circular or annular projected aperture on the parent paraboloid.

Factories:

- `vallieres_oap90()`
- `vallieres_hnap()`
- `vallieres_tp(obstruction_ratio=0.20, ...)`
- `fourmaux_tp_2025()`

Properties:

- `aperture_centre`
- `effective_focal_length`
- `projected_area`
- `focusing_angle_range`
- `generalized_numerical_aperture`

Quadratures:

- `surface_quadrature(n_radial=40, n_azimuthal=80)`
- `contour_quadrature(n_azimuthal=160, rim="outer")`

`SurfaceQuadrature3D` contains:

- `points`: `(n_surface, 3)`
- `normals`: `(n_surface, 3)`
- `weights`: surface-area quadrature
- `projected_weights`: aperture-plane quadrature
- `aperture_coordinates`: local projected pupil coordinates

`ContourQuadrature3D` contains `points`, `normals`, and oriented `d_ell`.

## Incident fields

Every 3D incident-field `fields()` method returns
`(electric, magnetic)`, each with shape `(n_points, 3)`.

Common keyword arguments:

- `k`: wavenumber; required if the object has no construction wavelength;
- `amplitude`: complex spectral coefficient;
- `spectral_phase`: additional phase in radians.

Common construction options:

- `wavefront_opd(points)`: fixed optical path difference in metres;
- `spatio_spectral_phase(points, angular_frequency)`: position--frequency
  phase in radians.

### `LinearPolarisedSuperGaussian3D`

Collimated super-Gaussian with arbitrary propagation and transverse linear
polarisation.

Useful members:

- `from_intensity_fwhm(intensity_fwhm, **kwargs)`
- `effective_area`
- `fields(points, *, k=None, amplitude=None, spectral_phase=0)`

### `RadiallyPolarisedSuperGaussian3D`

Simplified transverse radial envelope. Its electric field is set to zero on
the axis. It is not the Vallières TM01 field.

### `TM01RadiallyPolarisedBeam3D`

Collimated paper-accurate TM01 input, including longitudinal electric field.

- `effective_area(k=None)` returns \(\pi/k^2\);
- `fields(...)` returns all vector components.

### `FiniteRayleighTM01Beam3D`

Finite-Rayleigh-range form of the radial Gaussian. The waist is centred at
`centre`; signed longitudinal distance increases along `direction`.

### `ParaxialGaussLaguerreBeam3D`

Axisymmetric radial Gauss--Laguerre expansion for a linearly polarised
paraxial beam.

- `mode_coefficients` may be a sequence or `{n: complex_coefficient}` mapping;
- `effective_area` uses radial-mode orthogonality;
- azimuthal modes are not implemented.

### `IncidentFieldTM`

Two-dimensional monochromatic Gaussian input.

- `B_y(x, z, x_center=0)`
- `dBy_dn(x, z, nx, nz, x_center=0)`

### `ZernikeWavefront`

Callable optical-path-difference model.

```python
ZernikeWavefront(
    pupil_radius,
    coefficients,
    centre=(0, 0, 0),
    axis_u=(1, 0, 0),
    axis_v=(0, 1, 0),
    outside="raise",
)
```

Coefficients use OSA/ANSI `(n, m)` indexing and metres of OPD.

### `ChromaticZernikePhase`

Callable position--frequency phase in radians:

```python
ChromaticZernikePhase(
    pupil_radius,
    coefficients,
    carrier_angular_frequency=omega_c,
    centre=(0, 0, 0),
    axis_u=(1, 0, 0),
    axis_v=(0, 1, 0),
    outside="raise",
)
```

`coefficients` maps an OSA/ANSI `(n, m)` mode to either a finite constant in
radians or a callable of angular frequency returning radians.

Paper-specific constructors:

- `jolly_angular_dispersion(...)`
- `jolly_chromatic_curvature(...)`
- `jolly_chromatic_trefoil(...)`

All three-dimensional incident fields accept
`spatio_spectral_phase(points, angular_frequency)` in addition to
`wavefront_opd(points)`.

## Solvers

### `evaluate_SC_2D(...)`

Evaluates the 2D TM boundary integral at flattened observation coordinates.
`chunk_size=64` bounds the vectorised `(observation, surface)` temporaries.
See `main.py` for a complete monochromatic call.

### `evaluate_SC_3D(observation_points, surface, E_inc, B_inc, k, **options)`

Returns `(electric, magnetic)` at arbitrary observation points.

Options:

- `chunk_size=64`
- `contours=()`
- `B_inc_contours=()`
- `backend="numpy"` or `"cupy"`

Observation points must have shape `(n, 3)`. Observation points may not lie
on the integration surface.

### `electric_from_magnetic_tm(By, x, z, k, edge_order=2)`

Recovers the 2D electric components from a regularly sampled `By` grid with
shape `(len(x), len(z))`.

## Spectra and reconstruction

### `GaussianPulseSpectrum.from_intensity_fwhm(...)`

Builds a transform-limited Gaussian spectrum whose reconstructed intensity
envelope has the requested temporal FWHM. `minimum_period` selects an odd
component count whose discrete period exceeds a requested EPOCH file window.
The carrier is always an explicit central component.

### `SuperGaussianSpectrum.from_wavelength_bandwidth(...)`

Important arguments:

- `central_wavelength`
- `wavelength_fwhm`
- `spectral_order`
- `total_energy`
- `n_components`
- `span_fwhm`
- `conversion`
- `spectral_phase`

Conversion values:

- `"narrowband"`: Vallières-compatible;
- `"exact_wavelength_density"`: exact wavelength-density Jacobian.

### `SampledSpectrum`

Factories:

- `from_angular_frequency_samples(...)`
- `from_wavelength_samples(...)`

Shared spectrum methods and properties:

- `period`
- `nyquist_timestep`
- `envelope_nyquist_timestep(carrier_angular_frequency)`
- `with_spectral_phase(...)`
- `component_amplitudes(effective_area)`
- `component_coefficients(effective_area)`
- `recovered_energy(amplitudes, effective_area)`
- `validate_time_samples(times)`
- `validate_envelope_time_samples(times, carrier_angular_frequency)`

### `reconstruct_analytic_signal(components, angular_frequencies, times)`

`components` uses frequency as its first axis. The return has time as its
first axis.

### `reconstruct_complex_envelope(components, angular_frequencies, times, carrier_angular_frequency)`

Returns
\(2\sum_n E_n\exp[-i(\omega_n-\omega_c)t]\). Multiplication by
\(\exp(-i\omega_ct)\) recovers the analytic signal. Use this form for EPOCH
spatiotemporal amplitude/phase files.

### `electric_intensity(analytic_electric_field)`

Returns \(\epsilon_0c|\widetilde{\mathbf E}|^2/2\) in watts per square metre.

## Broadband propagation

### `propagate_broadband_2d(...)`

Returns `BroadbandPropagation2DResult` on a rectilinear `(x,z)` grid. Electric
shape is `(n_t,n_x,n_z,2)` for `(Ex,Ez)` and magnetic shape is
`(n_t,n_x,n_z)` for `By`. Important options include:

- `reference_point` and `reference_step`;
- `reference_normalisation="none"`, `"phase"`, or `"complex"`;
- `envelope_peak_time` with `carrier_angular_frequency`;
- `num_surface_points`, `solver_chunk_size`, and shared-memory `workers`.

### `generate_epoch2d_oap_pulse(directory, **options)`

Generates a complete normalised `(n_t,n_y)` EPOCH2D file pair and JSON
manifest. The high-level path defines its spectrum at the focus, predicts the
boundary arrival, maps the OAP90 result to `+x` injection from `x_min`, records
a direct carrier-focus reference, and regularises only phase below the selected
negligible-amplitude floor.

### `iter_broadband_field_chunks(...)`

Arguments:

- observation points;
- one fixed surface quadrature;
- incident-field object;
- spectrum object;
- requested times.

Important options:

- `effective_area=None`
- `propagation_phase=0`
- `carrier_angular_frequency=None`
- `observation_chunk_size=64`
- `contours=()`
- `communicator=None`
- `solver_options=None`

Yields `BroadbandFieldChunk(start, stop, electric, magnetic)`.

With the default carrier option, fields contain the full analytic signal.
Supplying a carrier returns its slowly varying complex envelope.

### `propagate_broadband_3d(...)`

Collects the local chunks and returns
`BroadbandPropagationResult(start, stop, electric, magnetic)`.

Under MPI, `start:stop` is the rank-local global observation range.

### `observation_partition(n_observations, rank=0, size=1)`

Returns the balanced contiguous `slice` assigned to one rank.

## Convergence

### `surface_quadrature_convergence(...)`

Refines a sequence of `(n_radial, n_azimuthal)` orders.

Returns `QuadratureConvergenceResult` containing:

- `levels`
- final `electric` and `magnetic`;
- `converged`.

Each `QuadratureConvergenceLevel` records electric, magnetic, and combined
relative changes.

## Diagnostics

### `maxwell_residuals(...)`

Input shape: `(nx, ny, nz, 3)`.

### `time_domain_maxwell_residuals(...)`

Input shape: `(nt, nx, ny, nz, 3)`.

Both return `MaxwellResiduals` with:

- `divergence_e`
- `divergence_b`
- `faraday`
- `ampere`

### `electromagnetic_energy_density(E, B, cycle_averaged=True)`

Returns energy density with the vector component on the final axis.

### `integrated_field_energy(E, B, coordinates, cycle_averaged=True)`

Integrates over a rectilinear volume.

### `integrated_poynting_flux(E, B, coordinates, normal, cycle_averaged=True)`

Integrates signed flux through a rectilinear plane.

### `relative_energy_error(reference_energy, measured_energy)`

Returns `measured/reference - 1`.

## EPOCH export

### `epoch_amplitude_phase(field, field_scale=None, phase_reference=0, unwrap_phase=True, phase_amplitude_floor=None)`

Returns `(amplitude, phase, field_scale)` without writing files.

Phase is unwrapped sequentially along every array axis by default so EPOCH's
linear interpolation does not cross artificial \(2\pi\) branch cuts.
When `phase_amplitude_floor` is set, samples below that fraction of peak
amplitude inherit the nearest reliable phase before unwrapping. Reliable field
samples are unchanged modulo (2\pi). The campaign-level 2D generator defaults
to a field-amplitude floor of (10^{-3}), corresponding to (10^{-6}) of peak
intensity; the lower-level exporter remains opt-in.

### `epoch_phase_diagnostics(field, amplitude_floor=1e-6)`

Returns `EpochPhaseDiagnostics`:

- `low_amplitude_fraction`
- `maximum_reliable_phase_step`
- `winding_cell_count`
- `has_phase_singularity`

Phase steps and winding cells touching samples below the relative amplitude
floor are excluded. A nonzero winding count indicates a topological phase
singularity that ordinary unwrapping cannot remove.

### `export_epoch_profile(directory, field, **options)`

Writes headerless native `float64` amplitude and phase streams and returns
`EpochProfileExport`:

- `amplitude_file`
- `phase_file`
- `shape`
- `field_scale`

Options:

- `field_scale`
- `phase_reference`
- `unwrap_phase`
- `amplitude_filename`
- `phase_filename`

`export_field_binary()` is a compatibility wrapper. `export_all_fields()`
intentionally rejects the obsolete volume-field interface.

Static exports take a monochromatic phasor. Time-dependent exports take a
complex envelope relative to the EPOCH carrier, not a full analytic signal.
