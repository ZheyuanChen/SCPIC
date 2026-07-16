# Space-time couplings

## Scope

SCPIC can represent a non-separable incident phase

\[
\phi=\phi(\mathbf r,\omega)
\]

before propagating each spectral component with the three-dimensional
Stratton--Chu solver. This follows the integral-model route in Jolly *et al.*,
*Nanophotonics* 14, 815--832 (2025).

The current implementation supports arbitrary position--frequency **phase**.
It does not yet make beam amplitude, centre, width, direction, or polarisation
arbitrary functions of frequency.

## Fixed OPD versus chromatic phase

These two inputs have deliberately different units and meanings:

- `wavefront_opd(points)` returns optical path difference in metres. SCPIC
  applies \(k\,\mathrm{OPD}\) at each frequency.
- `spatio_spectral_phase(points, angular_frequency)` returns phase directly in
  radians.

A measured mirror or wavefront map normally belongs in `wavefront_opd`. A
frequency-dependent Zernike coefficient, pulse-front tilt, or chromatic
aberration normally belongs in `spatio_spectral_phase`.

Both may be present. Their phases are added.

## General chromatic Zernike model

`ChromaticZernikePhase` uses orthonormal OSA/ANSI Zernike modes:

```python
from scpic import ChromaticZernikePhase

phase = ChromaticZernikePhase(
    pupil_radius=50e-3,
    carrier_angular_frequency=omega_c,
    coefficients={
        (1, 1): lambda omega: a1 * (omega - omega_c),
        (2, 0): lambda omega: a2 * (omega - omega_c),
        (3, 3): lambda omega: a3 * (omega - omega_c) ** 2,
    },
    centre=(*mirror.aperture_centre, 0.0),
)
```

Each coefficient is either a constant phase in radians or a callable of
angular frequency returning radians. Positive azimuthal index uses cosine
orientation and negative index uses sine orientation.

Attach the model to any three-dimensional incident field:

```python
incident = TM01RadiallyPolarisedBeam3D(
    w0=beam_radius,
    wavelength=800e-9,
    centre=(*mirror.aperture_centre, 0.0),
    spatio_spectral_phase=phase,
)
```

The same interface is available on:

- `LinearPolarisedSuperGaussian3D`;
- `RadiallyPolarisedSuperGaussian3D`;
- `TM01RadiallyPolarisedBeam3D`;
- `FiniteRayleighTM01Beam3D`;
- `ParaxialGaussLaguerreBeam3D`.

Broadband propagation evaluates the callable separately at every retained
frequency and on every mirror or contour point.

## Jolly et al. constructors

### Input angular dispersion

Jolly's equation (44) is:

\[
\phi(x,\omega)=
\tau_t(\omega-\omega_c)\frac{x}{w_i}\frac{\omega}{\omega_c}.
\]

For x-oriented tilt:

```python
angular_dispersion = ChromaticZernikePhase.jolly_angular_dispersion(
    pupil_radius=beam_radius,
    pulse_front_tilt=4.25e-15,
    carrier_angular_frequency=omega_c,
    centre=(*mirror.aperture_centre, 0.0),
    azimuthal_index=1,
)
```

`pulse_front_tilt` is the group-delay difference in seconds at one pupil
radius. Use `azimuthal_index=-1` for the orthogonal sine orientation.

The integral solver naturally converts this input tilt into frequency-dependent
transverse displacement, component mixing, coma-like distortions, and other
finite-aperture effects at focus.

### Chromatic curvature

Jolly's equation (45) is:

\[
\phi(r,\omega)=
\tau_p(\omega-\omega_c)\frac{\omega}{\omega_c}\frac{r^2}{w_i^2}.
\]

Use:

```python
chromatic_curvature = ChromaticZernikePhase.jolly_chromatic_curvature(
    pupil_radius=beam_radius,
    pulse_front_curvature=4.25e-15,
    carrier_angular_frequency=omega_c,
    centre=(*mirror.aperture_centre, 0.0),
)
```

The constructor retains the piston term used in the paper so the phase is zero
on axis and the group delay reaches `pulse_front_curvature` at one pupil
radius. Set `include_piston=False` only when the corresponding frequency-
dependent global delay is intentionally handled elsewhere.

### Chromatic trefoil

Jolly's equation (46) is available through:

```python
trefoil = ChromaticZernikePhase.jolly_chromatic_trefoil(
    pupil_radius=beam_radius,
    characteristic_delay=8.5e-15,
    carrier_angular_frequency=omega_c,
    centre=(*mirror.aperture_centre, 0.0),
    azimuthal_index=3,
)
```

Use `azimuthal_index=-3` for the sine-oriented trefoil.

## Composing effects

For arbitrary combinations, place all modes in one
`ChromaticZernikePhase`. Alternatively, combine callables explicitly:

```python
def combined_phase(points, omega):
    return (
        angular_dispersion(points, omega)
        + chromatic_curvature(points, omega)
        + measured_chromatic_phase(points, omega)
    )
```

Do not add the same effect both as OPD and as a chromatic phase.

## Validation

For every STC campaign:

1. Confirm that the phase at \(\omega_c\) has the intended reference.
2. Numerically differentiate phase with frequency and inspect the pupil
   group-delay map.
3. Plot individual spectral components before reconstructing time.
4. Converge frequency count, frequency extent, mirror quadrature, focal
   sampling, and time sampling.
5. Inspect all vector components. Nonparaxial STCs can affect them
   differently.
6. Compare the time-dependent field over several longitudinal planes rather
   than only at nominal focus.

The automated suite verifies the exact group-delay scaling of Jolly's
angular-dispersion and chromatic-curvature phases, trefoil symmetry, use by
every incident-field class, and evaluation at every broadband frequency.

## EPOCH phase topology

Space-time vortices and other field zeros can have nonzero phase circulation.
No globally continuous scalar phase exists around such a singularity.
Sequential unwrapping cannot remove this topological obstruction.

Before exporting a strongly coupled field, run:

```python
from scpic import epoch_phase_diagnostics

diagnostics = epoch_phase_diagnostics(
    envelope,
    amplitude_floor=1e-6,
)

if diagnostics.has_phase_singularity:
    raise RuntimeError(
        "The EPOCH amplitude/phase reader needs an explicit branch-cut study"
    )
```

The diagnostic reports:

- the fraction of samples below the chosen reliable-amplitude threshold;
- the largest wrapped adjacent phase step between reliable samples;
- the number of reliable two-dimensional cells carrying phase winding.

A nonzero winding count is a warning about the amplitude/phase storage
representation, not an error in the optical field. A future real/imaginary
quadrature reader or full-vector interior injector is the safer representation
for such fields.

## Present limitations

- Frequency-dependent amplitude and spectral filtering are not first-class.
- Beam centre, width, propagation direction, and polarisation remain fixed
  properties of an incident-field object.
- The convenience constructors reproduce the input phases in Jolly's
  equations (44)--(46); they do not reproduce every plotted result without
  matching the paper's mirror, spectrum, beam, grids, and normalisation.
- The local EPOCH smoke tests do not yet include a chromatic STC case.
- `simple_laser` still cannot impose the complete longitudinal field or both
  tangential electric and magnetic data.
