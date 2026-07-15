# Three-dimensional methodology

## Scope

The 3D path implements the linearly and radially polarised,
perfect-conductor physical-optics method used by Vallières *et al.* (2023),
building on Dumont *et al.* (2017).
It accepts an upstream collimated super-Gaussian beam, evaluates the reflected
vector field on an arbitrary observation grid, and reconstructs an
energy-normalised broadband pulse. The original 2D TM solver remains available
as a separate reduced model.

The implemented reflector is the parent paraboloid

\[
z = \frac{x^2+y^2}{4 f_0}-f_0,
\]

with the focus at the origin. `HNAP` and `TP` apertures are centred on the
parent axis; `TP` may be annular. The `OAP90` aperture is centred at
\((2f_0,0)\), so a chief ray travelling along \(-z\) reflects along \(-x\).
The Vallières OAP has \(f_0=57.5\) mm and \(D=220\) mm, giving the reported
115 mm effective focal length and 5 mm nearest approach to the parent axis.
The HNAP uses \(f_0=58\) mm. For the optimal 20% TP, Eqs. (6)--(7) give
\(D_{\mathrm{in}}=200\sqrt{0.2}=89.44\) mm and \(f_0=20\) mm for a 5 mm focus
margin. These are available through the three `vallieres_*` mirror factories.

## Stratton--Chu fields

SCPIC uses

\[
\mathrm{physical\ field}=\Re\left[\mathbf F(\mathbf r)
e^{-i\omega t}\right]
\]

and \(G=e^{ikR}/R\). The reflected surface fields are

\[
\mathbf E = \frac{1}{2\pi}\int_S
\left[ikc(\mathbf n\times\mathbf B_i)G
+(\mathbf n\cdot\mathbf E_i)\nabla_SG\right]dS,
\]

\[
\mathbf B = \frac{1}{2\pi}\int_S
(\mathbf n\times\mathbf B_i)\times\nabla_SG\,dS.
\]

The explicit \(c\) in the first equation restores SI dimensions; the papers
write the derivation in units where \(c=1\). `evaluate_SC_3D()` also supports
the electric-field rim term from Dumont *et al.* when oriented contour
quadratures and their incident magnetic fields are supplied. Vallières *et
al.* omitted this term after checking it was negligible for their geometry.

Surface integration uses Gauss--Legendre nodes in projected aperture radius
and azimuth. `projected_weights` integrate the incident plane, while `weights`
include the paraboloid surface Jacobian. The phase is unusually well behaved
near a parabolic focus because \(|\mathbf r_S|-z_S=2f_0\); convergence must
still be checked whenever the geometry, aberrations or observation region
changes.

## Incident polarisation, wavefront and energy

`LinearPolarisedSuperGaussian3D` implements

\[
\mathbf E_i=E_{0n}\exp[-(r/w_0)^p]e^{-ikz}\hat{\mathbf x},
\qquad
\mathbf B_i=-\mathbf E_i\hat{\mathbf y}/c,
\]

with arbitrary orthogonal propagation and polarisation vectors. The paper's
intensity-FWHM diameter is converted through

\[
w_0=\frac{w_{\mathrm{FWHM}}}
{2(\ln2/2)^{1/p}}.
\]

`TM01RadiallyPolarisedBeam3D` implements the paper's Eqs. (16)--(17), written
below in the package coordinates for propagation along \(-z\):

\[
\mathbf E_{i,n}=\frac{2E_{0,n}}{k_nw_0^2}e^{-(r/w_0)^2-ik_nz}
\left[r\hat{\mathbf r}+\frac{2i}{k_n}
\left(\frac{r^2}{w_0^2}-1\right)\hat{\mathbf z}\right],
\qquad
\mathbf B_{i,n}=-\frac{E_{r,n}}{c}\hat{\boldsymbol\phi}.
\]

The implementation generalises this to any propagation direction. Its radial
field vanishes continuously on axis and the longitudinal component is
retained. The longitudinal-flux effective area relative to \(E_{0,n}\) is
\(A_{\mathrm{eff},n}=\pi/k_n^2\), so broadband component normalisation accepts
one effective area per frequency. `RadiallyPolarisedSuperGaussian3D` is kept
as a reduced transverse-envelope model, but must not be used to claim a
reproduction of the paper's TM01 cases.

All incident-field classes accept `wavefront_opd(points)`, expressed in
metres. A spectral component of wavenumber (k_n) receives the phase
(+k_n\,\mathrm{OPD}), so a single measured surface map remains physically
consistent across the bandwidth. `ZernikeWavefront` supplies orthonormal
OSA/ANSI modes on a user-defined pupil: coefficients are RMS OPD in metres,
positive azimuthal index uses cosine dependence, and negative index uses sine
dependence. An arbitrary callable can instead interpolate measured data.

`SuperGaussianSpectrum` samples the order-seven 90 nm FWHM spectrum around
800 nm. For uniform angular-frequency spacing, \(T=2\pi/\Delta\omega\), and
the component amplitudes are normalised so that

\[
E_L=T\sum_n\int 2\epsilon_0c|E_n|^2dA.
\]

The analytic signal is \(2\sum_n\mathbf E_n e^{-i\omega_nt}\), and its
instantaneous envelope intensity is
\(I=\epsilon_0c|\widetilde{\mathbf E}|^2/2\).

## Numerical backends and field diagnostics

NumPy direct quadrature is the reference implementation. Passing
`backend="cupy"` transfers the fixed surface data and one observation chunk at
a time to a CUDA device, then returns ordinary NumPy arrays. CuPy is an
optional, externally installed dependency because its package must match the
local CUDA version. The GPU path shares the same algebra and validation as the
reference path but still requires a runtime comparison on CUDA hardware.

`maxwell_residuals()` evaluates dimensionless RMS residuals of both divergence
and curl equations for monochromatic phasors on a regular Cartesian volume.
On the small HNAP regression grid, all four residuals are below 3%; this limit
includes second-order finite-difference and surface-quadrature error rather
than representing an exact analytic tolerance.

## Reproduction status

`paper_benchmark_3d.py --suite` reproduces all six cases in Tables 1--2. A
32×64 surface quadrature, 47 spectral components and 161-point profiles give:

| Input | Mirror | Peak (SCPIC/paper), 10²³ W/cm² | FWHM x (SCPIC/paper), µm | FWHM y (SCPIC/paper), µm | zR (SCPIC/paper), µm |
|---|---|---:|---:|---:|---:|
| Linear | HNAP | 5.230 / 5.03 | 0.620 / 0.62 | 0.331 / 0.33 | 0.395 / 0.39 |
| Linear | OAP90 | 2.772 / 2.66 | 0.606 / 0.60 | 0.521 / 0.52 | 0.692 / 0.69 |
| Linear | TP | 1.545 / 1.49 | 0.877 / 0.87 | 1.262 / 1.26 | 0.570 / 0.56 |
| TM01 | HNAP | 3.807 / 4.13 | 0.344 / 0.35 | 0.344 / 0.35 | 0.458 / 0.45 |
| TM01 | OAP90 | 1.018 / 1.07 | 0.673 / 0.69 | 0.663 / 0.68 | 0.526 / 0.53 |
| TM01 | TP | 2.247 / 2.45 | 0.378 / 0.37 | 0.378 / 0.37 | 0.558 / 0.56 |

Thus all spot sizes and Rayleigh lengths agree within 2.6%; linear peak
intensities agree within 4.3% and TM01 peaks within 8.3%. Independent
refinements were run for every case. Moving from 12×24 to 24×48 surface nodes
changes any reported quantity by at most 0.077%. Moving from 31 to 47 spectral
components changes peak intensity by 0.091% and every width by at most 0.003%.
The combined workstation-to-fine refinement (24×48/31/121 to
32×64/47/161) changes any result by at most 0.12%.

This is a close numerical reproduction, not bitwise identity with the private
StrattoCalculator. The residual 5--8% TM01 peak difference is not a numerical
convergence error at these settings and may reflect unpublished discretisation
or normalisation details. Measured-wavefront studies still require their own
map-interpolation and aperture convergence checks.
