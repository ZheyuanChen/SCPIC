# Three-dimensional methodology

## Scope

The 3D path implements the linearly polarised, perfect-conductor physical-optics
method used by Vallières *et al.* (2023), building on Dumont *et al.* (2017).
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

`RadiallyPolarisedSuperGaussian3D` replaces the fixed polarisation vector by
the local transverse radial unit vector. Its value is set to zero at the
single undefined point on the beam axis, which has zero measure in the
surface and energy integrals.

Both incident-field classes accept `wavefront_opd(points)`, expressed in
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

`paper_benchmark_3d.py` reproduces Table 1's linearly polarised OAP90 case.
With the workstation defaults it gives:

| Quantity | SCPIC | Vallières *et al.* | Difference |
|---|---:|---:|---:|
| Peak intensity | 2.663 × 10²³ W/cm² | 2.66 × 10²³ W/cm² | +0.1% |
| Meridional FWHM | 0.600 µm | 0.600 µm | +0.02% |
| Sagittal FWHM | 0.503 µm | 0.520 µm | −3.4% |
| Rayleigh length | 0.650 µm | 0.690 µm | −5.8% |

These are strong checks of the geometry, phase, vector integral and pulse
normalisation, but they are not a complete reproduction of the paper. Radial
polarisation and arbitrary measured/Zernike wavefronts are now supported, but
the paper's full radial-polarisation and HNAP/TP numerical comparisons remain
to be run. Production results should include explicit convergence in surface
quadrature, spectral sampling, observation-grid spacing and, when relevant,
the interpolation of measured wavefront data.
