# Literature and existing software

Review date: 15 July 2026.

## Direct implementations

### StrattoCalculator (Dumont *et al.*, 2017; Vallières *et al.*, 2023)

Dumont *et al.* describe a C++ implementation parallelised with OpenMPI and
using HDF5. It evaluates the physical-optics Stratton--Chu surface and contour
integrals frequency by frequency for arbitrary spatial and temporal incident
fields. Vallières *et al.* used its successor, called **StrattoCalculator**, for
the OAP/HNAP/TP study on which this package is based. More recent work still
describes it as an in-house code able to handle complex mirrors and input
beams. No public source repository or software licence was located in the
papers, their supplementary material, arXiv records, or targeted repository
searches. It is therefore the closest prior implementation scientifically,
but not currently a reusable public dependency.

- Dumont *et al.*, *Journal of Optics* 19, 025604 (2017),
  <https://arxiv.org/abs/1609.08146>
- Vallières *et al.*, *Optics Express* 31, 19319 (2023),
  <https://doi.org/10.1364/OE.486230>
- Jolly *et al.*, *Nanophotonics* (2025),
  <https://doi.org/10.1515/nanoph-2024-0616>

### Fourmaux transmission-parabola experiment (2025)

Fourmaux *et al.* provide the first experimental case in this review that is
directly actionable as a geometry preset. They used the same Stratton--Chu
lineage to infer focal fields from a measured wavefront for an NA = 0.96
transmission parabola: 5.65 mm parent focal length, 65 mm illuminated diameter,
and 24.5 mm central opening. A second identical TP collected the focused light
for wavefront sensing relative to a pinhole reference. Deformable-mirror
correction improved the measured wavefront from 9.3 wavelengths peak-to-valley
and 2.13 wavelengths RMS to 1.02 and 0.16 wavelengths, while the calculated
peak rose from 6% to 68.1% of the ideal field.

This strongly supports SCPIC's measured-OPD and Zernike workflow and shows that
ideal-mirror intensity alone is not an adequate experimental prediction.
However, the phase maps and Zernike coefficients are not publicly available;
the paper states that underlying data may be requested from the authors. The
published geometry is therefore a regression target, whereas the aberrated
intensity maps are not yet independently reproducible.

- Fourmaux *et al.*, *Optics Letters* 50, 7027 (2025),
  <https://doi.org/10.1364/OL.576854>

### Nielsen's GPU laser--electron package (2022)

C. F. Nielsen published an MIT-licensed C++/CUDA package alongside a Monte
Carlo laser--electron interaction code. Its archive explicitly includes a
separate solver for the Stratton--Chu vector diffraction integrals from an
incident beam on a focusing mirror. This is the clearest public code base to
inspect for performance design, GPU batching and independent numerical
cross-checks. A source-level review confirmed that its electric integrand is
algebraically equivalent to SCPIC's general vector expression for a centred
paraboloid and real linear input. The released execution path nevertheless
hard-codes a monochromatic profile, leaves offset and Zernike paths disabled,
omits the observation-z term in two transverse magnetic component formulas,
and has generator/loader header and grid-spacing mismatches. It is not an
EPOCH profile generator and is therefore treated as a reference for corrected
component tests and GPU design rather than a dependency. No Nielsen source was
copied into SCPIC.

The dataset metadata and paper state MIT licensing, but the downloaded ZIP
does not contain a licence text. Any future direct code reuse should retain
clear provenance and the upstream permission notice rather than relying only
on the archive metadata.

- Code and data archive: <https://doi.org/10.17632/bd5m7tf5yr.1>

### Bulanov *et al.* library (2025)

Bulanov *et al.* report a programming library based on the same
physical-optics Stratton--Chu integrals, aimed at coherent multi-beam dipole
focusing with mirror placement, phase distortion and aberrations. The paper
contains enough algorithmic detail to be relevant to a future arbitrary-mirror
API, but no public source repository or licence was located.

- Paper and source text: <https://arxiv.org/abs/2412.07424>
- Published article: <https://doi.org/10.1364/AO.543161>

### Popov's 3D PIC code SCPIC (2009)

Konstantin Popov's PhD thesis describes a C/OpenMP 3D electromagnetic PIC code,
also named **SCPIC**, whose injected parabolic-mirror fields were calculated
with Stratton--Chu integrals. This predates the present project and the
Dumont/Vallières implementation. It appears to be a complete PIC code rather
than an upstream optical preprocessor, and no maintained public source release
was located. The name collision is worth documenting if this package is later
published or indexed.

- Thesis: <https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/AEU/TC-AEU-29914.pdf>

## Related but different approaches

There are several public or published approaches which create Maxwell-valid
tightly focused fields for PIC injection but do not propagate measured fields
from a reflector surface with the Vallières method. These include
Richards--Wolf/Debye pupil integrals, spectral boundary-injection algorithms,
and analytic complex-source or Lax-series beams. They may be useful for
cross-validation, but replacing the Stratton--Chu surface integral with one of
them would change this project's stated physical model.

## Conclusion

Yes: the methodology has been implemented several times. StrattoCalculator is
the direct scientific predecessor, Nielsen's 2022 archive is the most useful
public implementation found, and the Bulanov library is the most recent close
parallel. The present package still occupies a useful niche: a small,
inspectable Python implementation tied directly to EPOCH's custom injection
format, with an explicit 2D reduction, a full 3D vector path, and regression
benchmarks against Vallières *et al.*
The Fourmaux experiment additionally supplies a realistic TP geometry and a
clear measured-wavefront use case, while reinforcing that experimental phase
data must accompany peak-intensity claims.
