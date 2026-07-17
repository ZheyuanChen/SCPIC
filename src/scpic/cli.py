"""Command-line entry points for reproducible SCPIC profile generation."""

import argparse

from .profile2d import generate_epoch2d_oap_pulse


def _epoch2d_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a frequency-resolved 2D TM OAP profile for the "
            "EPOCH-mod spatiotemporal reader."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--wavelength-um", type=float, default=0.8)
    parser.add_argument("--tau-fwhm-fs", type=float, default=30.0)
    parser.add_argument("--focus-distance-um", type=float, default=24.0)
    parser.add_argument(
        "--boundary-peak-time-fs", type=float, default=143.31487238414786
    )
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--effective-focal-length-mm", type=float, default=50.8)
    parser.add_argument("--aperture-radius-waists", type=float, default=3.0)
    parser.add_argument("--time-start-fs", type=float, default=0.0)
    parser.add_argument("--time-end-fs", type=float, default=400.0)
    parser.add_argument("--n-time", type=int, default=801)
    parser.add_argument("--transverse-min-um", type=float, default=-20.0)
    parser.add_argument("--transverse-max-um", type=float, default=20.0)
    parser.add_argument("--n-transverse", type=int, default=1601)
    parser.add_argument("--n-surface", type=int, default=6000)
    parser.add_argument("--n-components", type=int)
    parser.add_argument("--spectrum-span-fwhm", type=float, default=2.0)
    parser.add_argument("--derivative-step-nm", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-chunk-size", type=int, default=64)
    parser.add_argument("--phase-reference", type=float, default=0.0)
    parser.add_argument("--phase-amplitude-floor", type=float, default=1e-3)
    parser.add_argument("--file-prefix", default="laser")
    return parser


def generate_epoch2d_main(argv=None):
    """Run the ``scpic-generate-epoch2d`` command."""
    args = _epoch2d_parser().parse_args(argv)
    generated = generate_epoch2d_oap_pulse(
        args.output,
        central_wavelength=args.wavelength_um * 1e-6,
        intensity_fwhm=args.tau_fwhm_fs * 1e-15,
        focus_distance=args.focus_distance_um * 1e-6,
        boundary_peak_time=args.boundary_peak_time_fs * 1e-15,
        f_number=args.f_number,
        effective_focal_length=args.effective_focal_length_mm * 1e-3,
        aperture_radius_in_beam_waists=args.aperture_radius_waists,
        time_start=args.time_start_fs * 1e-15,
        time_end=args.time_end_fs * 1e-15,
        n_time=args.n_time,
        transverse_min=args.transverse_min_um * 1e-6,
        transverse_max=args.transverse_max_um * 1e-6,
        n_transverse=args.n_transverse,
        n_surface=args.n_surface,
        n_components=args.n_components,
        spectrum_span_fwhm=args.spectrum_span_fwhm,
        derivative_step=args.derivative_step_nm * 1e-9,
        workers=args.workers,
        solver_chunk_size=args.solver_chunk_size,
        phase_reference=args.phase_reference,
        phase_amplitude_floor=args.phase_amplitude_floor,
        file_prefix=args.file_prefix,
    )
    print(f"Wrote {generated.amplitude_file}")
    print(f"Wrote {generated.phase_file}")
    print(f"Wrote {generated.manifest_file}")
    print(f"Generation wall time: {generated.generation_seconds:.3f} s")


if __name__ == "__main__":
    generate_epoch2d_main()
