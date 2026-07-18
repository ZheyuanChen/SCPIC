"""Generate the editable 2D f/2 SCPIC optical-workflow illustration.

Run from the repository root with::

    .venv/bin/python docs/figures/generate_2d_f2_workflow.py

The two optical panels deliberately use different length scales: the mirror
geometry is millimetre-scale, whereas EPOCH begins only micrometres before the
focus.  All plotted dimensions are derived from the command-line parameters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

matplotlib.rcParams["svg.hashsalt"] = "scpic-2d-f2-workflow"

BLUE = "#2468a2"
LIGHT_BLUE = "#8fc6e8"
ORANGE = "#d97706"
GREEN = "#238636"
DARK = "#25313c"
GREY = "#6b7280"


def _arrow(ax, start, end, *, colour=BLUE, width=1.6, mutation_scale=11, alpha=1):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=width,
        color=colour,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def _physical_geometry(
    ax, *, f_number, effective_focal_length_mm, aperture_radius_waists
):
    f0 = effective_focal_length_mm / 2
    incident_radius = effective_focal_length_mm / (2 * f_number)
    diameter = 2 * aperture_radius_waists * incident_radius
    centre = 2 * f0

    x_mirror = np.linspace(centre - diameter / 2, centre + diameter / 2, 800)
    z_mirror = x_mirror**2 / (4 * f0) - f0
    ax.plot(x_mirror, z_mirror, color=DARK, linewidth=3, label="Parabolic cylinder")

    ray_offsets = incident_radius * np.linspace(-2.2, 2.2, 9)
    z_start = float(np.max(z_mirror) + 12)
    for offset in ray_offsets:
        x_ray = centre + offset
        z_hit = x_ray**2 / (4 * f0) - f0
        weight = float(np.exp(-((offset / incident_radius) ** 2)))
        alpha = 0.12 + 0.78 * weight
        _arrow(
            ax,
            (x_ray, z_start),
            (x_ray, z_hit + 0.4),
            colour=BLUE,
            width=0.7 + 1.5 * weight,
            mutation_scale=7 + 5 * weight,
            alpha=alpha,
        )
        _arrow(
            ax,
            (x_ray, z_hit),
            (0.5, 0.0),
            colour=ORANGE,
            width=0.7 + 1.5 * weight,
            mutation_scale=7 + 5 * weight,
            alpha=alpha,
        )

    ax.scatter([centre], [0], s=42, color=DARK, zorder=5)
    ax.annotate(
        f"90° point\n({centre:.1f} mm, 0)",
        (centre, 0),
        xytext=(centre + 5, -11),
        arrowprops={"arrowstyle": "-", "color": GREY},
        fontsize=9,
    )
    ax.scatter([0], [0], marker="*", s=145, color=GREEN, zorder=6)
    ax.annotate("Focus (0, 0)", (0, 0), xytext=(4, 7), fontsize=9, color=GREEN)

    y_dimension = z_start + 5
    ax.annotate(
        "",
        xy=(centre - incident_radius, y_dimension),
        xytext=(centre + incident_radius, y_dimension),
        arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 1.5},
    )
    ax.text(
        centre,
        y_dimension + 2,
        rf"$2w_{{\rm in}}={2 * incident_radius:.1f}$ mm  ($f/{f_number:g}$ illumination)",
        ha="center",
        va="bottom",
        color=BLUE,
        fontsize=9,
    )
    ax.text(
        centre - 0.5,
        z_start + 1.5,
        r"Collimated Gaussian sheet beam  $\propto e^{-(x-x_c)^2/w_{\rm in}^2}$",
        ha="center",
        va="bottom",
        color=BLUE,
        fontsize=9,
    )

    ax.set_xlabel(r"SCPIC $x$ (mm)")
    ax.set_ylabel(r"SCPIC $z$ (mm)")
    ax.set_title("Upstream reflection geometry (millimetre scale)", loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.set_xlim(min(-9, x_mirror.min() - 5), x_mirror.max() + 7)
    ax.set_ylim(z_mirror.min() - 8, y_dimension + 8)


def _near_focus(
    ax,
    *,
    wavelength_um,
    f_number,
    focus_distance_um,
):
    waist = 2 * wavelength_um * f_number / np.pi
    rayleigh_length = np.pi * waist**2 / wavelength_um
    x_epoch = np.linspace(0, focus_distance_um + 3.5, 500)
    relative = x_epoch - focus_distance_um
    width = waist * np.sqrt(1 + (relative / rayleigh_length) ** 2)

    ax.fill_between(x_epoch, -width, width, color=LIGHT_BLUE, alpha=0.34)
    ax.plot(x_epoch, width, color=BLUE, linewidth=1.8)
    ax.plot(x_epoch, -width, color=BLUE, linewidth=1.8)
    ax.axvline(0, color=ORANGE, linewidth=2)
    ax.axvline(focus_distance_um, color=GREEN, linewidth=2)
    _arrow(
        ax,
        (2.0, 0),
        (focus_distance_um - 1.5, 0),
        colour=ORANGE,
        width=2,
        mutation_scale=13,
    )

    boundary_width = waist * np.sqrt(1 + (focus_distance_um / rayleigh_length) ** 2)
    ax.text(
        0.65,
        boundary_width + 0.35,
        f"EPOCH $x_{{min}}$\nSCPIC samples $E_z(t,z)$\n$w_{{par}}={boundary_width:.2f}$ µm",
        ha="left",
        va="bottom",
        fontsize=9,
        color=ORANGE,
    )
    ax.text(
        focus_distance_um + 0.45,
        1.7,
        f"Focus\n$w_{{par}}={waist:.3f}$ µm",
        ha="left",
        va="center",
        fontsize=9,
        color=GREEN,
    )
    ax.text(
        focus_distance_um / 2,
        -7.1,
        rf"$x_{{\rm EPOCH}}={focus_distance_um:g}\,\mu{{\rm m}}-x_{{\rm SCPIC}}$",
        ha="center",
        color=DARK,
        fontsize=10,
    )

    ax.set_xlim(-0.8, focus_distance_um + 6)
    ax.set_ylim(-8, 8)
    ax.set_xlabel(r"EPOCH $x$ (µm), propagation towards $+x$")
    ax.set_ylabel("EPOCH y = SCPIC z (µm)")
    ax.set_title(
        "Injection plane and near-focus propagation (micrometre scale)", loc="left"
    )
    ax.grid(alpha=0.18)

    top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda value: focus_distance_um - value,
            lambda value: focus_distance_um - value,
        ),
    )
    top.set_xlabel(r"SCPIC $x$ (µm), propagation towards $-x$")


def _workflow(ax):
    ax.set_axis_off()
    labels = (
        "Gaussian pupil\nB_y^inc(ω)",
        "Parabolic-cylinder\nsurface samples",
        "2D Green integral\nB_y(x,z,ω)",
        "Maxwell recovery\n$(E_x,E_z)$",
        "Complex focal\nnormalisation",
        "Boundary envelope\n$E_z(t,z)$",
        "EPOCH files\nA, π/2 − arg(E_z)",
    )
    left_margin = 0.015
    gap = 0.018
    width = (1 - 2 * left_margin - gap * (len(labels) - 1)) / len(labels)
    y0, height = 0.23, 0.52

    for index, label in enumerate(labels):
        x0 = left_margin + index * (width + gap)
        box = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            transform=ax.transAxes,
            facecolor="#f3f7fa" if index % 2 == 0 else "#fff7ed",
            edgecolor=GREY,
            linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(
            x0 + width / 2,
            y0 + height / 2,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.7,
            color=DARK,
        )
        if index < len(labels) - 1:
            _arrow(
                ax,
                (x0 + width + 0.002, y0 + height / 2),
                (x0 + width + gap - 0.002, y0 + height / 2),
                colour=GREY,
                width=1.2,
                mutation_scale=9,
            )
    ax.text(
        0.015, 0.92, "Frequency-resolved code path", transform=ax.transAxes, fontsize=11
    )


def build_figure(args):
    """Build the two-scale optical diagram and frequency-domain code path."""
    figure = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(4.6, 1.25))
    _physical_geometry(
        figure.add_subplot(grid[0, 0]),
        f_number=args.f_number,
        effective_focal_length_mm=args.effective_focal_length_mm,
        aperture_radius_waists=args.aperture_radius_waists,
    )
    _near_focus(
        figure.add_subplot(grid[0, 1]),
        wavelength_um=args.wavelength_um,
        f_number=args.f_number,
        focus_distance_um=args.focus_distance_um,
    )
    _workflow(figure.add_subplot(grid[1, :]))
    figure.suptitle(
        f"SCPIC 2D TM f/{args.f_number:g} profile workflow",
        fontsize=15,
        color=DARK,
    )
    return figure


def parse_args():
    """Return command-line settings for the optical geometry and output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/2d_f2_workflow.svg"),
    )
    parser.add_argument("--wavelength-um", type=float, default=0.8)
    parser.add_argument("--f-number", type=float, default=2.0)
    parser.add_argument("--effective-focal-length-mm", type=float, default=50.8)
    parser.add_argument("--focus-distance-um", type=float, default=24.0)
    parser.add_argument("--aperture-radius-waists", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    """Validate settings and write a deterministic SVG or raster figure."""
    args = parse_args()
    positive = (
        args.wavelength_um,
        args.f_number,
        args.effective_focal_length_mm,
        args.focus_distance_um,
        args.aperture_radius_waists,
        args.dpi,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("All optical parameters and --dpi must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = build_figure(args)
    metadata = {"Date": None} if args.output.suffix.lower() == ".svg" else None
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        metadata=metadata,
    )
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
