"""Export social-ready Matplotlib figures from reviewed loom measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
import numpy as np


WIDTH = 1600
HEIGHT = 1600
DPI = 125

BG = "#F7F8FA"
SURFACE = "#FFFFFF"
INK = "#101828"
MUTED = "#667085"
GRID = "#E4E7EC"
BLUE = "#155EEF"
BLUE_DARK = "#0B4AAF"
BLUE_LIGHT = "#84ADFF"
GOLD = "#F4B400"
ORANGE = "#F79009"
OLIVE = "#6B8E23"
GREY = "#98A2B3"
GREY_LIGHT = "#D0D5DD"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": BG,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": BG,
        }
    )


def _figure() -> plt.Figure:
    return plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=BG)


def _header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.955, "LOOM  /  TANG NANO 20K  /  INT8 @ 48 MHz", fontsize=11, color=BLUE_DARK, weight="bold")
    fig.text(0.06, 0.905, title, fontsize=25, color=INK, weight="bold")
    fig.text(0.06, 0.866, subtitle, fontsize=12, color=MUTED, linespacing=1.45)


def _footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.06, 0.035, text, fontsize=10, color=MUTED)


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BG, edgecolor="none", bbox_inches=None, metadata={"Software": "loom Matplotlib exporter"})
    plt.close(fig)


def throughput_boundaries(artifact: dict[str, Any], path: Path) -> None:
    summary = artifact["snapshot"]["datasets"]["hardware_summary"][0]
    labels = ["Theoretical peak", "Array active", "CFU + DMA window", "End-to-end"]
    values = [
        summary["theoretical_peak_gops"],
        summary["array_active_gops"],
        summary["accelerator_window_gops"],
        summary["achieved_issued_gops"],
    ]
    colors = [INK, BLUE, GOLD, ORANGE]

    fig = _figure()
    _header(
        fig,
        "MobileNet throughput by measurement boundary",
        "The array is efficient while active; complete inference is dominated by time outside CFU and DMA.",
    )
    ax = fig.add_axes([0.16, 0.38, 0.77, 0.37])
    y = list(reversed(range(len(labels))))
    ax.barh(y, values, height=0.56, color=colors, edgecolor="none")
    ax.set_yticks(y, labels, fontsize=13)
    ax.set_xlim(0, 6.75)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("GOPS  ·  multiply and add counted separately", labelpad=12)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=15)
    ax.tick_params(axis="x", length=0)
    for yy, value in zip(y, values):
        label = f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
        ax.text(max(value + 0.08, 0.12), yy, label, va="center", ha="left", fontsize=13, weight="bold", color=INK)

    fig.add_artist(Rectangle((0.06, 0.16), 0.41, 0.12, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
    fig.add_artist(Rectangle((0.53, 0.16), 0.41, 0.12, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
    fig.text(0.085, 0.225, "74.6%", fontsize=28, color=BLUE, weight="bold")
    fig.text(0.085, 0.180, "of compute peak while the array is active", fontsize=11, color=MUTED)
    fig.text(0.555, 0.225, "4.75%", fontsize=28, color=ORANGE, weight="bold")
    fig.text(0.555, 0.180, "of inference cycles touch CFU or DMA", fontsize=11, color=MUTED)
    _footer(fig, "Source: counter-enabled compiled MobileNet-v1 0.25 run · 48,993,503 cycles")
    _finish(fig, path)


def cycle_breakdown(artifact: dict[str, Any], path: Path) -> None:
    rows = artifact["snapshot"]["datasets"]["model_cycle_breakdown"]
    total = sum(int(row["cycles"]) for row in rows)
    palette = {
        "Outside CFU/DMA": GREY_LIGHT,
        "DMA only": GOLD,
        "Sequencer only": BLUE,
        "DMA + sequencer overlap": OLIVE,
    }

    fig = _figure()
    _header(
        fig,
        "Where one MobileNet inference spends 49.0M cycles",
        "Disjoint hardware-counter phases; segment widths use the full measured inference as denominator.",
    )
    ax = fig.add_axes([0.06, 0.58, 0.88, 0.16])
    left = 0.0
    for row in rows:
        pct = 100.0 * row["cycles"] / total
        ax.barh([0], [pct], left=[left], height=0.55, color=palette[row["phase"]], edgecolor=BG, linewidth=2)
        left += pct
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, 0.8)
    ax.set_yticks([])
    ax.set_xlabel("Share of complete inference cycles", labelpad=12)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
    outside_pct = 100.0 * rows[0]["cycles"] / total
    ax.text(outside_pct / 2, 0, f"Outside CFU/DMA  {outside_pct:.2f}%", ha="center", va="center", fontsize=14, weight="bold")

    box_positions = [(0.06, 0.40), (0.53, 0.40), (0.06, 0.20), (0.53, 0.20)]
    for (x, y), row in zip(box_positions, rows):
        pct = 100.0 * row["cycles"] / total
        fig.add_artist(Rectangle((x, y), 0.41, 0.14, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
        fig.add_artist(Rectangle((x + 0.025, y + 0.095), 0.014, 0.014, transform=fig.transFigure, facecolor=palette[row["phase"]], edgecolor="none"))
        fig.text(x + 0.050, y + 0.102, row["phase"], fontsize=11, color=MUTED, va="center")
        fig.text(x + 0.025, y + 0.045, f"{pct:.2f}%", fontsize=22, color=INK, weight="bold")
        fig.text(x + 0.235, y + 0.052, f"{row['cycles'] / 1e6:.3f}M cycles", fontsize=11, color=MUTED)

    _footer(fig, "Source: CFU sequencer, DMA-active, overlap counters, and RISC-V mcycle · one compiled-model run")
    _finish(fig, path)


def pure_gemm(artifact: dict[str, Any], path: Path) -> None:
    rows = {row["shape"]: row for row in artifact["snapshot"]["datasets"]["roofline"]}
    shapes = ["8x256x8", "64x256x64", "256x288x32", "16x1152x128"]
    display = [shape.replace("x", "×") for shape in shapes]
    array = [rows[shape]["array_active_gops"] for shape in shapes]
    window = [rows[shape]["accelerator_window_gops"] for shape in shapes]
    peak = artifact["snapshot"]["datasets"]["hardware_summary"][0]["theoretical_peak_gops"]

    fig = _figure()
    _header(
        fig,
        "Pure GEMM: compute is healthy, data movement is not",
        "Counter-derived throughput excludes UART; array-active and CFU+DMA use different measurement boundaries.",
    )
    ax = fig.add_axes([0.09, 0.23, 0.84, 0.55])
    x = list(range(len(shapes)))
    width = 0.31
    left = ax.bar([i - width / 2 for i in x], array, width=width, color=BLUE, label="Array active")
    right = ax.bar([i + width / 2 for i in x], window, width=width, color=GOLD, label="CFU + DMA window")
    ax.axhline(peak, color=INK, linewidth=1.7, linestyle=(0, (5, 4)), label=f"Compute peak  {peak:.3f} GOPS")
    ax.set_ylim(0, 6.8)
    ax.set_ylabel("GOPS")
    ax.set_xticks(x, display, fontsize=12)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper left", frameon=False, ncols=3, fontsize=11)
    for bars in (left, right):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.10,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                weight="bold",
            )
    _footer(fig, "Source: 12-shape verified GEMM sweep · every output exact within 1 INT8 LSB · streamed-KIR mcycle omitted")
    _finish(fig, path)


def placement(nextpnr_svg: Path, artifact: dict[str, Any], path: Path) -> None:
    """Compose nextpnr's native --placed-svg export on the square canvas."""
    if not nextpnr_svg.exists():
        raise FileNotFoundError(
            f"{nextpnr_svg} is missing; generate it with nextpnr-himbaechel --placed-svg"
        )
    with tempfile.NamedTemporaryFile(suffix=".png") as raster:
        subprocess.run(
            [
                "rsvg-convert",
                "--background-color",
                "white",
                "--width",
                "1800",
                "--keep-aspect-ratio",
                "--output",
                raster.name,
                str(nextpnr_svg),
            ],
            check=True,
        )
        native = plt.imread(raster.name)

    fig = _figure()
    _header(
        fig,
        "Native nextpnr placement",
        "Unmodified --placed-svg geometry from a seed-2 replay of the Tang Nano 20K Gowin design.",
    )
    ax = fig.add_axes([0.055, 0.25, 0.89, 0.56])
    ax.imshow(native)
    ax.set_axis_off()

    resources = artifact["snapshot"]["datasets"]["resource_utilization"]
    for x, row in zip([0.06, 0.29, 0.52, 0.75], resources):
        fig.add_artist(Rectangle((x, 0.13), 0.19, 0.075, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
        fig.text(x + 0.018, 0.175, row["resource"], fontsize=10, color=MUTED)
        fig.text(x + 0.018, 0.142, f"{row['utilization_pct']:.1f}%", fontsize=18, color=INK, weight="bold")
    _footer(fig, "Source: nextpnr-himbaechel --placed-svg · GW2AR-LV18QN88C8/I7 · placement seed 2")
    _finish(fig, path)


def _bel_xy(cell: dict[str, Any]) -> tuple[float, float] | None:
    bel = cell.get("attributes", {}).get("NEXTPNR_BEL", "")
    match = re.match(r"X(\d+)Y(\d+)", bel)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def routing_map(routed: dict[str, Any], path: Path) -> None:
    """Render route occupancy plus logical connections when Gowin has no wire decals."""
    module = routed["modules"]["top"]
    wire_re = re.compile(r"^X(\d+)Y(\d+)")
    heat = np.zeros((54, 56), dtype=np.int64)
    routed_nets = 0
    bound_wires = 0
    for net in module["netnames"].values():
        route = net.get("attributes", {}).get("ROUTING", "")
        fields = route.split(";")
        net_has_route = False
        for index in range(0, len(fields) - 2, 3):
            match = wire_re.match(fields[index])
            if not match:
                continue
            x, y = int(match.group(1)), int(match.group(2))
            if 0 <= x < heat.shape[1] and 0 <= y < heat.shape[0]:
                heat[y, x] += 1
                bound_wires += 1
                net_has_route = True
        routed_nets += int(net_has_route)

    cells = module["cells"]
    cell_positions = {name: pos for name, cell in cells.items() if (pos := _bel_xy(cell)) is not None}
    drivers: dict[int, tuple[float, float]] = {}
    sinks: dict[int, list[tuple[float, float]]] = {}
    for name, cell in cells.items():
        position = cell_positions.get(name)
        if position is None:
            continue
        directions = cell.get("port_directions", {})
        for port, bits in cell.get("connections", {}).items():
            direction = directions.get(port)
            for bit in bits:
                if not isinstance(bit, int):
                    continue
                if direction == "output":
                    drivers.setdefault(bit, position)
                elif direction == "input":
                    sinks.setdefault(bit, []).append(position)

    segments: list[list[tuple[float, float]]] = []
    for bit, driver in drivers.items():
        users = sinks.get(bit, [])
        if not users or len(users) > 32:
            continue
        for sink in users:
            if sink != driver:
                segments.append([driver, sink])

    fig = _figure()
    _header(
        fig,
        "Physical routing pressure + connections",
        "Heat = actual nextpnr bound-wire occupancy; blue lines = placed logical driver-to-sink endpoints.\n"
        "Gowin's native SVG backend does not implement wire or pip drawing.",
    )
    ax = fig.add_axes([0.08, 0.28, 0.84, 0.49], facecolor="#0B1220")
    masked = np.ma.masked_equal(heat, 0)
    image = ax.imshow(
        masked,
        origin="upper",
        extent=(-0.5, 55.5, 53.5, -0.5),
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1, vmax=max(int(heat.max()), 1)),
        aspect="equal",
    )
    ax.add_collection(LineCollection(segments, colors=BLUE_LIGHT, linewidths=0.22, alpha=0.075, rasterized=True))
    ax.set_xlim(-0.5, 55.5)
    ax.set_ylim(53.5, -0.5)
    ax.set_xlabel("Device X coordinate")
    ax.set_ylabel("Device Y coordinate")
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(length=0, labelsize=9)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.075, aspect=35)
    colorbar.set_label("Bound route wires per device tile · logarithmic color scale", color=MUTED)
    colorbar.ax.tick_params(colors=MUTED, length=0, labelsize=9)
    colorbar.outline.set_visible(False)

    stats = [
        ("ROUTED NETS", f"{routed_nets:,}"),
        ("BOUND WIRES", f"{bound_wires:,}"),
        ("LOGICAL EDGES", f"{len(segments):,}"),
    ]
    for x, (label, value) in zip([0.06, 0.37, 0.68], stats):
        fig.add_artist(Rectangle((x, 0.11), 0.26, 0.085, transform=fig.transFigure, facecolor=SURFACE, edgecolor=GRID))
        fig.text(x + 0.020, 0.162, label, fontsize=9, color=MUTED, weight="bold")
        fig.text(x + 0.020, 0.125, value, fontsize=19, color=INK, weight="bold")
    _footer(fig, "Source: routed JSON ROUTING attributes and NEXTPNR_BEL placement · seed 2")
    _finish(fig, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("build/showcase/artifact.json"))
    parser.add_argument(
        "--nextpnr-svg",
        type=Path,
        default=Path("build/showcase/nextpnr/placed.svg"),
    )
    parser.add_argument(
        "--routed-json",
        type=Path,
        default=Path("build/sipeed_tang_nano_20k/gateware/sipeed_tang_nano_20k_routed.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("build/showcase/png"))
    args = parser.parse_args()

    _configure_matplotlib()
    artifact = _load(args.artifact)
    routed = _load(args.routed_json)
    outputs = {
        "throughput": args.output_dir / "01_throughput_boundaries.png",
        "cycles": args.output_dir / "02_cycle_breakdown.png",
        "gemm": args.output_dir / "03_pure_gemm_comparison.png",
        "placement": args.output_dir / "04_fpga_placement.png",
        "routing": args.output_dir / "05_fpga_routing.png",
    }
    throughput_boundaries(artifact, outputs["throughput"])
    cycle_breakdown(artifact, outputs["cycles"])
    pure_gemm(artifact, outputs["gemm"])
    placement(args.nextpnr_svg, artifact, outputs["placement"])
    routing_map(routed, outputs["routing"])
    for output in outputs.values():
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
