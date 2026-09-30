"""Generate raw toolkit PNG velocity maps for the TD-RTM paper models.

The images deliberately use the toolkit's legacy RGB palette: blue is the
water/defect region, red is aluminium, and white is a colocated source and
receiver marker.  Pillow writes the RGB pixels directly; there are no axes,
interpolation, antialiasing, or plot borders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


NX = 801
NZ = 1041
DX_MM = 0.05
DZ_MM = 0.05
X_MIN_MM = -20.0
X_MAX_MM = 20.0
Z_MIN_MM = 0.0
Z_MAX_MM = 52.0
SURFACE_MEAN_MM = 14.0
WATER_RGB = np.array([0, 0, 255], dtype=np.uint8)
ALUMINIUM_RGB = np.array([255, 0, 0], dtype=np.uint8)
MARKER_RGB = np.array([255, 255, 255], dtype=np.uint8)
MARKER_X_INDICES = np.arange(85, 716, 10, dtype=np.int32)
MARKER_Z_INDEX = 1


def build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return exact physical coordinates corresponding to image indices."""

    x_mm = np.linspace(X_MIN_MM, X_MAX_MM, NX, dtype=np.float64)
    z_mm = np.linspace(Z_MIN_MM, Z_MAX_MM, NZ, dtype=np.float64)
    return x_mm, z_mm


def surface_z_mm(x_mm: np.ndarray, amplitude_mm: float) -> np.ndarray:
    """Evaluate the paper's Eq. (8) in plotted millimetre coordinates."""

    return SURFACE_MEAN_MM + amplitude_mm * np.sin(0.05 * np.pi * x_mm)


def _segment_mask(
    x_mm: np.ndarray,
    z_mm: np.ndarray,
    start_mm: tuple[float, float],
    end_mm: tuple[float, float],
    width_mm: float = 0.4,
) -> np.ndarray:
    """Rasterize a finite square-ended crack strip at grid nodes."""

    x_grid, z_grid = np.meshgrid(x_mm, z_mm)
    x0, z0 = start_mm
    x1, z1 = end_mm
    dx = x1 - x0
    dz = z1 - z0
    length_squared = dx * dx + dz * dz
    projection = ((x_grid - x0) * dx + (z_grid - z0) * dz) / length_squared
    projected_x = x0 + projection * dx
    projected_z = z0 + projection * dz
    distance = np.hypot(x_grid - projected_x, z_grid - projected_z)
    tolerance = 1e-6
    return (
        (projection >= -tolerance)
        & (projection <= 1.0 + tolerance)
        & (distance <= width_mm / 2.0 + tolerance)
    )


def crack_mask(name: str, x_mm: np.ndarray, z_mm: np.ndarray) -> np.ndarray:
    """Return the inferred T or Y geometry from Figs. 4(a) and 6."""

    junction_z_mm = 33.0
    branch_length_mm = 4.0
    mask = np.zeros((NZ, NX), dtype=bool)
    junction = (0.0, junction_z_mm)

    # The schematic stem points upward from the junction.
    mask |= _segment_mask(
        x_mm,
        z_mm,
        junction,
        (0.0, junction_z_mm - branch_length_mm),
    )

    if name == "t_b1":
        mask |= _segment_mask(
            x_mm,
            z_mm,
            (-branch_length_mm / 2.0, junction_z_mm),
            (branch_length_mm / 2.0, junction_z_mm),
        )
        return mask

    if name.startswith("y_b"):
        # Two 4 mm arms with a 120 degree included angle; z is positive down.
        horizontal = branch_length_mm * np.cos(np.deg2rad(30.0))
        vertical = branch_length_mm * np.sin(np.deg2rad(30.0))
        mask |= _segment_mask(
            x_mm,
            z_mm,
            junction,
            (-horizontal, junction_z_mm + vertical),
        )
        mask |= _segment_mask(
            x_mm,
            z_mm,
            junction,
            (horizontal, junction_z_mm + vertical),
        )
        return mask

    raise ValueError(f"Unknown model name: {name}")


def build_model(name: str, amplitude_mm: float) -> np.ndarray:
    """Create one raw RGB map with the 64-point colocated array."""

    x_mm, z_mm = build_grid()
    interface = surface_z_mm(x_mm, amplitude_mm)
    water = z_mm[:, np.newaxis] <= interface[np.newaxis, :]

    rgb = np.empty((NZ, NX, 3), dtype=np.uint8)
    rgb[:] = ALUMINIUM_RGB
    rgb[water] = WATER_RGB
    # Defects are filled with the water/blue material, with no marker-like
    # white coating.  The paper does not specify crack material; this follows
    # the light-blue crack slots in the schematics.
    rgb[crack_mask(name, x_mm, z_mm)] = WATER_RGB

    # A single white pixel is a point approximation of each finite 0.4 mm
    # element.  The bitmap loader interprets white as colocated TX/RX.
    rgb[MARKER_Z_INDEX, MARKER_X_INDICES] = MARKER_RGB
    return rgb


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write RGB rows directly without palette conversion or antialiasing."""

    image = Image.frombytes("RGB", (NX, NZ), rgb.tobytes())
    image.save(path, format="PNG")


def generate(output_dir: Path) -> list[Path]:
    """Generate the five paper simulation maps and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "t_b1": 1.0,
        "y_b1": 1.0,
        "y_b2": 2.0,
        "y_b3": 3.0,
        "y_b4": 4.0,
    }
    paths = []
    for name, amplitude_mm in models.items():
        path = output_dir / f"td_rtm_{name}.png"
        write_png(path, build_model(name, amplitude_mm))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "models",
        help="Directory for the five PNG maps.",
    )
    args = parser.parse_args()
    paths = generate(args.output_dir)
    print(f"Generated {len(paths)} raw RGB TD-RTM maps on ({NZ}, {NX}) grids.")


if __name__ == "__main__":
    main()
