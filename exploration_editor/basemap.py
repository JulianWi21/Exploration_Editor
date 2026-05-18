from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import rasterio
from rasterio.enums import Resampling


OCEAN_COLOR_DEEP = np.array([70, 120, 195], dtype=np.float32)
OCEAN_COLOR_SHORE = np.array([130, 190, 240], dtype=np.float32)
TERRAIN_COLORS = [
    (0, (30, 120, 50)),
    (50, (60, 160, 60)),
    (100, (100, 180, 70)),
    (200, (160, 200, 80)),
    (300, (200, 210, 100)),
    (400, (220, 200, 80)),
    (500, (220, 170, 60)),
    (600, (210, 140, 50)),
    (800, (190, 100, 40)),
    (1000, (170, 70, 30)),
    (1500, (140, 50, 25)),
    (2000, (120, 100, 90)),
    (3000, (200, 200, 210)),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_world_dem_candidates() -> list[Path]:
    workspace_root = repo_root().parent
    return [
        repo_root() / "data" / "world_source" / "etopo2022_surface_15s_world.tif",
        workspace_root / "RisingSeaLevel" / "data" / "world" / "etopo2022_surface_15s_world.tif",
        workspace_root / "RisingSeaLevel" / "data" / "world" / "etopo2022_bedrock_15s_world.tif",
    ]


def find_default_world_dem() -> Path | None:
    for candidate in default_world_dem_candidates():
        if candidate.exists():
            return candidate
    return None


def create_placeholder_basemap(size: tuple[int, int] = (4096, 2048)) -> Image.Image:
    width, height = size
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    ocean = np.zeros((height, width, 3), dtype=np.float32)
    ocean[..., 0] = 36.0 + yy * 22.0
    ocean[..., 1] = 60.0 + yy * 34.0
    ocean[..., 2] = 92.0 + yy * 56.0
    ocean[..., 1] += np.sin(xx * np.pi * 6.0) * 3.0
    ocean[..., 2] += np.cos(yy * np.pi * 8.0) * 4.0

    image = Image.fromarray(np.clip(ocean, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    grid_color = (86, 116, 150)
    for lon in range(-150, 181, 30):
        x = int(round(((lon + 180.0) / 360.0) * (width - 1)))
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for lat in range(-60, 91, 30):
        y = int(round(((90.0 - lat) / 180.0) * (height - 1)))
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)
    return image


def _terrain_lut(max_elevation: int = 3000) -> np.ndarray:
    lut = np.zeros((max_elevation + 1, 3), dtype=np.uint8)
    for value in range(max_elevation + 1):
        if value <= TERRAIN_COLORS[0][0]:
            lut[value] = TERRAIN_COLORS[0][1]
            continue
        if value >= TERRAIN_COLORS[-1][0]:
            lut[value] = TERRAIN_COLORS[-1][1]
            continue
        for index in range(len(TERRAIN_COLORS) - 1):
            left_value, left_color = TERRAIN_COLORS[index]
            right_value, right_color = TERRAIN_COLORS[index + 1]
            if left_value <= value <= right_value:
                t = (value - left_value) / float(right_value - left_value)
                lut[value] = [
                    int(left_color[channel] * (1.0 - t) + right_color[channel] * t)
                    for channel in range(3)
                ]
                break
    return lut


def _adjust_saturation(rgb: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return rgb
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return luma[..., None] + (rgb - luma[..., None]) * factor


def _compute_hillshade(dem: np.ndarray, z_exag: float = 10.5) -> np.ndarray:
    filled = np.nan_to_num(dem, nan=0.0).astype(np.float32)
    dy, dx = np.gradient(filled)
    slope = np.arctan(np.sqrt(dx * dx + dy * dy) * z_exag)
    aspect = np.arctan2(-dy, dx)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    hillshade = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    return np.clip(hillshade, 0.34, 1.0)


def build_world_basemap(
    dem_path: str | Path,
    output_path: str | Path,
    width: int = 8192,
    height: int | None = None,
    land_brightness: float = 1.06,
    land_saturation: float = 1.04,
) -> Path:
    dem_path = Path(dem_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if height is None:
        height = max(1, width // 2)

    with rasterio.open(dem_path) as src:
        dem = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        dem[dem == nodata] = np.nan

    land_mask = np.isfinite(dem) & (dem >= 0.0)
    ocean_mask = np.isfinite(dem) & (dem < 0.0)

    lut = _terrain_lut()
    rgb = np.zeros((height, width, 3), dtype=np.float32)

    if np.any(land_mask):
        land_elev = np.clip(dem[land_mask], 0.0, 3000.0).astype(np.int32)
        land_rgb = lut[land_elev].astype(np.float32)
        land_rgb = _adjust_saturation(land_rgb, land_saturation) * land_brightness
        hillshade = _compute_hillshade(dem)
        land_rgb *= hillshade[land_mask][:, None]
        rgb[land_mask] = land_rgb

    if np.any(ocean_mask):
        depth = np.clip((-dem[ocean_mask]) / 5500.0, 0.0, 1.0)
        ocean_rgb = OCEAN_COLOR_SHORE * (1.0 - depth[:, None]) + OCEAN_COLOR_DEEP * depth[:, None]
        rgb[ocean_mask] = ocean_rgb

    rgb[~np.isfinite(dem)] = np.array([28, 38, 48], dtype=np.float32)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95, optimize=True)
    else:
        image.save(output_path)
    return output_path


def load_basemap_image(path: str | Path | None, fallback_size: tuple[int, int] = (4096, 2048)) -> Image.Image:
    if path:
        candidate = Path(path)
        if candidate.exists():
            with Image.open(candidate) as image:
                return image.convert("RGB")
    return create_placeholder_basemap(fallback_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static world basemap from an equirectangular DEM.")
    parser.add_argument("--dem", default=None, help="Input world DEM GeoTIFF. Auto-detects a sibling RisingSeaLevel DEM if omitted.")
    parser.add_argument("--output", default=str(repo_root() / "data" / "basemaps" / "world_etopo_8192.jpg"))
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dem_path = Path(args.dem) if args.dem else find_default_world_dem()
    if dem_path is None or not dem_path.exists():
        raise SystemExit("No world DEM found. Pass --dem or place a world GeoTIFF in data/world_source.")
    output = build_world_basemap(dem_path, args.output, width=args.width, height=args.height)
    print(f"Basemap saved to: {output}")


if __name__ == "__main__":
    main()
