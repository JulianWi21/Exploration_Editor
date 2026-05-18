# Exploration_Editor

Exploration_Editor is a standalone Python tool for authoring and exporting historical world-map videos with fog of war, animated reveal polygons, and expedition routes.

The repo is intentionally separate from RisingSeaLevel and Tsunami. It reuses only lightweight assets such as city metadata and fonts. Heavy DEM sources stay outside this repo.

## Current MVP

- Interactive PyQt6 editor
- Load an equirectangular world basemap
- Draw reveal polygons on a world map
- Add polygon keyframes and interpolate between them
- Draw expedition routes and animate path progress
- Dark fog-of-war overlay with feathered reveal masks
- Legend for active routes
- PNG export
- MP4 export through FFmpeg
- Helper script to build a static world basemap from a world DEM

## Repo Layout

- `exploration_editor/` -> application package
- `config/` -> lightweight copied metadata from RisingSeaLevel
- `fonts/` -> local UI/export fonts
- `examples/` -> starter project files
- `data/basemaps/` -> local rendered world basemap output

## Setup

```powershell
cd Exploration_Editor
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Build A Local World Basemap

If you already have the world ETOPO DEM in the sibling RisingSeaLevel repo:

```powershell
python build_world_basemap.py --dem ..\RisingSeaLevel\data\world\etopo2022_surface_15s_world.tif --output data\basemaps\world_etopo_8192.jpg --width 8192
```

The script also tries to auto-detect the sibling DEM path when `--dem` is omitted.

## Run The Editor

```powershell
python run_editor.py --project examples\age_of_discovery_demo.json
```

If the basemap file in the project does not exist yet, open a basemap manually from the UI or build it first.

## Notes

- This MVP uses a clean headless renderer plus a thin PyQt6 editor.
- Route labels are currently handled as a legend, not text-on-path.
- Polygon interpolation uses perimeter resampling, which is robust enough for authoring but still simple to understand and extend.
