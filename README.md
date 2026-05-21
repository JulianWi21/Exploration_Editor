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
- MP4 export through FFmpeg with RisingSeaLevel-style fade in/out, 3 second end hold, and optional soundtrack muxing
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

## Render A Video From The Terminal

Headless export without opening the editor:

```powershell
python render_project.py --project examples\age_of_discovery_demo.json
```

This writes an MP4 to `C:\Projekte\YouTube\GEOPANDA\exploration\exports\age_of_discovery_demo.mp4` by default.

New project JSON files created from the editor are saved to `C:\Projekte\YouTube\GEOPANDA\exploration\examples` by default.

If `C:\Projekte\YouTube\GEOPANDA\sound\soundtrack` exists, the export automatically picks a random MP3, adds a 2 second audio fade-in, keeps the last map frame on screen for 3 seconds, and fades both picture and music out at the end. You can override the soundtrack folder with `EXPLORATION_EDITOR_SOUNDTRACK_DIR`.

Quick test render with smaller output and shorter duration:

```powershell
python render_project.py --project examples\age_of_discovery_demo.json --output C:\Projekte\YouTube\GEOPANDA\exploration\exports\age_of_discovery_test.mp4 --width 960 --height 540 --fps 24 --duration 6
```

Useful overrides:

- `--width` / `--height` -> render size override
- `--fps` -> frame rate override
- `--duration` -> temporary duration override for short test renders
- `--output` -> custom MP4 output path

## Notes

- This MVP uses a clean headless renderer plus a thin PyQt6 editor.
- Route labels are currently handled as a legend, not text-on-path.
- Polygon interpolation keeps matched vertices stable across keyframes and only inserts proxy points locally when topology changes.
- Polygon keyframes can now retime their outgoing segment with Linear, Ease In, Ease Out, or Ease In-Out timing curves from the editor.
- Polygon keyframes can optionally equalize revealed area over time with the Constant Area checkbox on the outgoing segment.
