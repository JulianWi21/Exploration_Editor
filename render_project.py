from __future__ import annotations

import argparse
from pathlib import Path
import time

from exploration_editor.basemap import load_basemap_image, repo_root
from exploration_editor.export import export_video
from exploration_editor.model import load_project, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an Exploration_Editor project to MP4 without opening the GUI.",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Path to the project JSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output MP4 path. Defaults to exports/<project-name>.mp4.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration override in seconds for quick test renders.",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Optional font path. Defaults to fonts/Montserrat-ExtraBold.ttf when available.",
    )
    return parser.parse_args()


def _default_output_path(project_path: Path) -> Path:
    return repo_root() / "exports" / f"{project_path.stem}.mp4"


def _default_font_path() -> Path | None:
    candidate = repo_root() / "fonts" / "Montserrat-ExtraBold.ttf"
    return candidate if candidate.exists() else None


def _progress_printer(start_time: float):
    last_report = {"done": 0}

    def _callback(done: int, total: int) -> None:
        interval = max(1, total // 20)
        if done != total and done - last_report["done"] < interval:
            return
        last_report["done"] = done
        elapsed = max(1e-6, time.time() - start_time)
        fps = done / elapsed
        eta = (total - done) / fps if fps > 0 else 0.0
        pct = (done / total) * 100.0 if total > 0 else 100.0
        print(f"  Frame {done}/{total} ({pct:.0f}%) - {fps:.1f} fps - ETA {eta:.0f}s")

    return _callback


def main() -> None:
    args = parse_args()
    project_path = Path(args.project).resolve()
    if not project_path.exists():
        raise SystemExit(f"Project not found: {project_path}")

    project = load_project(project_path)
    project.basemap_path = resolve_project_path(project_path, project.basemap_path)
    basemap_path = Path(project.basemap_path)
    if not basemap_path.exists():
        raise SystemExit(
            "Basemap not found. Build it with build_world_basemap.py or point the project to an existing image."
        )

    if args.width is not None:
        project.width = int(args.width)
    if args.height is not None:
        project.height = int(args.height)
    if args.fps is not None:
        project.fps = int(args.fps)
    if args.duration is not None:
        project.duration_sec = float(args.duration)

    output_path = Path(args.output).resolve() if args.output else _default_output_path(project_path)
    font_path = Path(args.font).resolve() if args.font else _default_font_path()
    basemap_image = load_basemap_image(basemap_path)

    print(f"Project : {project_path.name}")
    print(f"Basemap : {basemap_path}")
    print(f"Output  : {output_path}")
    print(f"Render  : {project.width}x{project.height} @ {project.fps} fps, {project.duration_sec:.1f}s")

    start_time = time.time()
    export_video(
        project,
        basemap_image,
        output_path,
        font_path=font_path,
        progress_callback=_progress_printer(start_time),
    )
    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.1f}s: {output_path}")


if __name__ == "__main__":
    main()