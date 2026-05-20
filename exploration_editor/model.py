from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any
import uuid


DEFAULT_BASEMAP_PATH = "data/basemaps/world_etopo_8192.jpg"


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ViewState:
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass
class PolygonKeyframe:
    frame: int
    points: list[list[float]] = field(default_factory=list)


@dataclass
class PolygonLayer:
    id: str = field(default_factory=lambda: _uid("poly"))
    name: str = "Reveal"
    color: list[int] = field(default_factory=lambda: [232, 241, 255])
    feather_px: int = 20
    rounding_px: int = 18
    opacity: float = 1.0
    visible: bool = True
    keyframes: list[PolygonKeyframe] = field(default_factory=list)


@dataclass
class RouteLayer:
    id: str = field(default_factory=lambda: _uid("route"))
    name: str = "Route"
    color: list[int] = field(default_factory=lambda: [190, 228, 255])
    width_px: int = 6
    reveal_px: int = 28
    start_frame: int = 0
    end_frame: int = 120
    label: str = ""
    show_in_legend: bool = True
    visible: bool = True
    points: list[list[float]] = field(default_factory=list)


@dataclass
class Project:
    title: str = "Untitled Exploration"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    duration_sec: float = 20.0
    fog_opacity: float = 0.74
    fog_color: list[int] = field(default_factory=lambda: [20, 26, 34])
    basemap_path: str = DEFAULT_BASEMAP_PATH
    view: ViewState = field(default_factory=ViewState)
    polygon_layers: list[PolygonLayer] = field(default_factory=list)
    route_layers: list[RouteLayer] = field(default_factory=list)

    @property
    def frame_count(self) -> int:
        return max(2, int(round(float(self.duration_sec) * int(self.fps))))


def resolve_project_path(project_file: str | Path | None, value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute() or project_file is None:
        return str(path)
    return str((Path(project_file).resolve().parent / path).resolve())


def default_project(basemap_path: str = DEFAULT_BASEMAP_PATH) -> Project:
    return Project(basemap_path=basemap_path)


def _coerce_points(raw_points: list[Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    return points


def _polygon_from_dict(data: dict[str, Any]) -> PolygonLayer:
    keyframes = [
        PolygonKeyframe(
            frame=int(item.get("frame", 0)),
            points=_coerce_points(item.get("points", [])),
        )
        for item in data.get("keyframes", [])
    ]
    keyframes.sort(key=lambda item: item.frame)
    return PolygonLayer(
        id=str(data.get("id") or _uid("poly")),
        name=str(data.get("name") or "Reveal"),
        color=[int(v) for v in data.get("color", [232, 241, 255])[:3]],
        feather_px=int(data.get("feather_px", 20)),
        rounding_px=int(data.get("rounding_px", 18)),
        opacity=float(data.get("opacity", 1.0)),
        visible=bool(data.get("visible", True)),
        keyframes=keyframes,
    )


def _route_from_dict(data: dict[str, Any]) -> RouteLayer:
    return RouteLayer(
        id=str(data.get("id") or _uid("route")),
        name=str(data.get("name") or "Route"),
        color=[int(v) for v in data.get("color", [190, 228, 255])[:3]],
        width_px=int(data.get("width_px", 6)),
        reveal_px=int(data.get("reveal_px", 28)),
        start_frame=int(data.get("start_frame", 0)),
        end_frame=int(data.get("end_frame", 120)),
        label=str(data.get("label") or ""),
        show_in_legend=bool(data.get("show_in_legend", True)),
        visible=bool(data.get("visible", True)),
        points=_coerce_points(data.get("points", [])),
    )


def project_to_dict(project: Project, project_file: str | Path | None = None) -> dict[str, Any]:
    data = asdict(project)
    data["polygon_layers"] = sorted(
        data["polygon_layers"],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    if project_file and project.basemap_path:
        basemap_path = Path(project.basemap_path)
        if basemap_path.is_absolute():
            try:
                data["basemap_path"] = os.path.relpath(
                    basemap_path,
                    Path(project_file).resolve().parent,
                )
            except ValueError:
                data["basemap_path"] = str(basemap_path)
    return data


def load_project(path: str | Path) -> Project:
    project_path = Path(path)
    data = json.loads(project_path.read_text(encoding="utf-8"))
    view_data = data.get("view", {})
    project = Project(
        title=str(data.get("title") or "Untitled Exploration"),
        width=int(data.get("width", 1920)),
        height=int(data.get("height", 1080)),
        fps=int(data.get("fps", 30)),
        duration_sec=float(data.get("duration_sec", 20.0)),
        fog_opacity=float(data.get("fog_opacity", 0.74)),
        fog_color=[int(v) for v in data.get("fog_color", [20, 26, 34])[:3]],
        basemap_path=str(data.get("basemap_path") or DEFAULT_BASEMAP_PATH),
        view=ViewState(
            zoom=float(view_data.get("zoom", 1.0)),
            offset_x=float(view_data.get("offset_x", 0.0)),
            offset_y=float(view_data.get("offset_y", 0.0)),
        ),
        polygon_layers=[_polygon_from_dict(item) for item in data.get("polygon_layers", [])],
        route_layers=[_route_from_dict(item) for item in data.get("route_layers", [])],
    )
    return project


def save_project(project: Project, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = project_to_dict(project, project_file=target)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
