from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any
import uuid


DEFAULT_BASEMAP_PATH = "data/basemaps/world_natural_earth_ii_8192.jpg"
POLYGON_EASING_LINEAR = "linear"
POLYGON_EASING_MODES = (
    POLYGON_EASING_LINEAR,
    "ease_in",
    "ease_out",
    "ease_in_out",
)
TEXT_BOX_ANCHORS = (
    "top_left",
    "top_center",
    "top_right",
    "center_left",
    "center",
    "center_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)
TEXT_ALIGN_MODES = (
    "left",
    "center",
    "right",
)


_TEMPLATE_TOKEN_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


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
    outgoing_easing: str = POLYGON_EASING_LINEAR
    outgoing_constant_area: bool = False


@dataclass
class PolygonLayer:
    id: str = field(default_factory=lambda: _uid("poly"))
    name: str = "Reveal"
    color: list[int] = field(default_factory=lambda: [232, 241, 255])
    feather_px: int = 2
    rounding_px: int = 222
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
    feather_px: int = 12
    rounding_px: int = 120
    start_frame: int = 0
    end_frame: int = 120
    label: str = ""
    show_in_legend: bool = True
    draw_mode: str = "colored"
    visible: bool = True
    points: list[list[float]] = field(default_factory=list)
    keyframes: list["RouteKeyframe"] = field(default_factory=list)


@dataclass
class RouteKeyframe:
    frame: int
    progress: float = 0.0


@dataclass
class TextOverlayLayer:
    id: str = field(default_factory=lambda: _uid("text"))
    name: str = "Text"
    color: list[int] = field(default_factory=lambda: [247, 250, 252])
    visible: bool = True
    opacity: float = 1.0
    template: str = ""
    anchor: str = "top_right"
    offset_x: float = -0.03
    offset_y: float = 0.03
    alignment: str = "right"
    font_size: int = 48
    padding_x: int = 18
    padding_y: int = 12
    background_color: list[int] = field(default_factory=lambda: [11, 15, 21])
    background_opacity: float = 0.72
    border_color: list[int] = field(default_factory=lambda: [225, 232, 242])
    border_opacity: float = 0.28
    border_width: int = 1
    corner_radius: int = 12
    frame_start: int = 0
    frame_end: int = -1
    text_keyframes: list["TextOverlayKeyframe"] = field(default_factory=list)


@dataclass
class TextOverlayKeyframe:
    frame: int
    template: str = ""


@dataclass
class TimeKeyframe:
    frame: int
    year: float = 0.0
    label: str = ""


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
    text_layers: list[TextOverlayLayer] = field(default_factory=list)
    time_keyframes: list[TimeKeyframe] = field(default_factory=list)

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


def _coerce_polygon_easing(value: Any) -> str:
    candidate = str(value or POLYGON_EASING_LINEAR).strip().lower()
    return candidate if candidate in POLYGON_EASING_MODES else POLYGON_EASING_LINEAR


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_anchor(value: Any) -> str:
    candidate = str(value or "top_right").strip().lower()
    return candidate if candidate in TEXT_BOX_ANCHORS else "top_right"


def _coerce_alignment(value: Any) -> str:
    candidate = str(value or "right").strip().lower()
    return candidate if candidate in TEXT_ALIGN_MODES else "right"


def _coerce_color(data: Any, fallback: list[int]) -> list[int]:
    raw = data if isinstance(data, (list, tuple)) else fallback
    values = [int(v) for v in list(raw)[:3]]
    if len(values) < 3:
        values += [int(v) for v in fallback[len(values):3]]
    return values[:3]


def _polygon_from_dict(data: dict[str, Any]) -> PolygonLayer:
    keyframes = [
        PolygonKeyframe(
            frame=int(item.get("frame", 0)),
            points=_coerce_points(item.get("points", [])),
            outgoing_easing=_coerce_polygon_easing(item.get("outgoing_easing", item.get("easing", POLYGON_EASING_LINEAR))),
            outgoing_constant_area=_coerce_bool(item.get("outgoing_constant_area", item.get("constant_area", False))),
        )
        for item in data.get("keyframes", [])
    ]
    keyframes.sort(key=lambda item: item.frame)
    return PolygonLayer(
        id=str(data.get("id") or _uid("poly")),
        name=str(data.get("name") or "Reveal"),
        color=[int(v) for v in data.get("color", [232, 241, 255])[:3]],
        feather_px=int(data.get("feather_px", 2)),
        rounding_px=int(data.get("rounding_px", 222)),
        opacity=float(data.get("opacity", 1.0)),
        visible=bool(data.get("visible", True)),
        keyframes=keyframes,
    )


def _route_from_dict(data: dict[str, Any]) -> RouteLayer:
    keyframes = [
        RouteKeyframe(
            frame=int(item.get("frame", 0)),
            progress=max(0.0, min(1.0, float(item.get("progress", 0.0)))),
        )
        for item in data.get("keyframes", data.get("route_keyframes", []))
    ]
    keyframes.sort(key=lambda item: int(item.frame))
    return RouteLayer(
        id=str(data.get("id") or _uid("route")),
        name=str(data.get("name") or "Route"),
        color=_coerce_color(data.get("color", [190, 228, 255]), [190, 228, 255]),
        width_px=max(1, int(data.get("width_px", 6))),
        reveal_px=max(1, int(data.get("reveal_px", 28))),
        feather_px=max(0, int(data.get("feather_px", 12))),
        rounding_px=max(0, int(data.get("rounding_px", 120))),
        start_frame=int(data.get("start_frame", 0)),
        end_frame=int(data.get("end_frame", 120)),
        label=str(data.get("label") or ""),
        show_in_legend=bool(data.get("show_in_legend", True)),
        draw_mode=str(data.get("draw_mode") or "colored").strip().lower() or "colored",
        visible=bool(data.get("visible", True)),
        points=_coerce_points(data.get("points", [])),
        keyframes=keyframes,
    )


def _text_layer_from_dict(data: dict[str, Any]) -> TextOverlayLayer:
    text_keyframes = [
        TextOverlayKeyframe(
            frame=int(item.get("frame", 0)),
            template=str(item.get("template", item.get("text", "")) or ""),
        )
        for item in data.get("text_keyframes", [])
    ]
    text_keyframes.sort(key=lambda item: int(item.frame))
    return TextOverlayLayer(
        id=str(data.get("id") or _uid("text")),
        name=str(data.get("name") or "Text"),
        color=_coerce_color(data.get("color", [247, 250, 252]), [247, 250, 252]),
        visible=bool(data.get("visible", True)),
        opacity=float(data.get("opacity", 1.0)),
        template=str(data.get("template", data.get("text", "")) or ""),
        anchor=_coerce_anchor(data.get("anchor", "top_right")),
        offset_x=float(data.get("offset_x", -0.03)),
        offset_y=float(data.get("offset_y", 0.03)),
        alignment=_coerce_alignment(data.get("alignment", data.get("align", "right"))),
        font_size=max(1, int(data.get("font_size", 48))),
        padding_x=max(0, int(data.get("padding_x", 18))),
        padding_y=max(0, int(data.get("padding_y", 12))),
        background_color=_coerce_color(data.get("background_color", [11, 15, 21]), [11, 15, 21]),
        background_opacity=float(data.get("background_opacity", 0.72)),
        border_color=_coerce_color(data.get("border_color", [225, 232, 242]), [225, 232, 242]),
        border_opacity=float(data.get("border_opacity", 0.28)),
        border_width=max(0, int(data.get("border_width", 1))),
        corner_radius=max(0, int(data.get("corner_radius", 12))),
        frame_start=int(data.get("frame_start", 0)),
        frame_end=int(data.get("frame_end", -1)),
        text_keyframes=text_keyframes,
    )


def _time_keyframe_from_dict(data: dict[str, Any]) -> TimeKeyframe:
    return TimeKeyframe(
        frame=int(data.get("frame", 0)),
        year=float(data.get("year", 0.0)),
        label=str(data.get("label") or ""),
    )


def format_project_year(year: float | None) -> str:
    if year is None:
        return ""
    rounded = int(round(float(year)))
    if rounded < 0:
        return f"{abs(rounded):,} BC"
    return f"{rounded:,}"


def interpolate_project_year(project: Project, frame_index: int) -> float | None:
    keyframes = sorted(project.time_keyframes, key=lambda item: int(item.frame))
    if not keyframes:
        return None
    current_frame = int(frame_index)
    if current_frame <= int(keyframes[0].frame):
        return float(keyframes[0].year)
    if current_frame >= int(keyframes[-1].frame):
        return float(keyframes[-1].year)
    for left, right in zip(keyframes, keyframes[1:]):
        left_frame = int(left.frame)
        right_frame = int(right.frame)
        if left_frame <= current_frame <= right_frame:
            if right_frame <= left_frame:
                return float(right.year)
            if current_frame == left_frame:
                return float(left.year)
            if current_frame == right_frame:
                return float(right.year)
            progress = (current_frame - left_frame) / float(right_frame - left_frame)
            return float(left.year) + (float(right.year) - float(left.year)) * progress
    return float(keyframes[-1].year)


def project_time_label(project: Project, frame_index: int) -> str:
    current_frame = int(frame_index)
    for keyframe in sorted(project.time_keyframes, key=lambda item: int(item.frame)):
        if int(keyframe.frame) == current_frame and str(keyframe.label).strip():
            return str(keyframe.label).strip()
    return format_project_year(interpolate_project_year(project, current_frame))


def route_layer_progress_at_frame(layer: RouteLayer, frame_index: int) -> float:
    current_frame = int(frame_index)
    keyframes = sorted(layer.keyframes, key=lambda item: int(item.frame))
    if not keyframes:
        if int(layer.end_frame) <= int(layer.start_frame):
            return 1.0 if current_frame >= int(layer.end_frame) else 0.0
        progress = (current_frame - int(layer.start_frame)) / float(int(layer.end_frame) - int(layer.start_frame))
        return max(0.0, min(1.0, progress))

    if current_frame <= int(keyframes[0].frame):
        return max(0.0, min(1.0, float(keyframes[0].progress)))
    if current_frame >= int(keyframes[-1].frame):
        return max(0.0, min(1.0, float(keyframes[-1].progress)))

    for left, right in zip(keyframes, keyframes[1:]):
        left_frame = int(left.frame)
        right_frame = int(right.frame)
        if left_frame <= current_frame <= right_frame:
            left_progress = max(0.0, min(1.0, float(left.progress)))
            right_progress = max(0.0, min(1.0, float(right.progress)))
            if right_frame <= left_frame:
                return right_progress
            if current_frame == left_frame:
                return left_progress
            if current_frame == right_frame:
                return right_progress
            progress = (current_frame - left_frame) / float(right_frame - left_frame)
            return left_progress + (right_progress - left_progress) * progress
    return max(0.0, min(1.0, float(keyframes[-1].progress)))


def text_overlay_template_at_frame(layer: TextOverlayLayer, frame_index: int) -> str:
    current_frame = int(frame_index)
    keyframes = sorted(layer.text_keyframes, key=lambda item: int(item.frame))
    active_template = str(layer.template or "")
    for keyframe in keyframes:
        if int(keyframe.frame) > current_frame:
            break
        active_template = str(keyframe.template or "")
    return active_template


def build_text_template_context(project: Project, frame_index: int) -> dict[str, str]:
    current_frame = int(frame_index)
    frame_max = max(0, int(project.frame_count) - 1)
    progress = 1.0 if frame_max <= 0 else current_frame / float(frame_max)
    current_year = interpolate_project_year(project, current_frame)
    year_text = ""
    if current_year is not None:
        year_text = str(int(round(float(current_year))))
    progress_pct = progress * 100.0
    return {
        "project_title": str(project.title or ""),
        "title": str(project.title or ""),
        "frame": str(current_frame),
        "frame_max": str(frame_max),
        "progress": f"{progress:.3f}",
        "progress_pct": f"{progress_pct:.0f}%",
        "progress_percent": f"{progress_pct:.0f}%",
        "year": year_text,
        "time_label": project_time_label(project, current_frame),
        "fps": str(int(project.fps)),
        "duration_sec": f"{float(project.duration_sec):.1f}",
    }


def render_text_template(template: str, project: Project, frame_index: int) -> str:
    context = build_text_template_context(project, frame_index)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        return context.get(token, match.group(0))

    return _TEMPLATE_TOKEN_RE.sub(_replace, str(template or ""))


def project_to_dict(project: Project, project_file: str | Path | None = None) -> dict[str, Any]:
    data = asdict(project)
    data["polygon_layers"] = sorted(
        data["polygon_layers"],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    data["route_layers"] = sorted(
        data["route_layers"],
        key=lambda item: str(item.get("name", "")).lower(),
    )
    for route_layer in data["route_layers"]:
        route_layer["keyframes"] = sorted(
            route_layer.get("keyframes", route_layer.get("route_keyframes", [])),
            key=lambda item: int(item.get("frame", 0)),
        )
    data["text_layers"] = sorted(
        data.get("text_layers", []),
        key=lambda item: str(item.get("name", "")).lower(),
    )
    for text_layer in data["text_layers"]:
        text_layer["text_keyframes"] = sorted(
            text_layer.get("text_keyframes", []),
            key=lambda item: int(item.get("frame", 0)),
        )
    data["time_keyframes"] = sorted(
        data.get("time_keyframes", []),
        key=lambda item: int(item.get("frame", 0)),
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
        text_layers=[_text_layer_from_dict(item) for item in data.get("text_layers", [])],
        time_keyframes=[_time_keyframe_from_dict(item) for item in data.get("time_keyframes", [])],
    )
    return project


def save_project(project: Project, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = project_to_dict(project, project_file=target)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
