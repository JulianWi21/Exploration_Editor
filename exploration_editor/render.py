from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from exploration_editor.basemap import load_basemap_image
from exploration_editor.geometry import lonlat_to_world, path_prefix, polygon_points_at_frame, rounded_closed_path, route_progress, unwrap_longitudes
from exploration_editor.model import Project, RouteLayer, ViewState


@dataclass(frozen=True)
class PreparedFrameRender:
    output_size: tuple[int, int]
    layout: dict[str, float]
    background_array: np.ndarray
    map_alpha: np.ndarray


def clamp_view_state(
    frame_size: tuple[int, int],
    world_size: tuple[int, int],
    view: ViewState,
) -> ViewState:
    frame_w, frame_h = frame_size
    world_w, world_h = world_size
    base_scale = max(frame_w / float(world_w), frame_h / float(world_h))
    zoom = max(1.0, float(view.zoom))
    scale = max(0.05, base_scale * zoom)
    draw_w = world_w * scale
    draw_h = world_h * scale
    center_x = (frame_w - draw_w) * 0.5
    center_y = (frame_h - draw_h) * 0.5
    offset_x = center_x + float(view.offset_x)
    offset_y = center_y + float(view.offset_y)

    if draw_w >= frame_w:
        offset_x = min(0.0, max(frame_w - draw_w, offset_x))
    else:
        offset_x = center_x
    if draw_h >= frame_h:
        offset_y = min(0.0, max(frame_h - draw_h, offset_y))
    else:
        offset_y = center_y

    return ViewState(
        zoom=zoom,
        offset_x=offset_x - center_x,
        offset_y=offset_y - center_y,
    )


def compute_map_layout(
    frame_size: tuple[int, int],
    world_size: tuple[int, int],
    view: ViewState,
) -> dict[str, float]:
    frame_w, frame_h = frame_size
    world_w, world_h = world_size
    clamped_view = clamp_view_state(frame_size, world_size, view)
    base_scale = max(frame_w / float(world_w), frame_h / float(world_h))
    scale = max(0.05, base_scale * max(1.0, float(clamped_view.zoom)))
    draw_w = world_w * scale
    draw_h = world_h * scale
    offset_x = (frame_w - draw_w) * 0.5 + float(clamped_view.offset_x)
    offset_y = (frame_h - draw_h) * 0.5 + float(clamped_view.offset_y)
    return {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "world_w": world_w,
        "world_h": world_h,
        "scale": scale,
        "draw_w": draw_w,
        "draw_h": draw_h,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def _screen_path_variants(points: list[list[float]], layout: dict[str, float]) -> list[list[tuple[float, float]]]:
    if not points:
        return []
    world_size = (layout["world_w"], layout["world_h"])
    continuous_points = unwrap_longitudes(points)
    base_points = [
        lonlat_to_world(lon, lat, world_size, wrap=False)
        for lon, lat in continuous_points
    ]
    variants: list[list[tuple[float, float]]] = []
    for shift in (-layout["world_w"], 0.0, layout["world_w"]):
        variants.append([
            (
                layout["offset_x"] + (x + shift) * layout["scale"],
                layout["offset_y"] + y * layout["scale"],
            )
            for x, y in base_points
        ])
    return variants


def _build_map_mask(frame_size: tuple[int, int], layout: dict[str, float]) -> Image.Image:
    image = Image.new("L", frame_size, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [
            layout["offset_x"],
            layout["offset_y"],
            layout["offset_x"] + layout["draw_w"],
            layout["offset_y"] + layout["draw_h"],
        ],
        fill=255,
    )
    return image


def _lighter_mask(base: Image.Image, candidate: Image.Image) -> Image.Image:
    return ImageChops.lighter(base, candidate)


def _build_reveal_mask(
    project: Project,
    frame_index: int,
    frame_size: tuple[int, int],
    layout: dict[str, float],
) -> Image.Image:
    reveal = Image.new("L", frame_size, 0)

    for layer in project.polygon_layers:
        if not layer.visible:
            continue
        points = polygon_points_at_frame(layer, frame_index)
        if len(points) < 3:
            continue
        temp = Image.new("L", frame_size, 0)
        draw = ImageDraw.Draw(temp)
        for path in _screen_path_variants(points, layout):
            render_path = rounded_closed_path(path, float(layer.rounding_px))
            draw.polygon(render_path, fill=int(255 * max(0.0, min(1.0, layer.opacity))))
        blur_radius = max(0, int(layer.feather_px))
        if blur_radius > 0:
            temp = temp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        reveal = _lighter_mask(reveal, temp)

    for layer in project.route_layers:
        if not layer.visible or len(layer.points) < 2:
            continue
        progress = route_progress(frame_index, layer.start_frame, layer.end_frame)
        if progress <= 0.0:
            continue
        route_points = path_prefix(layer.points, progress)
        temp = Image.new("L", frame_size, 0)
        draw = ImageDraw.Draw(temp)
        for path in _screen_path_variants(route_points, layout):
            if len(path) >= 2:
                draw.line(path, fill=255, width=max(2, int(layer.reveal_px)), joint="curve")
        blur_radius = max(1, int(layer.reveal_px * 0.45))
        temp = temp.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        reveal = _lighter_mask(reveal, temp)

    return reveal


@lru_cache(maxsize=32)
def _load_font_cached(font_key: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_key:
        candidate = Path(font_key)
        if candidate.exists():
            return ImageFont.truetype(candidate.as_posix(), size=size)
    return ImageFont.load_default()


def _load_font(font_path: str | Path | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_key = str(Path(font_path).resolve()) if font_path else ""
    return _load_font_cached(font_key, int(size))


def prepare_frame_render(
    project: Project,
    basemap_image: Image.Image | None = None,
    display_basemap_image: Image.Image | None = None,
    output_size: tuple[int, int] | None = None,
    preview: bool = True,
) -> PreparedFrameRender:
    if output_size is None:
        output_size = (project.width, project.height)
    frame_w, frame_h = output_size
    if basemap_image is None:
        basemap_image = load_basemap_image(project.basemap_path)
    if basemap_image.mode != "RGB":
        basemap_image = basemap_image.convert("RGB")
    if display_basemap_image is None:
        display_basemap_image = basemap_image
    elif display_basemap_image.mode != "RGB":
        display_basemap_image = display_basemap_image.convert("RGB")

    layout = compute_map_layout((frame_w, frame_h), basemap_image.size, project.view)
    background = Image.new("RGB", (frame_w, frame_h), tuple(project.fog_color[:3]))

    draw_w = max(1, int(round(layout["draw_w"])))
    draw_h = max(1, int(round(layout["draw_h"])))
    resample = Image.Resampling.BILINEAR if preview else Image.Resampling.LANCZOS
    scaled_map = display_basemap_image.resize((draw_w, draw_h), resample=resample)
    background.paste(scaled_map, (int(round(layout["offset_x"])), int(round(layout["offset_y"]))))

    map_mask = _build_map_mask((frame_w, frame_h), layout)
    return PreparedFrameRender(
        output_size=(frame_w, frame_h),
        layout=layout,
        background_array=np.asarray(background, dtype=np.float32),
        map_alpha=np.asarray(map_mask, dtype=np.float32) / 255.0,
    )


def _draw_routes(
    frame_img: Image.Image,
    project: Project,
    frame_index: int,
    layout: dict[str, float],
) -> None:
    overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for layer in project.route_layers:
        if not layer.visible or len(layer.points) < 2:
            continue
        progress = route_progress(frame_index, layer.start_frame, layer.end_frame)
        if progress <= 0.0:
            continue
        route_points = path_prefix(layer.points, progress)
        for path in _screen_path_variants(route_points, layout):
            if len(path) >= 2:
                halo_width = max(3, int(layer.width_px) + 4)
                draw.line(path, fill=(255, 255, 255, 155), width=halo_width, joint="curve")
                draw.line(path, fill=tuple(layer.color[:3]) + (235,), width=max(1, int(layer.width_px)), joint="curve")
            elif len(path) == 1:
                px, py = path[0]
                radius = max(3, int(layer.width_px))
                draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=tuple(layer.color[:3]) + (235,))
    composed = Image.alpha_composite(frame_img.convert("RGBA"), overlay)
    frame_img.paste(composed.convert("RGB"), (0, 0))


def _draw_legend(
    frame_img: Image.Image,
    project: Project,
    frame_index: int,
    font_path: str | Path | None,
) -> None:
    visible_layers: list[RouteLayer] = []
    for layer in project.route_layers:
        if not layer.visible or not layer.show_in_legend:
            continue
        if route_progress(frame_index, layer.start_frame, layer.end_frame) <= 0.0:
            continue
        visible_layers.append(layer)
    if not visible_layers:
        return

    frame_w, frame_h = frame_img.size
    scale = frame_h / 1080.0
    title_font = _load_font(font_path, max(18, int(24 * scale)))
    item_font = _load_font(font_path, max(15, int(20 * scale)))
    pad_x = max(14, int(18 * scale))
    pad_y = max(10, int(12 * scale))
    line_gap = max(10, int(14 * scale))
    sample_w = max(38, int(54 * scale))
    sample_line_w = max(3, int(5 * scale))

    labels = [layer.label or layer.name for layer in visible_layers]
    title_text = "Routes"
    title_box = title_font.getbbox(title_text)
    label_boxes = [item_font.getbbox(text) for text in labels]
    text_width = max([title_box[2] - title_box[0]] + [box[2] - box[0] for box in label_boxes])
    line_height = max((box[3] - box[1]) for box in label_boxes)
    panel_w = pad_x * 2 + sample_w + pad_x + text_width
    panel_h = pad_y * 2 + (title_box[3] - title_box[1]) + line_gap + len(labels) * line_height + max(0, len(labels) - 1) * line_gap
    panel_x = frame_w - panel_w - max(18, int(26 * scale))
    panel_y = max(18, int(26 * scale))

    overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h],
        radius=max(10, int(12 * scale)),
        fill=(11, 15, 21, 178),
        outline=(225, 232, 242, 70),
        width=max(1, int(scale)),
    )

    cursor_y = panel_y + pad_y
    draw.text((panel_x + pad_x, cursor_y), title_text, font=title_font, fill=(247, 250, 252, 235))
    cursor_y += (title_box[3] - title_box[1]) + line_gap

    sample_x1 = panel_x + pad_x
    sample_x2 = sample_x1 + sample_w
    text_x = sample_x2 + pad_x
    for layer, text, box in zip(visible_layers, labels, label_boxes):
        cy = cursor_y + (box[3] - box[1]) * 0.5
        draw.line([(sample_x1, cy), (sample_x2, cy)], fill=(255, 255, 255, 140), width=sample_line_w + 3)
        draw.line([(sample_x1, cy), (sample_x2, cy)], fill=tuple(layer.color[:3]) + (235,), width=sample_line_w)
        draw.text((text_x, cursor_y), text, font=item_font, fill=(240, 244, 248, 230))
        cursor_y += line_height + line_gap

    composed = Image.alpha_composite(frame_img.convert("RGBA"), overlay)
    frame_img.paste(composed.convert("RGB"), (0, 0))


def render_frame(
    project: Project,
    basemap_image: Image.Image | None = None,
    display_basemap_image: Image.Image | None = None,
    frame_index: int = 0,
    output_size: tuple[int, int] | None = None,
    font_path: str | Path | None = None,
    preview: bool = True,
    prepared_render: PreparedFrameRender | None = None,
) -> Image.Image:
    if prepared_render is None:
        prepared_render = prepare_frame_render(
            project,
            basemap_image=basemap_image,
            display_basemap_image=display_basemap_image,
            output_size=output_size,
            preview=preview,
        )

    frame_w, frame_h = prepared_render.output_size
    layout = prepared_render.layout
    reveal_mask = _build_reveal_mask(project, frame_index, (frame_w, frame_h), layout)

    frame_arr = prepared_render.background_array.copy()
    reveal_alpha = np.asarray(reveal_mask, dtype=np.float32) / 255.0
    fog_mix = float(project.fog_opacity) * prepared_render.map_alpha * (1.0 - reveal_alpha)
    fog_color = np.asarray(project.fog_color[:3], dtype=np.float32)
    frame_arr = frame_arr * (1.0 - fog_mix[..., None]) + fog_color * fog_mix[..., None]
    frame_img = Image.fromarray(np.clip(frame_arr, 0, 255).astype(np.uint8), mode="RGB")

    _draw_routes(frame_img, project, frame_index, layout)
    _draw_legend(frame_img, project, frame_index, font_path)
    return frame_img
