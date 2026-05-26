from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from exploration_editor.basemap import load_basemap_image
from exploration_editor.geometry import expand_closed_path_structure, lonlat_to_world, path_prefix, polygon_points_at_frame, polygon_segment_progress, rounded_closed_path, route_progress, unwrap_longitudes
from exploration_editor.model import PolygonLayer, Project, RouteLayer, TextOverlayLayer, ViewState, render_text_template, text_overlay_template_at_frame


@dataclass(frozen=True)
class PreparedFrameRender:
    output_size: tuple[int, int]
    layout: dict[str, float]
    background_array: np.ndarray
    map_alpha: np.ndarray


_TEXT_ANCHOR_RATIOS = {
    "top_left": (0.0, 0.0),
    "top_center": (0.5, 0.0),
    "top_right": (1.0, 0.0),
    "center_left": (0.0, 0.5),
    "center": (0.5, 0.5),
    "center_right": (1.0, 0.5),
    "bottom_left": (0.0, 1.0),
    "bottom_center": (0.5, 1.0),
    "bottom_right": (1.0, 1.0),
}


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

    if draw_w > 1e-6:
        offset_x = offset_x % draw_w
        if offset_x > 0.0:
            offset_x -= draw_w
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


def _rounded_screen_variants(points: list[list[float]], layout: dict[str, float], radius: float) -> list[list[tuple[float, float]]]:
    variants = _screen_path_variants(points, layout)
    if radius <= 0.0:
        return variants
    return [rounded_closed_path(path, radius) for path in variants]


_SMOOTH_RESAMPLE_COUNT = 256


def _smooth_resampled_screen_variants(
    points: list[list[float]],
    layout: dict[str, float],
    radius: float,
) -> list[list[tuple[float, float]]]:
    variants = _screen_path_variants(points, layout)
    result = []
    for variant in variants:
        if len(variant) < 3:
            result.append(variant)
            continue
        smooth = rounded_closed_path(variant, radius) if radius > 0.0 else variant
        result.append(resample_closed_screen_path(smooth, _SMOOTH_RESAMPLE_COUNT))
    return result


def polygon_outline_screen_paths_at_frame(layer: PolygonLayer, frame_index: int, layout: dict[str, float]) -> list[list[tuple[float, float]]]:
    if not layer.keyframes:
        return []
    radius = float(layer.rounding_px)
    keyframes = sorted(layer.keyframes, key=lambda kf: kf.frame)

    # At or before first keyframe
    if frame_index <= keyframes[0].frame:
        pts = [list(p) for p in keyframes[0].points]
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []

    # At or after last keyframe
    if frame_index >= keyframes[-1].frame:
        pts = [list(p) for p in keyframes[-1].points]
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []

    # Find bounding keyframes
    left = right = None
    for idx in range(len(keyframes) - 1):
        if keyframes[idx].frame <= frame_index <= keyframes[idx + 1].frame:
            left = keyframes[idx]
            right = keyframes[idx + 1]
            break
    if left is None or right is None:
        return []

    # At exact keyframe boundary: Catmull-Rom through exact keyframe points
    if frame_index == left.frame:
        pts = [list(p) for p in left.points]
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []
    if frame_index == right.frame:
        pts = [list(p) for p in right.points]
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []

    # Between keyframes — same topology: interpolate control points first, then smooth.
    # The Catmull-Rom curve passes through every interpolated control point.
    if len(left.points) == len(right.points):
        pts = polygon_points_at_frame(layer, frame_index)
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []

    # Between keyframes — different topology (N vs M points): place proxy points ON
    # the source polygon's smooth Catmull-Rom arc (not on straight chords) so that
    # at t≈0 the curve matches the source keyframe exactly, and during the transition
    # every interpolated point stays near the polygon boundary (no wild bulging).
    left_pts = [list(p) for p in left.points]
    right_pts = [list(p) for p in right.points]
    span = max(1, right.frame - left.frame)
    raw_t = (frame_index - left.frame) / span
    t = polygon_segment_progress(
        left_pts, right_pts, raw_t,
        easing=getattr(left, "outgoing_easing", None),
        constant_area=getattr(left, "outgoing_constant_area", False),
    )

    if radius <= 0.0:
        # No smoothing: proxy points on straight chords are fine.
        pts = polygon_points_at_frame(layer, frame_index)
        return _rounded_screen_variants(pts, layout, radius) if len(pts) >= 3 else []

    # Determine smaller vs larger polygon.
    if len(left_pts) <= len(right_pts):
        smaller_pts, larger_pts = left_pts, right_pts
        smaller_is_left = True
    else:
        smaller_pts, larger_pts = right_pts, left_pts
        smaller_is_left = False

    aligned_larger, structure = expand_closed_path_structure(smaller_pts, larger_pts)

    result = []
    for v_small, v_large in zip(
        _screen_path_variants(smaller_pts, layout),
        _screen_path_variants(aligned_larger, layout),
    ):
        if len(v_small) < 3:
            result.append(v_small)
            continue

        # Smooth arc for the smaller polygon: proxy points will land on this curve.
        small_smooth = rounded_closed_path(v_small, radius)
        n_s = len(v_small)
        s = len(small_smooth) // n_s if n_s > 0 else 1

        if s < 2 or len(small_smooth) != n_s * s:
            # Degenerate case: fall back to straight-chord proxies.
            pts = polygon_points_at_frame(layer, frame_index)
            scr = _screen_path_variants(pts, layout)
            result.extend(rounded_closed_path(v, radius) if len(v) >= 3 else v for v in scr)
            return result

        # Build expanded screen positions: originals at their screen coords,
        # proxies sampled from the smooth arc at the correct fractional position.
        expanded: list[tuple[float, float]] = []
        for item in structure:
            if item[0] == 'original':
                expanded.append(v_small[item[1]])
            else:  # 'proxy'
                edge_i: int = item[1]
                frac: float = item[2]
                arc_float = edge_i * s + frac * (s - 1)
                idx0 = max(edge_i * s, min(int(arc_float), (edge_i + 1) * s - 1))
                idx1 = min(idx0 + 1, (edge_i + 1) * s - 1)
                f = arc_float - int(arc_float)
                px = small_smooth[idx0][0] + f * (small_smooth[idx1][0] - small_smooth[idx0][0])
                py = small_smooth[idx0][1] + f * (small_smooth[idx1][1] - small_smooth[idx0][1])
                expanded.append((px, py))

        # Interpolate: expanded points lie on the smaller KF's smooth arc;
        # v_large holds the larger KF's raw screen positions.
        if smaller_is_left:
            interp = [(lx + t * (rx - lx), ly + t * (ry - ly))
                      for (lx, ly), (rx, ry) in zip(expanded, v_large)]
        else:
            interp = [(lx + t * (rx - lx), ly + t * (ry - ly))
                      for (lx, ly), (rx, ry) in zip(v_large, expanded)]

        result.append(rounded_closed_path(interp, radius) if len(interp) >= 3 else interp)
    return result


def _build_map_mask(frame_size: tuple[int, int], layout: dict[str, float]) -> Image.Image:
    image = Image.new("L", frame_size, 0)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [
            0,
            layout["offset_y"],
            frame_size[0],
            layout["offset_y"] + layout["draw_h"],
        ],
        fill=255,
    )
    return image


def _paste_clipped(base: Image.Image, tile: Image.Image, x: int, y: int) -> None:
    left = max(0, int(x))
    top = max(0, int(y))
    right = min(base.width, int(x) + tile.width)
    bottom = min(base.height, int(y) + tile.height)
    if right <= left or bottom <= top:
        return
    crop = tile.crop((left - int(x), top - int(y), right - int(x), bottom - int(y)))
    base.paste(crop, (left, top))


def _paste_wrapped_basemap(base: Image.Image, tile: Image.Image, layout: dict[str, float]) -> None:
    tile_w = max(1, tile.width)
    x = int(round(layout["offset_x"]))
    y = int(round(layout["offset_y"]))
    while x > 0:
        x -= tile_w
    while x < base.width:
        _paste_clipped(base, tile, x, y)
        x += tile_w


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
        paths = polygon_outline_screen_paths_at_frame(layer, frame_index, layout)
        if not paths:
            continue
        temp = Image.new("L", frame_size, 0)
        draw = ImageDraw.Draw(temp)
        for path in paths:
            draw.polygon(path, fill=int(255 * max(0.0, min(1.0, layer.opacity))))
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


def _scale_frame_value(value: float | int, scale: float, minimum: int = 0) -> int:
    return max(int(minimum), int(round(float(value) * float(scale))))


def _layer_rgba(color: list[int], alpha: float) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in color[:3]) + (max(0, min(255, int(round(max(0.0, min(1.0, alpha)) * 255.0)))),)


def _text_layer_visible(layer: TextOverlayLayer, frame_index: int) -> bool:
    if not layer.visible or layer.opacity <= 0.0:
        return False
    if int(frame_index) < int(layer.frame_start):
        return False
    if int(layer.frame_end) >= 0 and int(frame_index) > int(layer.frame_end):
        return False
    return True


def _text_panel_origin(
    layer: TextOverlayLayer,
    frame_size: tuple[int, int],
    panel_size: tuple[int, int],
) -> tuple[float, float]:
    frame_w, frame_h = frame_size
    panel_w, panel_h = panel_size
    ratio_x, ratio_y = _TEXT_ANCHOR_RATIOS.get(str(layer.anchor), (1.0, 0.0))
    anchor_x = ratio_x * frame_w + float(layer.offset_x) * frame_w
    anchor_y = ratio_y * frame_h + float(layer.offset_y) * frame_h
    return anchor_x - panel_w * ratio_x, anchor_y - panel_h * ratio_y


def _draw_text_overlays(
    frame_img: Image.Image,
    project: Project,
    frame_index: int,
    font_path: str | Path | None,
) -> None:
    if not project.text_layers:
        return

    frame_w, frame_h = frame_img.size
    scale = frame_h / 1080.0
    overlay = Image.new("RGBA", frame_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for layer in project.text_layers:
        if not _text_layer_visible(layer, frame_index):
            continue

        text = render_text_template(text_overlay_template_at_frame(layer, frame_index), project, frame_index).strip()
        if not text:
            continue

        font = _load_font(font_path, max(1, _scale_frame_value(layer.font_size, scale, minimum=1)))
        spacing = max(0, _scale_frame_value(6, scale))
        text_box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align=layer.alignment)
        text_w = max(0, int(round(text_box[2] - text_box[0])))
        text_h = max(0, int(round(text_box[3] - text_box[1])))
        pad_x = _scale_frame_value(layer.padding_x, scale)
        pad_y = _scale_frame_value(layer.padding_y, scale)
        panel_w = text_w + pad_x * 2
        panel_h = text_h + pad_y * 2
        panel_x, panel_y = _text_panel_origin(layer, (frame_w, frame_h), (panel_w, panel_h))
        panel_rect = [panel_x, panel_y, panel_x + panel_w, panel_y + panel_h]

        background_alpha = float(layer.opacity) * float(layer.background_opacity)
        if background_alpha > 0.0:
            draw.rounded_rectangle(
                panel_rect,
                radius=max(0, _scale_frame_value(layer.corner_radius, scale)),
                fill=_layer_rgba(layer.background_color, background_alpha),
            )

        border_width = _scale_frame_value(layer.border_width, scale)
        border_alpha = float(layer.opacity) * float(layer.border_opacity)
        if border_width > 0 and border_alpha > 0.0:
            draw.rounded_rectangle(
                panel_rect,
                radius=max(0, _scale_frame_value(layer.corner_radius, scale)),
                outline=_layer_rgba(layer.border_color, border_alpha),
                width=border_width,
            )

        text_x = panel_x + pad_x - float(text_box[0])
        text_y = panel_y + pad_y - float(text_box[1])
        draw.multiline_text(
            (text_x, text_y),
            text,
            font=font,
            fill=_layer_rgba(layer.color, float(layer.opacity)),
            spacing=spacing,
            align=layer.alignment,
        )

    composed = Image.alpha_composite(frame_img.convert("RGBA"), overlay)
    frame_img.paste(composed.convert("RGB"), (0, 0))


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
    _paste_wrapped_basemap(background, scaled_map, layout)

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
    _draw_text_overlays(frame_img, project, frame_index, font_path)
    return frame_img
