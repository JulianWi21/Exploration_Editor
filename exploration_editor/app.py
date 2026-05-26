from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

from PIL import Image
from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from exploration_editor.basemap import load_basemap_image, repo_root
from exploration_editor.export import export_frame_count, export_frame_png, export_video
from exploration_editor.geometry import (
    clamp,
    lonlat_to_world,
    polygon_edit_points_at_frame,
    resample_path,
    world_to_lonlat,
    unwrap_longitudes,
)
from exploration_editor.model import (
    DEFAULT_BASEMAP_PATH,
    POLYGON_EASING_LINEAR,
    PolygonKeyframe,
    PolygonLayer,
    Project,
    RouteLayer,
    default_project,
    load_project,
    save_project,
)
from exploration_editor.paths import exploration_examples_dir, exploration_exports_dir
from exploration_editor.render import clamp_view_state, compute_map_layout, polygon_outline_screen_paths_at_frame, render_frame


POLYGON_COLORS = [
    [232, 241, 255],
    [221, 236, 248],
    [210, 231, 245],
    [244, 241, 224],
]
ROUTE_COLORS = [
    [194, 228, 255],
    [156, 239, 173],
    [255, 196, 138],
    [255, 140, 140],
    [238, 180, 255],
]
VIDEO_FORMAT_PRESETS = [
    ("Full HD 1080p (1920x1080)", (1920, 1080)),
    ("QHD 1440p (2560x1440)", (2560, 1440)),
    ("4K UHD 2160p (3840x2160)", (3840, 2160)),
]
POLYGON_EASING_OPTIONS = [
    ("Linear", POLYGON_EASING_LINEAR),
    ("Ease In", "ease_in"),
    ("Ease Out", "ease_out"),
    ("Ease In-Out", "ease_in_out"),
]


def _pil_to_qimage(image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimage.copy()


def clamp_keyframe_frame(keyframes: list[int], index: int, target_frame: int, minimum: int, maximum: int) -> int:
    minimum_frame = int(minimum)
    maximum_frame = max(minimum_frame, int(maximum))
    lower_bound = minimum_frame
    upper_bound = maximum_frame

    if 0 <= index < len(keyframes):
        if index > 0:
            lower_bound = max(lower_bound, int(keyframes[index - 1]) + 1)
        if index + 1 < len(keyframes):
            upper_bound = min(upper_bound, int(keyframes[index + 1]) - 1)

    if lower_bound > upper_bound:
        lower_bound = upper_bound = max(minimum_frame, min(maximum_frame, int(target_frame)))

    return int(clamp(int(target_frame), lower_bound, upper_bound))


class PolygonKeyframeTrack(QWidget):
    frameSelected = pyqtSignal(int)
    keyframeMoved = pyqtSignal(int, int)
    keyframeMoveFinished = pyqtSignal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFixedHeight(30)
        self.setToolTip("Drag polygon keyframe markers to retime the interpolation between saved polygon states.")
        self._minimum = 0
        self._maximum = 0
        self._current_frame = 0
        self._keyframes: list[int] = []
        self._interactive = False
        self._message = "Select a polygon layer to edit its keyframes."
        self._drag_index: int | None = None
        self._drag_start_frame: int | None = None

    def set_state(
        self,
        *,
        minimum: int,
        maximum: int,
        current_frame: int,
        keyframes: list[int],
        interactive: bool,
        message: str,
    ) -> None:
        self._minimum = int(minimum)
        self._maximum = max(self._minimum, int(maximum))
        self._current_frame = int(clamp(int(current_frame), self._minimum, self._maximum))
        self._keyframes = [int(frame) for frame in keyframes]
        self._interactive = bool(interactive)
        self._message = str(message)
        if self._drag_index is not None and not (0 <= self._drag_index < len(self._keyframes)):
            self._drag_index = None
            self._drag_start_frame = None
        self.update()

    def _track_rect(self) -> QRectF:
        return QRectF(10.0, 7.0, max(1.0, self.width() - 20.0), max(1.0, self.height() - 14.0))

    def _frame_to_x(self, frame: int) -> float:
        rect = self._track_rect()
        if self._maximum <= self._minimum or rect.width() <= 1e-6:
            return rect.center().x()
        ratio = (clamp(int(frame), self._minimum, self._maximum) - self._minimum) / float(self._maximum - self._minimum)
        return rect.left() + ratio * rect.width()

    def _frame_from_x(self, x: float) -> int:
        rect = self._track_rect()
        if self._maximum <= self._minimum or rect.width() <= 1e-6:
            return self._minimum
        ratio = clamp((float(x) - rect.left()) / rect.width(), 0.0, 1.0)
        return int(round(self._minimum + ratio * (self._maximum - self._minimum)))

    def _marker_index_at(self, x: float, y: float) -> int | None:
        if not self._keyframes:
            return None
        rect = self._track_rect()
        if float(y) < rect.top() - 6.0 or float(y) > rect.bottom() + 6.0:
            return None
        best_index: int | None = None
        best_distance = 10.0
        for index, frame in enumerate(self._keyframes):
            distance = abs(self._frame_to_x(frame) - float(x))
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _update_hover_cursor(self, x: float, y: float) -> None:
        if self._interactive and self._marker_index_at(x, y) is not None:
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        else:
            self.unsetCursor()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        outer = QRectF(2.0, 2.0, max(1.0, self.width() - 4.0), max(1.0, self.height() - 4.0))
        painter.setPen(QPen(QColor(62, 74, 88), 1.0))
        painter.setBrush(QColor(20, 24, 31))
        painter.drawRoundedRect(outer, 6.0, 6.0)

        track_rect = self._track_rect()
        baseline_y = track_rect.center().y()
        painter.setPen(QPen(QColor(76, 87, 101), 1.0))
        painter.drawLine(QPointF(track_rect.left(), baseline_y), QPointF(track_rect.right(), baseline_y))

        if self._maximum >= self._minimum:
            current_x = self._frame_to_x(self._current_frame)
            painter.setPen(QPen(QColor(80, 152, 255, 180), 1.5))
            painter.drawLine(QPointF(current_x, track_rect.top() - 1.0), QPointF(current_x, track_rect.bottom() + 1.0))

        if not self._keyframes:
            painter.setPen(QColor(148, 160, 176))
            painter.drawText(outer.adjusted(10.0, 0.0, -10.0, 0.0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._message)
            return

        for index, frame in enumerate(self._keyframes):
            marker_x = self._frame_to_x(frame)
            is_current = int(frame) == int(self._current_frame)
            is_dragged = index == self._drag_index
            marker_color = QColor(255, 191, 102) if is_dragged else QColor(80, 152, 255) if is_current else QColor(228, 233, 240)
            marker_height = track_rect.height() if is_dragged else track_rect.height() * (0.92 if is_current else 0.72)
            marker_radius = 4.3 if is_dragged else 4.0 if is_current else 3.5
            top_y = baseline_y - marker_height * 0.5
            bottom_y = baseline_y + marker_height * 0.5

            painter.setPen(QPen(marker_color, 2.0))
            painter.drawLine(QPointF(marker_x, top_y), QPointF(marker_x, bottom_y))
            painter.setBrush(marker_color)
            painter.drawEllipse(QRectF(marker_x - marker_radius, baseline_y - marker_radius, marker_radius * 2.0, marker_radius * 2.0))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        if not self._interactive:
            return

        marker_index = self._marker_index_at(event.position().x(), event.position().y())
        if marker_index is None:
            self.frameSelected.emit(self._frame_from_x(event.position().x()))
            return

        self._drag_index = marker_index
        self._drag_start_frame = self._keyframes[marker_index]
        self.frameSelected.emit(self._keyframes[marker_index])
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_index is None:
            self._update_hover_cursor(event.position().x(), event.position().y())
            super().mouseMoveEvent(event)
            return

        target_frame = clamp_keyframe_frame(
            self._keyframes,
            self._drag_index,
            self._frame_from_x(event.position().x()),
            self._minimum,
            self._maximum,
        )
        if target_frame == self._keyframes[self._drag_index]:
            return

        self._keyframes[self._drag_index] = target_frame
        self._current_frame = target_frame
        self.keyframeMoved.emit(self._drag_index, target_frame)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_index is None:
            super().mouseReleaseEvent(event)
            return

        dragged_index = self._drag_index
        final_frame = self._keyframes[dragged_index]
        start_frame = self._drag_start_frame
        self._drag_index = None
        self._drag_start_frame = None
        self._update_hover_cursor(event.position().x(), event.position().y())
        self.update()

        if start_frame is not None and int(final_frame) != int(start_frame):
            self.keyframeMoveFinished.emit(dragged_index, final_frame)

    def leaveEvent(self, event) -> None:
        if self._drag_index is None:
            self.unsetCursor()
        super().leaveEvent(event)


class MapCanvas(QWidget):
    drawingFinished = pyqtSignal(object)
    viewChanged = pyqtSignal()
    polygonKeyframesChanged = pyqtSignal()

    def __init__(self, font_path: str | Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(960, 540)
        self.project: Project = default_project()
        self.project_path: str | None = None
        self.font_path = str(font_path) if font_path else None
        self.basemap_image = load_basemap_image(None)
        self._preview_basemap_levels = [self.basemap_image]
        self.frame_index = 0
        self.draw_kind: str | None = None
        self.target_layer_id: str | None = None
        self.selected_polygon_layer_id: str | None = None
        self.temp_points: list[list[float]] = []
        self.hover_point: list[float] | None = None
        self._dragging_vertex_index: int | None = None
        self._selected_vertex_index: int | None = None
        self._preserve_interpolated_overlay_during_drag = False
        self.show_edit_overlays = True
        self.is_panning = False
        self.last_mouse_pos: tuple[float, float] | None = None
        self._last_layout: dict[str, float] | None = None
        self._cached_qimage: QImage | None = None
        self._cached_key: tuple[object, ...] | None = None
        self._last_render_time = 0.0
        self.playback_active = False
        self._interactive_render_timer = QTimer(self)
        self._interactive_render_timer.setSingleShot(True)
        self._interactive_render_timer.timeout.connect(self._render_now)

    def sizeHint(self):
        return self.minimumSize()

    def set_document(self, project: Project, project_path: str | None = None) -> None:
        self.project = project
        self.project_path = project_path
        self.frame_index = min(self.frame_index, project.frame_count - 1)
        self.load_basemap(project.basemap_path)

    def load_basemap(self, basemap_path: str | None) -> None:
        self.basemap_image = load_basemap_image(basemap_path)
        self._preview_basemap_levels = self._build_preview_basemap_levels(self.basemap_image)
        self._constrain_view()
        self._render_now()

    def set_frame_index(self, frame_index: int) -> None:
        self.frame_index = max(0, min(project_frame_max(self.project), int(frame_index)))
        self._preserve_interpolated_overlay_during_drag = False
        self._render_now()

    def set_selected_polygon_layer(self, layer_id: str | None) -> None:
        self.selected_polygon_layer_id = layer_id
        self._dragging_vertex_index = None
        self._selected_vertex_index = None
        self._preserve_interpolated_overlay_during_drag = False
        self.update()

    def set_show_edit_overlays(self, show_edit_overlays: bool) -> None:
        self.show_edit_overlays = bool(show_edit_overlays)
        self._dragging_vertex_index = None
        self._selected_vertex_index = None
        self._preserve_interpolated_overlay_during_drag = False
        self.update()

    def invalidate_cache(self) -> None:
        self._interactive_render_timer.stop()
        self._cached_qimage = None
        self._cached_key = None

    def set_playback_active(self, playback_active: bool) -> None:
        playback_active = bool(playback_active)
        if self.playback_active == playback_active:
            return
        self.playback_active = playback_active
        self._render_now()

    def _build_preview_basemap_levels(self, image):
        levels = []
        for target_width in (512, 1024, 2048, 4096):
            if image.width <= target_width:
                continue
            target_height = max(1, int(round(image.height * (target_width / image.width))))
            levels.append(image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR))
        levels.append(image)
        return levels

    def _preview_render_size(self, frame_size: tuple[int, int]) -> tuple[int, int]:
        frame_w, frame_h = frame_size
        scale = 1.0
        if self.playback_active:
            scale = 0.5
        elif self.is_panning:
            scale = 0.6
        elif self._interactive_render_timer.isActive():
            scale = 0.75
        if scale >= 0.999:
            return frame_size
        return (
            min(frame_w, max(320, int(round(frame_w * scale)))),
            min(frame_h, max(180, int(round(frame_h * scale)))),
        )

    def _select_preview_basemap(self, layout: dict[str, float], preview_frame_size: tuple[int, int]):
        desired_width = max(256, int(round(preview_frame_size[0] * 1.2)))
        for image in self._preview_basemap_levels:
            if image.width >= desired_width:
                return image
        return self._preview_basemap_levels[-1]

    def _render_now(self) -> None:
        self.invalidate_cache()
        self.update()

    def _schedule_interactive_render(self) -> None:
        target_interval = 1.0 / 30.0
        elapsed = time.perf_counter() - self._last_render_time
        if elapsed >= target_interval and not self._interactive_render_timer.isActive():
            self._render_now()
            return
        delay_ms = max(1, int(round(max(0.0, target_interval - elapsed) * 1000.0)))
        self._interactive_render_timer.start(delay_ms)

    def _constrain_view(self) -> None:
        normalized = clamp_view_state(
            (max(2, self.width()), max(2, self.height())),
            self.basemap_image.size,
            self.project.view,
        )
        self.project.view.zoom = normalized.zoom
        self.project.view.offset_x = normalized.offset_x
        self.project.view.offset_y = normalized.offset_y

    def _selected_polygon_layer(self) -> PolygonLayer | None:
        if not self.selected_polygon_layer_id:
            return None
        for layer in self.project.polygon_layers:
            if layer.id == self.selected_polygon_layer_id:
                return layer
        return None

    def _find_polygon_keyframe(self, layer: PolygonLayer, frame: int) -> PolygonKeyframe | None:
        for keyframe in layer.keyframes:
            if int(keyframe.frame) == int(frame):
                return keyframe
        return None

    def _ensure_polygon_keyframe(self, layer: PolygonLayer, frame: int) -> PolygonKeyframe:
        existing = self._find_polygon_keyframe(layer, frame)
        if existing is not None:
            return existing
        keyframe = PolygonKeyframe(
            frame=int(frame),
            points=polygon_edit_points_at_frame(layer, int(frame)),
        )
        layer.keyframes.append(keyframe)
        layer.keyframes.sort(key=lambda item: item.frame)
        return keyframe

    def _editable_polygon_points(self) -> list[list[float]]:
        if self.draw_kind:
            return []
        layer = self._selected_polygon_layer()
        if layer is None or not layer.visible:
            return []
        exact = self._find_polygon_keyframe(layer, self.frame_index)
        if exact is not None:
            return exact.points
        return polygon_edit_points_at_frame(layer, self.frame_index)

    def _nearest_editable_vertex(self, x: float, y: float) -> int | None:
        if self._last_layout is None:
            return None
        points = self._editable_polygon_points()
        if len(points) < 3:
            return None
        tolerance_px = max(8.0, self.height() * 0.012)
        best_index: int | None = None
        best_dist_sq = tolerance_px * tolerance_px
        for path in self._screen_variants(points):
            for index, (sx, sy) in enumerate(path):
                dx = sx - x
                dy = sy - y
                dist_sq = dx * dx + dy * dy
                if dist_sq <= best_dist_sq:
                    best_dist_sq = dist_sq
                    best_index = index
        return best_index

    def _nearest_editable_edge(self, x: float, y: float) -> tuple[int, float] | None:
        if self._last_layout is None:
            return None
        raw_points = self._editable_polygon_points()
        n = len(raw_points)
        if n < 3:
            return None
        layer = self._selected_polygon_layer()
        tolerance_px = max(10.0, self.height() * 0.014)
        best_match: tuple[int, float] | None = None
        best_dist_sq = tolerance_px * tolerance_px

        smooth_paths = polygon_outline_screen_paths_at_frame(layer, self.frame_index, self._last_layout) if layer is not None else []
        raw_screen_paths = self._screen_variants(raw_points)
        radius = float(getattr(layer, "rounding_px", 0.0)) if layer is not None else 0.0
        s = max(6, min(24, int(radius / 15))) if radius > 0.0 else 1

        for smooth_path, raw_path in zip(smooth_paths, raw_screen_paths):
            m = len(smooth_path)
            if m == 0:
                continue
            for j in range(m):
                sx1, sy1 = smooth_path[j]
                sx2, sy2 = smooth_path[(j + 1) % m]
                delta_x = sx2 - sx1
                delta_y = sy2 - sy1
                seg_len_sq = delta_x * delta_x + delta_y * delta_y
                if seg_len_sq <= 1e-6:
                    continue
                t_proj = clamp(((x - sx1) * delta_x + (y - sy1) * delta_y) / seg_len_sq, 0.0, 1.0)
                proj_x = sx1 + delta_x * t_proj
                proj_y = sy1 + delta_y * t_proj
                dist_sq = (proj_x - x) * (proj_x - x) + (proj_y - y) * (proj_y - y)
                if dist_sq <= best_dist_sq:
                    best_dist_sq = dist_sq
                    raw_edge = (j // s) % n
                    rs_x1, rs_y1 = raw_path[raw_edge]
                    rs_x2, rs_y2 = raw_path[(raw_edge + 1) % n]
                    rdx = rs_x2 - rs_x1
                    rdy = rs_y2 - rs_y1
                    raw_seg_sq = rdx * rdx + rdy * rdy
                    if raw_seg_sq > 1e-6:
                        t_raw = clamp(((x - rs_x1) * rdx + (y - rs_y1) * rdy) / raw_seg_sq, 0.0, 1.0)
                    else:
                        t_raw = 0.0
                    best_match = (raw_edge, t_raw)

        # Fallback: no smooth paths available, test raw segments directly
        if not smooth_paths:
            for path in raw_screen_paths:
                for index, (start_x, start_y) in enumerate(path):
                    end_x, end_y = path[(index + 1) % len(path)]
                    delta_x = end_x - start_x
                    delta_y = end_y - start_y
                    seg_len_sq = delta_x * delta_x + delta_y * delta_y
                    if seg_len_sq <= 1e-6:
                        continue
                    t = clamp(((x - start_x) * delta_x + (y - start_y) * delta_y) / seg_len_sq, 0.0, 1.0)
                    proj_x = start_x + delta_x * t
                    proj_y = start_y + delta_y * t
                    dist_sq = (proj_x - x) * (proj_x - x) + (proj_y - y) * (proj_y - y)
                    if dist_sq <= best_dist_sq:
                        best_dist_sq = dist_sq
                        best_match = (index, float(t))
        return best_match

    def _interpolate_segment_point(self, start: list[float], end: list[float], t: float) -> list[float]:
        pair = unwrap_longitudes([start, end])
        lon = pair[0][0] * (1.0 - t) + pair[1][0] * t
        lat = float(start[1] * (1.0 - t) + end[1] * t)
        while lon < -180.0:
            lon += 360.0
        while lon > 180.0:
            lon -= 360.0
        return [float(lon), lat]

    def _sync_polygon_keyframe_topology(self, layer: PolygonLayer, point_count: int) -> None:
        target_count = max(3, int(point_count))
        for keyframe in layer.keyframes:
            if len(keyframe.points) != target_count:
                keyframe.points = resample_path(keyframe.points, target_count, closed=True)

    def _sync_polygon_keyframe_topology_from_frame(self, layer: PolygonLayer, point_count: int, frame: int) -> None:
        target_count = max(3, int(point_count))
        for keyframe in layer.keyframes:
            if int(keyframe.frame) < int(frame):
                continue
            if len(keyframe.points) != target_count:
                keyframe.points = resample_path(keyframe.points, target_count, closed=True)

    def insert_selected_polygon_vertex(self, x: float, y: float) -> bool:
        if self.draw_kind or not self.show_edit_overlays:
            return False
        if self._nearest_editable_vertex(x, y) is not None:
            return False
        edge_match = self._nearest_editable_edge(x, y)
        layer = self._selected_polygon_layer()
        point = self._screen_to_lonlat(x, y)
        if edge_match is None or layer is None or point is None:
            return False

        keyframe = self._ensure_polygon_keyframe(layer, self.frame_index)
        if len(keyframe.points) < 3:
            return False
        self._sync_polygon_keyframe_topology_from_frame(layer, len(keyframe.points), self.frame_index)

        edge_index, edge_t = edge_match
        for polygon_keyframe in layer.keyframes:
            if int(polygon_keyframe.frame) < int(self.frame_index):
                continue
            points = polygon_keyframe.points
            if len(points) < 3:
                continue
            next_index = (edge_index + 1) % len(points)
            insert_index = len(points) if next_index == 0 else next_index
            insert_point = point[:] if polygon_keyframe is keyframe else self._interpolate_segment_point(points[edge_index], points[next_index], edge_t)
            polygon_keyframe.points.insert(insert_index, insert_point)

        self._selected_vertex_index = len(keyframe.points) - 1 if (edge_index + 1) % len(keyframe.points) == 0 else edge_index + 1
        self._dragging_vertex_index = None
        self._render_now()
        self.polygonKeyframesChanged.emit()
        return True

    def delete_selected_vertex(self) -> bool:
        if self.draw_kind or not self.show_edit_overlays:
            return False
        layer = self._selected_polygon_layer()
        if layer is None or self._selected_vertex_index is None:
            return False
        keyframe = self._ensure_polygon_keyframe(layer, self.frame_index)
        if len(keyframe.points) <= 3 or not (0 <= self._selected_vertex_index < len(keyframe.points)):
            return False

        self._sync_polygon_keyframe_topology_from_frame(layer, len(keyframe.points), self.frame_index)
        delete_index = self._selected_vertex_index
        for polygon_keyframe in layer.keyframes:
            if int(polygon_keyframe.frame) < int(self.frame_index):
                continue
            if len(polygon_keyframe.points) > 3 and delete_index < len(polygon_keyframe.points):
                polygon_keyframe.points.pop(delete_index)

        if len(keyframe.points) > 3:
            self._selected_vertex_index = min(delete_index, len(keyframe.points) - 1)
        else:
            self._selected_vertex_index = None
        self._dragging_vertex_index = None
        self._render_now()
        self.polygonKeyframesChanged.emit()
        return True

    def begin_polygon_draw(self, target_layer_id: str | None = None) -> None:
        self.draw_kind = "polygon"
        self.target_layer_id = target_layer_id
        self.temp_points = []
        self.hover_point = None
        self._selected_vertex_index = None
        self.update()

    def begin_route_draw(self) -> None:
        self.draw_kind = "route"
        self.target_layer_id = None
        self.temp_points = []
        self.hover_point = None
        self._selected_vertex_index = None
        self.update()

    def cancel_drawing(self) -> None:
        self.draw_kind = None
        self.target_layer_id = None
        self.temp_points = []
        self.hover_point = None
        self._dragging_vertex_index = None
        self._preserve_interpolated_overlay_during_drag = False
        self.update()

    def paintEvent(self, _event) -> None:
        frame_size = (max(2, self.width()), max(2, self.height()))
        layout = compute_map_layout(frame_size, self.basemap_image.size, self.project.view)
        freeze_preview = self._dragging_vertex_index is not None and self._cached_qimage is not None
        preview_frame_size = self._preview_render_size(frame_size)
        render_project = self.project
        if preview_frame_size != frame_size:
            scale_x = preview_frame_size[0] / float(frame_size[0])
            scale_y = preview_frame_size[1] / float(frame_size[1])
            render_project = replace(
                self.project,
                view=replace(
                    self.project.view,
                    offset_x=float(self.project.view.offset_x) * scale_x,
                    offset_y=float(self.project.view.offset_y) * scale_y,
                ),
            )
        preview_layout = compute_map_layout(preview_frame_size, self.basemap_image.size, render_project.view)
        preview_basemap = self._select_preview_basemap(preview_layout, preview_frame_size)
        key = (
            frame_size,
            preview_frame_size,
            self.frame_index,
            round(self.project.view.zoom, 4),
            round(self.project.view.offset_x, 2),
            round(self.project.view.offset_y, 2),
            self.project.title,
            len(self.project.polygon_layers),
            len(self.project.route_layers),
            tuple((layer.id, layer.visible, len(layer.keyframes)) for layer in self.project.polygon_layers),
            tuple((layer.id, layer.visible, layer.start_frame, layer.end_frame, len(layer.points)) for layer in self.project.route_layers),
            self.basemap_image.size,
            preview_basemap.size,
        )
        if not freeze_preview and (self._cached_qimage is None or self._cached_key != key):
            image = render_frame(
                render_project,
                basemap_image=self.basemap_image,
                display_basemap_image=preview_basemap,
                frame_index=self.frame_index,
                output_size=preview_frame_size,
                font_path=self.font_path,
                preview=True,
            )
            self._cached_qimage = _pil_to_qimage(image)
            self._cached_key = key
            self._last_render_time = time.perf_counter()
        self._last_layout = layout

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._cached_qimage is not None:
            painter.drawImage(self.rect(), self._cached_qimage)
        if self.show_edit_overlays:
            self._draw_selected_polygon_overlay(painter)
            self._draw_temp_overlay(painter)
        painter.end()

    def _draw_selected_polygon_overlay(self, painter: QPainter) -> None:
        if self.draw_kind or self._last_layout is None:
            return
        layer = self._selected_polygon_layer()
        points = self._editable_polygon_points()
        if layer is None or len(points) < 3:
            return

        is_exact_keyframe = self._find_polygon_keyframe(layer, self.frame_index) is not None
        show_exact_keyframe = is_exact_keyframe and not (
            self._dragging_vertex_index is not None and self._preserve_interpolated_overlay_during_drag
        )
        outline = QColor(layer.color[0], layer.color[1], layer.color[2], 235)
        handle_outer = QColor(6, 8, 12, 220)
        handle_inner = QColor(245, 248, 252, 235)
        selected_handle = QColor(88, 170, 255, 245)
        active_handle = QColor(255, 208, 84, 245)

        pen = QPen(outline)
        pen.setWidth(3)
        if not show_exact_keyframe:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for outline_path in polygon_outline_screen_paths_at_frame(layer, self.frame_index, self._last_layout):
            polygon = QPolygonF([QPointF(x, y) for x, y in outline_path])
            painter.drawPolygon(polygon)

        for path in self._screen_variants(points):
            for index, (x, y) in enumerate(path):
                painter.setPen(Qt.PenStyle.NoPen)
                if index == self._dragging_vertex_index:
                    painter.setBrush(active_handle)
                elif index == self._selected_vertex_index:
                    painter.setBrush(selected_handle)
                else:
                    painter.setBrush(handle_outer)
                painter.drawEllipse(QPointF(x, y), 5.5, 5.5)
                painter.setBrush(handle_inner)
                painter.drawEllipse(QPointF(x, y), 2.9, 2.9)
                painter.setPen(pen)

    def _draw_temp_overlay(self, painter: QPainter) -> None:
        if not self.draw_kind or not self.temp_points or self._last_layout is None:
            return
        pen = QPen(QColor(255, 255, 255, 230))
        pen.setWidth(3)
        painter.setPen(pen)
        for path in self._screen_variants(self._preview_points()):
            polygon = QPolygonF([QPointF(x, y) for x, y in path])
            if self.draw_kind == "polygon" and len(path) >= 3:
                painter.drawPolygon(polygon)
            else:
                painter.drawPolyline(polygon)
            for x, y in path:
                painter.setBrush(QColor(8, 10, 14, 220))
                painter.drawEllipse(QPointF(x, y), 4.5, 4.5)
                painter.setBrush(QColor(255, 255, 255, 230))
                painter.drawEllipse(QPointF(x, y), 2.5, 2.5)

    def _preview_points(self) -> list[list[float]]:
        points = [point[:] for point in self.temp_points]
        if self.hover_point is not None:
            points.append(self.hover_point[:])
        return points

    def _screen_variants(self, points: list[list[float]]) -> list[list[tuple[float, float]]]:
        if self._last_layout is None:
            return []
        layout = self._last_layout
        world_size = self.basemap_image.size
        continuous = unwrap_longitudes(points)
        world_points = [lonlat_to_world(lon, lat, world_size, wrap=False) for lon, lat in continuous]
        variants: list[list[tuple[float, float]]] = []
        for shift in (-world_size[0], 0.0, world_size[0]):
            variants.append([
                (
                    layout["offset_x"] + (x + shift) * layout["scale"],
                    layout["offset_y"] + y * layout["scale"],
                )
                for x, y in world_points
            ])
        return variants

    def _screen_to_lonlat(self, x: float, y: float) -> list[float] | None:
        layout = compute_map_layout((max(2, self.width()), max(2, self.height())), self.basemap_image.size, self.project.view)
        world_x = (x - layout["offset_x"]) / layout["scale"]
        world_y = (y - layout["offset_y"]) / layout["scale"]
        if world_y < 0.0 or world_y > self.basemap_image.height:
            return None
        world_x = world_x % float(self.basemap_image.width)
        return world_to_lonlat(world_x, world_y, self.basemap_image.size)

    def _zoom_at(self, x: float, y: float, factor: float) -> None:
        before = self._screen_to_lonlat(x, y)
        old_zoom = self.project.view.zoom
        self.project.view.zoom = clamp(old_zoom * factor, 1.0, 14.0)
        if before is not None:
            new_layout = compute_map_layout((max(2, self.width()), max(2, self.height())), self.basemap_image.size, self.project.view)
            world_x, world_y = lonlat_to_world(before[0], before[1], self.basemap_image.size)
            screen_x = new_layout["offset_x"] + world_x * new_layout["scale"]
            screen_y = new_layout["offset_y"] + world_y * new_layout["scale"]
            self.project.view.offset_x += x - screen_x
            self.project.view.offset_y += y - screen_y
        self._constrain_view()
        self._schedule_interactive_render()

    def _finish_drawing(self) -> None:
        if self.draw_kind == "polygon" and len(self.temp_points) >= 3:
            payload = {
                "kind": "polygon",
                "target_layer_id": self.target_layer_id,
                "points": [point[:] for point in self.temp_points],
            }
            self.drawingFinished.emit(payload)
        elif self.draw_kind == "route" and len(self.temp_points) >= 2:
            payload = {
                "kind": "route",
                "points": [point[:] for point in self.temp_points],
            }
            self.drawingFinished.emit(payload)
        self.cancel_drawing()

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        if not self.show_edit_overlays:
            if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton}:
                self.is_panning = True
                self.last_mouse_pos = (pos.x(), pos.y())
            return
        if self.draw_kind and event.button() == Qt.MouseButton.LeftButton:
            point = self._screen_to_lonlat(pos.x(), pos.y())
            if point is not None:
                self.temp_points.append(point)
                self.hover_point = point
                self.update()
            return
        if self.draw_kind and event.button() == Qt.MouseButton.RightButton:
            if self.temp_points:
                self._finish_drawing()
            else:
                self.cancel_drawing()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            handle_index = self._nearest_editable_vertex(pos.x(), pos.y())
            if handle_index is not None:
                layer = self._selected_polygon_layer()
                if layer is not None:
                    had_exact_keyframe = self._find_polygon_keyframe(layer, self.frame_index) is not None
                    keyframe = self._ensure_polygon_keyframe(layer, self.frame_index)
                    if 0 <= handle_index < len(keyframe.points):
                        self._selected_vertex_index = handle_index
                        self._dragging_vertex_index = handle_index
                        self._preserve_interpolated_overlay_during_drag = not had_exact_keyframe
                        self._interactive_render_timer.stop()
                        point = self._screen_to_lonlat(pos.x(), pos.y())
                        if point is not None:
                            keyframe.points[handle_index] = point
                        self.update()
                        return
            self._selected_vertex_index = None
            self._preserve_interpolated_overlay_during_drag = False
        if event.button() in {Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton}:
            self.is_panning = True
            self.last_mouse_pos = (pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if not self.show_edit_overlays:
            if self.is_panning and self.last_mouse_pos is not None:
                dx = pos.x() - self.last_mouse_pos[0]
                dy = pos.y() - self.last_mouse_pos[1]
                self.project.view.offset_x += dx
                self.project.view.offset_y += dy
                self._constrain_view()
                self.last_mouse_pos = (pos.x(), pos.y())
                self._schedule_interactive_render()
            return
        if self.draw_kind:
            self.hover_point = self._screen_to_lonlat(pos.x(), pos.y())
            self.update()
            return
        if self._dragging_vertex_index is not None:
            point = self._screen_to_lonlat(pos.x(), pos.y())
            layer = self._selected_polygon_layer()
            if point is not None and layer is not None:
                keyframe = self._ensure_polygon_keyframe(layer, self.frame_index)
                if 0 <= self._dragging_vertex_index < len(keyframe.points):
                    keyframe.points[self._dragging_vertex_index] = point
                    self.update()
            return
        if self.is_panning and self.last_mouse_pos is not None:
            dx = pos.x() - self.last_mouse_pos[0]
            dy = pos.y() - self.last_mouse_pos[1]
            self.project.view.offset_x += dx
            self.project.view.offset_y += dy
            self._constrain_view()
            self.last_mouse_pos = (pos.x(), pos.y())
            self._schedule_interactive_render()

    def mouseReleaseEvent(self, _event) -> None:
        if self._dragging_vertex_index is not None:
            self._dragging_vertex_index = None
            self._preserve_interpolated_overlay_during_drag = False
            self._render_now()
            self.polygonKeyframesChanged.emit()
        self.is_panning = False
        self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event) -> None:
        if not self.draw_kind and event.button() == Qt.MouseButton.LeftButton:
            if self.insert_selected_polygon_vertex(event.position().x(), event.position().y()):
                return
        if not self.draw_kind:
            return
        point = self._screen_to_lonlat(event.position().x(), event.position().y())
        if point is not None and (not self.temp_points or point != self.temp_points[-1]):
            self.temp_points.append(point)
        self._finish_drawing()

    def wheelEvent(self, event) -> None:
        factor = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self._zoom_at(event.position().x(), event.position().y(), factor)

    def resizeEvent(self, event) -> None:
        self._constrain_view()
        self._render_now()
        super().resizeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, initial_project_path: str | None = None) -> None:
        super().__init__()
        self.repo_root = repo_root()
        self.examples_root = exploration_examples_dir()
        self.exports_root = exploration_exports_dir()
        self.examples_root.mkdir(parents=True, exist_ok=True)
        self.exports_root.mkdir(parents=True, exist_ok=True)
        self.font_path = self.repo_root / "fonts" / "Montserrat-ExtraBold.ttf"
        self.project_path: str | None = None
        self.project = default_project(str(self.repo_root / DEFAULT_BASEMAP_PATH))
        self._playback_start_time: float | None = None
        self._playback_start_frame = 0
        self._updating_form = False
        self._build_ui()
        self._connect_signals()
        self._apply_dark_palette()
        self._reset_project(self.project, None)
        if initial_project_path:
            self._load_project_from_path(initial_project_path)

    def _build_ui(self) -> None:
        self.setWindowTitle("Exploration_Editor")
        self.resize(1680, 960)

        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.new_project_button = QPushButton("New")
        self.open_project_button = QPushButton("Open")
        self.save_project_button = QPushButton("Save")
        self.save_project_as_button = QPushButton("Save As")
        self.open_basemap_button = QPushButton("Open Basemap")
        self.new_polygon_button = QPushButton("New Polygon")
        self.add_keyframe_button = QPushButton("Add Polygon Keyframe")
        self.insert_keyframe_button = QPushButton("Insert Keyframe")
        self.insert_keyframe_button.setToolTip("Insert a new keyframe at the current frame, initialised from the interpolated polygon state.")
        self.delete_keyframe_button = QPushButton("Delete Keyframe")
        self.delete_keyframe_button.setToolTip("Delete the keyframe that is exactly at the current frame.")
        self.new_route_button = QPushButton("New Route")
        self.delete_layer_button = QPushButton("Delete Layer")
        self.export_png_button = QPushButton("Export PNG")
        self.export_video_button = QPushButton("Export MP4")
        self.play_button = QPushButton("Play")
        self.final_preview_button = QPushButton("Final Preview")
        self.final_preview_button.setCheckable(True)
        self.final_preview_button.setToolTip("Hide edit handles and show only the rendered fog of war. Shortcut: H")
        for button in [
            self.new_project_button,
            self.open_project_button,
            self.save_project_button,
            self.save_project_as_button,
            self.open_basemap_button,
            self.new_polygon_button,
            self.add_keyframe_button,
            self.insert_keyframe_button,
            self.delete_keyframe_button,
            self.new_route_button,
            self.delete_layer_button,
            self.export_png_button,
            self.export_video_button,
            self.play_button,
            self.final_preview_button,
        ]:
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        root_layout.addLayout(toolbar)

        splitter = QSplitter()
        self.canvas = MapCanvas(font_path=self.font_path)
        splitter.addWidget(self.canvas)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)

        project_box = QWidget()
        project_form = QFormLayout(project_box)
        project_form.setContentsMargins(0, 0, 0, 0)
        self.title_edit = QLineEdit()
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(2.0, 300.0)
        self.duration_spin.setSingleStep(1.0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(12, 60)
        self.fog_spin = QDoubleSpinBox()
        self.fog_spin.setRange(0.0, 1.0)
        self.fog_spin.setSingleStep(0.02)
        self.video_format_combo = QComboBox()
        self.basemap_combo = QComboBox()
        project_form.addRow("Title", self.title_edit)
        project_form.addRow("Duration", self.duration_spin)
        project_form.addRow("FPS", self.fps_spin)
        project_form.addRow("Fog Opacity", self.fog_spin)
        project_form.addRow("Video Format", self.video_format_combo)
        project_form.addRow("Basemap", self.basemap_combo)
        side_layout.addWidget(project_box)

        side_layout.addWidget(QLabel("Layers"))
        self.layer_list = QListWidget()
        side_layout.addWidget(self.layer_list, 1)

        props_box = QWidget()
        props_form = QFormLayout(props_box)
        props_form.setContentsMargins(0, 0, 0, 0)
        self.layer_name_edit = QLineEdit()
        self.layer_visible_check = QCheckBox()
        self.color_button = QPushButton("Pick Color")
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 240)
        self.rounding_spin = QSpinBox()
        self.rounding_spin.setRange(0, 240)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.05, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.polygon_constant_area_check = QCheckBox()
        self.polygon_constant_area_check.setToolTip(
            "Equalize discovered polygon area across time for the segment from the current exact keyframe to the next one."
        )
        self.polygon_easing_combo = QComboBox()
        for label, value in POLYGON_EASING_OPTIONS:
            self.polygon_easing_combo.addItem(label, value)
        self.polygon_easing_combo.setToolTip("Controls the timing curve from the exact polygon keyframe at the current frame to the next polygon keyframe.")
        self.route_width_spin = QSpinBox()
        self.route_width_spin.setRange(1, 60)
        self.route_reveal_spin = QSpinBox()
        self.route_reveal_spin.setRange(4, 160)
        self.route_start_spin = QSpinBox()
        self.route_end_spin = QSpinBox()
        self.route_label_edit = QLineEdit()
        self.route_legend_check = QCheckBox()
        props_form.addRow("Layer Name", self.layer_name_edit)
        props_form.addRow("Visible", self.layer_visible_check)
        props_form.addRow("Color", self.color_button)
        props_form.addRow("Feather", self.feather_spin)
        props_form.addRow("Rounding", self.rounding_spin)
        props_form.addRow("Opacity", self.opacity_spin)
        props_form.addRow("Constant Area", self.polygon_constant_area_check)
        props_form.addRow("Next Segment", self.polygon_easing_combo)
        props_form.addRow("Route Width", self.route_width_spin)
        props_form.addRow("Reveal Width", self.route_reveal_spin)
        props_form.addRow("Start Frame", self.route_start_spin)
        props_form.addRow("End Frame", self.route_end_spin)
        props_form.addRow("Legend Label", self.route_label_edit)
        props_form.addRow("Show Legend", self.route_legend_check)
        side_layout.addWidget(props_box)

        splitter.addWidget(side_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root_layout.addWidget(splitter, 1)

        label_width = max(QLabel("Keyframes").sizeHint().width(), QLabel("Timeline").sizeHint().width()) + 8

        keyframe_timeline = QHBoxLayout()
        self.keyframe_track_label = QLabel("Keyframes")
        self.keyframe_track_label.setFixedWidth(label_width)
        self.keyframe_track = PolygonKeyframeTrack()
        keyframe_timeline.addWidget(self.keyframe_track_label)
        keyframe_timeline.addWidget(self.keyframe_track, 1)
        root_layout.addLayout(keyframe_timeline)

        timeline = QHBoxLayout()
        timeline_title = QLabel("Timeline")
        timeline_title.setFixedWidth(label_width)
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_label = QLabel("Frame 0 / 0")
        timeline.addWidget(timeline_title)
        timeline.addWidget(self.timeline_slider, 1)
        timeline.addWidget(self.timeline_label)
        root_layout.addLayout(timeline)

        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.new_project_button.clicked.connect(self._new_project)
        self.open_project_button.clicked.connect(self._open_project)
        self.save_project_button.clicked.connect(self._save_project)
        self.save_project_as_button.clicked.connect(self._save_project_as)
        self.open_basemap_button.clicked.connect(self._open_basemap)
        self.new_polygon_button.clicked.connect(lambda: self.canvas.begin_polygon_draw(None))
        self.add_keyframe_button.clicked.connect(self._begin_polygon_keyframe)
        self.insert_keyframe_button.clicked.connect(self._insert_current_keyframe)
        self.delete_keyframe_button.clicked.connect(self._delete_current_keyframe)
        self.new_route_button.clicked.connect(self.canvas.begin_route_draw)
        self.delete_layer_button.clicked.connect(self._delete_selected_layer)
        self.export_png_button.clicked.connect(self._export_png)
        self.export_video_button.clicked.connect(self._export_video)
        self.play_button.clicked.connect(self._toggle_play)
        self.final_preview_button.toggled.connect(self._set_final_preview_mode)
        self.timeline_slider.valueChanged.connect(self._on_frame_changed)
        self.keyframe_track.frameSelected.connect(self._set_current_frame)
        self.keyframe_track.keyframeMoved.connect(self._move_selected_polygon_keyframe)
        self.keyframe_track.keyframeMoveFinished.connect(self._announce_keyframe_move)
        self.layer_list.currentRowChanged.connect(self._populate_layer_form)
        self.color_button.clicked.connect(self._pick_color)
        self.layer_name_edit.editingFinished.connect(self._apply_layer_form)
        self.layer_visible_check.stateChanged.connect(self._apply_layer_form)
        self.feather_spin.valueChanged.connect(self._apply_layer_form)
        self.rounding_spin.valueChanged.connect(self._apply_layer_form)
        self.opacity_spin.valueChanged.connect(self._apply_layer_form)
        self.polygon_constant_area_check.stateChanged.connect(self._apply_polygon_constant_area)
        self.polygon_easing_combo.currentIndexChanged.connect(self._apply_polygon_easing)
        self.route_width_spin.valueChanged.connect(self._apply_layer_form)
        self.route_reveal_spin.valueChanged.connect(self._apply_layer_form)
        self.route_start_spin.valueChanged.connect(self._apply_layer_form)
        self.route_end_spin.valueChanged.connect(self._apply_layer_form)
        self.route_label_edit.editingFinished.connect(self._apply_layer_form)
        self.route_legend_check.stateChanged.connect(self._apply_layer_form)
        self.title_edit.editingFinished.connect(self._apply_project_form)
        self.duration_spin.valueChanged.connect(self._apply_project_form)
        self.fps_spin.valueChanged.connect(self._apply_project_form)
        self.fog_spin.valueChanged.connect(self._apply_project_form)
        self.video_format_combo.currentIndexChanged.connect(self._apply_video_format_selection)
        self.basemap_combo.currentIndexChanged.connect(self._apply_basemap_selection)
        self.canvas.drawingFinished.connect(self._handle_drawing_finished)
        self.canvas.polygonKeyframesChanged.connect(self._refresh_keyframe_track)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)

    def _set_final_preview_mode(self, enabled: bool) -> None:
        self.final_preview_button.blockSignals(True)
        self.final_preview_button.setChecked(bool(enabled))
        self.final_preview_button.blockSignals(False)
        self.canvas.set_show_edit_overlays(not enabled)
        mode_text = "final preview" if enabled else "edit overlay"
        self.statusBar().showMessage(f"Viewer mode: {mode_text}", 2500)

    def _refresh_video_format_selector(self) -> None:
        current_size = (int(self.project.width), int(self.project.height))
        self.video_format_combo.blockSignals(True)
        self.video_format_combo.clear()

        current_index = -1
        for index, (label, size) in enumerate(VIDEO_FORMAT_PRESETS):
            self.video_format_combo.addItem(label, size)
            if size == current_size:
                current_index = index

        if current_index < 0:
            self.video_format_combo.addItem(f"Custom ({current_size[0]}x{current_size[1]})", current_size)
            current_index = self.video_format_combo.count() - 1

        self.video_format_combo.setCurrentIndex(current_index)
        self.video_format_combo.blockSignals(False)

    def _apply_video_format_selection(self, _index: int) -> None:
        if self._updating_form:
            return
        size = self.video_format_combo.currentData()
        if not size:
            return
        width, height = int(size[0]), int(size[1])
        if width == self.project.width and height == self.project.height:
            return
        self.project.width = width
        self.project.height = height
        self.statusBar().showMessage(f"Video format set: {width}x{height}", 2500)

    def _available_basemap_paths(self) -> list[Path]:
        basemap_dir = self.repo_root / "data" / "basemaps"
        if not basemap_dir.exists():
            return []
        return sorted(
            [
                path.resolve()
                for path in basemap_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ],
            key=lambda item: item.name.lower(),
        )

    def _refresh_basemap_selector(self) -> None:
        current_path = Path(self.project.basemap_path).resolve() if self.project.basemap_path else None
        available_paths = self._available_basemap_paths()

        self.basemap_combo.blockSignals(True)
        self.basemap_combo.clear()

        current_index = -1
        for index, path in enumerate(available_paths):
            self.basemap_combo.addItem(path.name, str(path))
            if current_path is not None and path == current_path:
                current_index = index

        if current_path is not None and current_index < 0:
            prefix = "Missing" if not current_path.exists() else "Custom"
            self.basemap_combo.addItem(f"{prefix}: {current_path.name}", str(current_path))
            current_index = self.basemap_combo.count() - 1

        if self.basemap_combo.count() > 0:
            self.basemap_combo.setCurrentIndex(current_index if current_index >= 0 else 0)
        self.basemap_combo.setEnabled(self.basemap_combo.count() > 0)
        self.basemap_combo.blockSignals(False)

    def _set_basemap_path(self, basemap_path: str | Path) -> None:
        normalized = str(Path(basemap_path).resolve())
        changed = normalized != self.project.basemap_path
        self.project.basemap_path = normalized
        self._refresh_basemap_selector()
        if changed:
            self.canvas.load_basemap(normalized)
            self.statusBar().showMessage(f"Basemap loaded: {Path(normalized).name}", 2500)

    def _apply_basemap_selection(self, _index: int) -> None:
        if self._updating_form:
            return
        selected_path = self.basemap_combo.currentData()
        if selected_path:
            self._set_basemap_path(str(selected_path))

    def _apply_dark_palette(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(18, 22, 28))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(233, 237, 242))
        palette.setColor(QPalette.ColorRole.Base, QColor(12, 16, 22))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(24, 30, 38))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(233, 237, 242))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(18, 22, 28))
        palette.setColor(QPalette.ColorRole.Text, QColor(233, 237, 242))
        palette.setColor(QPalette.ColorRole.Button, QColor(30, 36, 45))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(233, 237, 242))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(70, 120, 195))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        QApplication.instance().setPalette(palette)

    def _reset_project(self, project: Project, project_path: str | None) -> None:
        self.project = project
        self.project_path = project_path
        self.canvas.set_document(project, project_path=project_path)
        self._populate_project_form()
        self._refresh_layer_list()
        self._refresh_timeline()
        self._populate_layer_form()

    def _new_project(self) -> None:
        basemap = str(self.repo_root / DEFAULT_BASEMAP_PATH)
        self._reset_project(default_project(basemap), None)

    def _open_project(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open Project", str(self.examples_root), "JSON Files (*.json)")
        if path:
            self._load_project_from_path(path)

    def _load_project_from_path(self, path: str) -> None:
        project = load_project(path)
        basemap_path = Path(project.basemap_path)
        if not basemap_path.is_absolute():
            project.basemap_path = str((Path(path).resolve().parent / basemap_path).resolve())
        loaded_path = Path(path).resolve()
        bundled_examples_dir = (self.repo_root / "examples").resolve()
        project_path = None if bundled_examples_dir in loaded_path.parents else path
        self._reset_project(project, project_path)

    def _save_project(self) -> None:
        if self.project_path is None:
            default_name = self.project.title.lower().replace(" ", "_") or "exploration_project"
            path, _filter = QFileDialog.getSaveFileName(self, "Save Project", str(self.examples_root / f"{default_name}.json"), "JSON Files (*.json)")
            if not path:
                return
            self.project_path = path
        save_project(self.project, self.project_path)
        self.statusBar().showMessage(f"Project saved: {self.project_path}", 4000)

    def _save_project_as(self) -> None:
        default_name = self.project.title.lower().replace(" ", "_") or "exploration_project"
        default_path = Path(self.project_path) if self.project_path else self.examples_root / f"{default_name}.json"
        path, _filter = QFileDialog.getSaveFileName(self, "Save Project As", str(default_path), "JSON Files (*.json)")
        if not path:
            return
        self.project_path = path
        save_project(self.project, self.project_path)
        self.statusBar().showMessage(f"Project saved as: {self.project_path}", 4000)

    def _open_basemap(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "Open Basemap", str(self.repo_root / "data" / "basemaps"), "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        self._set_basemap_path(path)

    def _refresh_layer_list(self) -> None:
        self.layer_list.blockSignals(True)
        current_key = self._selected_layer_key()
        self.layer_list.clear()
        for layer in self.project.polygon_layers:
            item = QListWidgetItem(f"[P] {layer.name}")
            item.setData(Qt.ItemDataRole.UserRole, ("polygon", layer.id))
            self.layer_list.addItem(item)
        for layer in self.project.route_layers:
            item = QListWidgetItem(f"[R] {layer.name}")
            item.setData(Qt.ItemDataRole.UserRole, ("route", layer.id))
            self.layer_list.addItem(item)
        if current_key is not None:
            for index in range(self.layer_list.count()):
                item = self.layer_list.item(index)
                if item.data(Qt.ItemDataRole.UserRole) == current_key:
                    self.layer_list.setCurrentRow(index)
                    break
        elif self.layer_list.count() > 0:
            self.layer_list.setCurrentRow(0)
        self.layer_list.blockSignals(False)

    def _selected_layer_key(self):
        item = self.layer_list.currentItem()
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _selected_layer(self):
        key = self._selected_layer_key()
        if key is None:
            return None, None
        kind, layer_id = key
        if kind == "polygon":
            for layer in self.project.polygon_layers:
                if layer.id == layer_id:
                    return kind, layer
        if kind == "route":
            for layer in self.project.route_layers:
                if layer.id == layer_id:
                    return kind, layer
        return None, None

    def _populate_project_form(self) -> None:
        self._updating_form = True
        self.title_edit.setText(self.project.title)
        self.duration_spin.setValue(float(self.project.duration_sec))
        self.fps_spin.setValue(int(self.project.fps))
        self.fog_spin.setValue(float(self.project.fog_opacity))
        self._refresh_video_format_selector()
        self._refresh_basemap_selector()
        self._updating_form = False

    def _apply_project_form(self) -> None:
        if self._updating_form:
            return
        self.project.title = self.title_edit.text().strip() or "Untitled Exploration"
        self.project.duration_sec = float(self.duration_spin.value())
        self.project.fps = int(self.fps_spin.value())
        self.project.fog_opacity = float(self.fog_spin.value())
        self._refresh_timeline()
        self._refresh_canvas_only()

    def _populate_layer_form(self) -> None:
        kind, layer = self._selected_layer()
        self.canvas.set_selected_polygon_layer(layer.id if kind == "polygon" and layer is not None else None)
        self._refresh_keyframe_track()
        self._updating_form = True
        enabled = layer is not None
        for widget in [
            self.layer_name_edit,
            self.layer_visible_check,
            self.color_button,
            self.feather_spin,
            self.rounding_spin,
            self.opacity_spin,
            self.polygon_constant_area_check,
            self.polygon_easing_combo,
            self.route_width_spin,
            self.route_reveal_spin,
            self.route_start_spin,
            self.route_end_spin,
            self.route_label_edit,
            self.route_legend_check,
        ]:
            widget.setEnabled(enabled)
        if layer is None:
            self.layer_name_edit.setText("")
            self._set_color_button([255, 255, 255])
            self._populate_polygon_easing_form(None, None, False)
            self._updating_form = False
            return
        self.layer_name_edit.setText(layer.name)
        self.layer_visible_check.setChecked(layer.visible)
        self._set_color_button(layer.color)
        is_polygon = kind == "polygon"
        self.feather_spin.setEnabled(is_polygon)
        self.rounding_spin.setEnabled(is_polygon)
        self.opacity_spin.setEnabled(is_polygon)
        self.polygon_constant_area_check.setEnabled(is_polygon)
        self.polygon_easing_combo.setEnabled(is_polygon)
        self.route_width_spin.setEnabled(not is_polygon)
        self.route_reveal_spin.setEnabled(not is_polygon)
        self.route_start_spin.setEnabled(not is_polygon)
        self.route_end_spin.setEnabled(not is_polygon)
        self.route_label_edit.setEnabled(not is_polygon)
        self.route_legend_check.setEnabled(not is_polygon)
        self.route_start_spin.setMaximum(project_frame_max(self.project))
        self.route_end_spin.setMaximum(project_frame_max(self.project))
        if is_polygon:
            self.feather_spin.setValue(layer.feather_px)
            self.rounding_spin.setValue(layer.rounding_px)
            self.opacity_spin.setValue(layer.opacity)
            exact_keyframe, has_outgoing_segment = self._selected_polygon_keyframe_at_current_frame(layer)
            self._populate_polygon_easing_form(layer, exact_keyframe, has_outgoing_segment)
        else:
            self.route_width_spin.setValue(layer.width_px)
            self.route_reveal_spin.setValue(layer.reveal_px)
            self.route_start_spin.setValue(layer.start_frame)
            self.route_end_spin.setValue(layer.end_frame)
            self.route_label_edit.setText(layer.label)
            self.route_legend_check.setChecked(layer.show_in_legend)
            self._populate_polygon_easing_form(None, None, False)
        self._updating_form = False

    def _selected_polygon_keyframe_at_current_frame(self, layer: PolygonLayer) -> tuple[PolygonKeyframe | None, bool]:
        keyframes = sorted(layer.keyframes, key=lambda item: int(item.frame))
        current_frame = int(self.timeline_slider.value())
        for index, keyframe in enumerate(keyframes):
            if int(keyframe.frame) == current_frame:
                return keyframe, index < len(keyframes) - 1
        return None, False

    def _populate_polygon_easing_form(
        self,
        layer: PolygonLayer | None,
        keyframe: PolygonKeyframe | None,
        has_outgoing_segment: bool,
    ) -> None:
        self.polygon_constant_area_check.blockSignals(True)
        self.polygon_easing_combo.blockSignals(True)
        easing_value = getattr(keyframe, "outgoing_easing", POLYGON_EASING_LINEAR) if keyframe is not None else POLYGON_EASING_LINEAR
        constant_area_enabled = bool(getattr(keyframe, "outgoing_constant_area", False)) if keyframe is not None else False
        combo_index = self.polygon_easing_combo.findData(easing_value)
        if combo_index < 0:
            combo_index = self.polygon_easing_combo.findData(POLYGON_EASING_LINEAR)
        self.polygon_easing_combo.setCurrentIndex(max(0, combo_index))
        self.polygon_constant_area_check.setChecked(constant_area_enabled)

        if layer is None:
            self.polygon_constant_area_check.setEnabled(False)
            self.polygon_constant_area_check.setToolTip("Select a polygon layer to edit segment timing.")
            self.polygon_easing_combo.setEnabled(False)
            self.polygon_easing_combo.setToolTip("Select a polygon layer to edit keyframe easing.")
        elif keyframe is None:
            self.polygon_constant_area_check.setEnabled(False)
            self.polygon_constant_area_check.setToolTip("Move the timeline to an exact polygon keyframe to edit the outgoing segment timing.")
            self.polygon_easing_combo.setEnabled(False)
            self.polygon_easing_combo.setToolTip("Move the timeline to an exact polygon keyframe to edit the outgoing segment easing.")
        elif not has_outgoing_segment:
            self.polygon_constant_area_check.setEnabled(False)
            self.polygon_constant_area_check.setToolTip("The last polygon keyframe has no outgoing segment.")
            self.polygon_easing_combo.setEnabled(False)
            self.polygon_easing_combo.setToolTip("The last polygon keyframe has no outgoing segment.")
        else:
            self.polygon_constant_area_check.setEnabled(True)
            self.polygon_constant_area_check.setToolTip(
                "Equalize discovered area over time for the segment from this polygon keyframe to the next one."
            )
            self.polygon_easing_combo.setEnabled(not constant_area_enabled)
            self.polygon_easing_combo.setToolTip(
                "Disabled while Constant Area is active for this segment."
                if constant_area_enabled
                else "Controls the timing curve from this polygon keyframe to the next one."
            )
        self.polygon_constant_area_check.blockSignals(False)
        self.polygon_easing_combo.blockSignals(False)

    def _apply_polygon_constant_area(self, _state: int) -> None:
        if self._updating_form:
            return
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            return

        keyframe, has_outgoing_segment = self._selected_polygon_keyframe_at_current_frame(layer)
        if keyframe is None or not has_outgoing_segment:
            return

        keyframe.outgoing_constant_area = self.polygon_constant_area_check.isChecked()
        self._populate_polygon_easing_form(layer, keyframe, has_outgoing_segment)
        mode_label = "enabled" if keyframe.outgoing_constant_area else "disabled"
        self.statusBar().showMessage(f"Constant area pacing {mode_label} for polygon segment", 2500)
        self._refresh_canvas_only()

    def _apply_polygon_easing(self, _index: int) -> None:
        if self._updating_form:
            return
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            return

        keyframe, has_outgoing_segment = self._selected_polygon_keyframe_at_current_frame(layer)
        if keyframe is None or not has_outgoing_segment:
            return
        if getattr(keyframe, "outgoing_constant_area", False):
            return

        keyframe.outgoing_easing = str(self.polygon_easing_combo.currentData() or POLYGON_EASING_LINEAR)
        easing_label = self.polygon_easing_combo.currentText()
        self.statusBar().showMessage(f"Polygon segment easing set: {easing_label}", 2500)
        self._refresh_canvas_only()

    def _apply_layer_form(self) -> None:
        if self._updating_form:
            return
        kind, layer = self._selected_layer()
        if layer is None:
            return
        layer.name = self.layer_name_edit.text().strip() or layer.name
        layer.visible = self.layer_visible_check.isChecked()
        layer.color = list(self.color_button.property("rgb") or layer.color)
        if kind == "polygon":
            layer.feather_px = int(self.feather_spin.value())
            layer.rounding_px = int(self.rounding_spin.value())
            layer.opacity = float(self.opacity_spin.value())
        else:
            layer.width_px = int(self.route_width_spin.value())
            layer.reveal_px = int(self.route_reveal_spin.value())
            layer.start_frame = int(self.route_start_spin.value())
            layer.end_frame = max(layer.start_frame, int(self.route_end_spin.value()))
            layer.label = self.route_label_edit.text().strip()
            layer.show_in_legend = self.route_legend_check.isChecked()
        self._refresh_layer_list()
        self._refresh_canvas_only()

    def _set_color_button(self, rgb: list[int]) -> None:
        rgb = [int(v) for v in rgb[:3]]
        self.color_button.setProperty("rgb", rgb)
        self.color_button.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); color: rgb(15, 18, 22);"
        )

    def _pick_color(self) -> None:
        kind, layer = self._selected_layer()
        if layer is None:
            return
        current = self.color_button.property("rgb") or layer.color
        color = QColorDialog.getColor(QColor(*current), self, "Pick Layer Color")
        if not color.isValid():
            return
        self._set_color_button([color.red(), color.green(), color.blue()])
        self._apply_layer_form()

    def _begin_polygon_keyframe(self) -> None:
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            QMessageBox.information(self, "Polygon Keyframe", "Select a polygon layer first.")
            return
        self.canvas.begin_polygon_draw(layer.id)

    def _insert_current_keyframe(self) -> None:
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            QMessageBox.information(self, "Insert Keyframe", "Select a polygon layer first.")
            return
        if not layer.keyframes:
            QMessageBox.information(self, "Insert Keyframe", "Draw a polygon keyframe first.")
            return
        frame_index = self.timeline_slider.value()
        if any(kf.frame == frame_index for kf in layer.keyframes):
            QMessageBox.information(self, "Insert Keyframe", f"There is already a keyframe at frame {frame_index}.")
            return
        points = polygon_edit_points_at_frame(layer, frame_index)
        if not points:
            QMessageBox.information(self, "Insert Keyframe", "No polygon data at the current frame.")
            return
        layer.keyframes.append(PolygonKeyframe(frame=frame_index, points=points))
        layer.keyframes.sort(key=lambda kf: kf.frame)
        self._refresh_keyframe_track()
        self._populate_layer_form()
        self._refresh_canvas_only()
        self.statusBar().showMessage(f"Keyframe inserted at frame {frame_index}.", 2500)

    def _delete_current_keyframe(self) -> None:
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            QMessageBox.information(self, "Delete Keyframe", "Select a polygon layer first.")
            return
        frame_index = self.timeline_slider.value()
        existing = next((kf for kf in layer.keyframes if kf.frame == frame_index), None)
        if existing is None:
            QMessageBox.information(self, "Delete Keyframe", f"No keyframe at frame {frame_index}.")
            return
        if len(layer.keyframes) <= 1:
            QMessageBox.information(self, "Delete Keyframe", "Cannot delete the last remaining keyframe.")
            return
        layer.keyframes = [kf for kf in layer.keyframes if kf.frame != frame_index]
        self._refresh_keyframe_track()
        self._populate_layer_form()
        self._refresh_canvas_only()
        self.statusBar().showMessage(f"Keyframe at frame {frame_index} deleted.", 2500)

    def _delete_selected_layer(self) -> None:
        kind, layer = self._selected_layer()
        if layer is None:
            return
        if kind == "polygon":
            self.project.polygon_layers = [item for item in self.project.polygon_layers if item.id != layer.id]
        else:
            self.project.route_layers = [item for item in self.project.route_layers if item.id != layer.id]
        self._refresh_layer_list()
        self._populate_layer_form()
        self._refresh_canvas_only()

    def _handle_drawing_finished(self, payload: dict) -> None:
        frame_index = self.timeline_slider.value()
        if payload.get("kind") == "polygon":
            points = payload.get("points", [])
            target_id = payload.get("target_layer_id")
            if target_id:
                for layer in self.project.polygon_layers:
                    if layer.id == target_id:
                        existing = next((kf for kf in layer.keyframes if kf.frame == frame_index), None)
                        if existing is None:
                            layer.keyframes.append(PolygonKeyframe(frame=frame_index, points=points))
                        else:
                            existing.points = points
                        layer.keyframes.sort(key=lambda item: item.frame)
                        break
            else:
                color = POLYGON_COLORS[len(self.project.polygon_layers) % len(POLYGON_COLORS)]
                layer = PolygonLayer(
                    name=f"Reveal {len(self.project.polygon_layers) + 1}",
                    color=color,
                    keyframes=[PolygonKeyframe(frame=frame_index, points=points)],
                )
                self.project.polygon_layers.append(layer)
            self._refresh_layer_list()
        elif payload.get("kind") == "route":
            points = payload.get("points", [])
            color = ROUTE_COLORS[len(self.project.route_layers) % len(ROUTE_COLORS)]
            end_frame = min(project_frame_max(self.project), frame_index + self.project.fps * 4)
            layer = RouteLayer(
                name=f"Route {len(self.project.route_layers) + 1}",
                color=color,
                label=f"Route {len(self.project.route_layers) + 1}",
                start_frame=frame_index,
                end_frame=end_frame,
                points=points,
            )
            self.project.route_layers.append(layer)
            self._refresh_layer_list()
        self._populate_layer_form()
        self._refresh_canvas_only()

    def _refresh_timeline(self) -> None:
        current = min(self.timeline_slider.value(), project_frame_max(self.project))
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(project_frame_max(self.project))
        self.timeline_slider.setValue(current)
        self.timeline_slider.blockSignals(False)
        self.timeline_label.setText(f"Frame {current} / {project_frame_max(self.project)}")
        self.play_timer.setInterval(16)
        self._refresh_keyframe_track()

    def _on_frame_changed(self, value: int) -> None:
        self.timeline_label.setText(f"Frame {value} / {project_frame_max(self.project)}")
        self.canvas.set_frame_index(value)
        self._populate_layer_form()

    def _set_current_frame(self, frame: int) -> None:
        clamped_frame = max(0, min(project_frame_max(self.project), int(frame)))
        if clamped_frame != self.timeline_slider.value():
            self.timeline_slider.setValue(clamped_frame)

    def _selected_polygon_keyframes(self) -> list[PolygonKeyframe]:
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            return []
        return sorted(layer.keyframes, key=lambda item: int(item.frame))

    def _refresh_keyframe_track(self) -> None:
        keyframes = self._selected_polygon_keyframes()
        if keyframes:
            message = "Drag a marker left or right to retime the polygon interpolation."
        else:
            kind, layer = self._selected_layer()
            if kind == "polygon" and layer is not None:
                message = "This polygon layer has no keyframes yet."
            else:
                message = "Select a polygon layer to edit its keyframes."

        self.keyframe_track.set_state(
            minimum=0,
            maximum=project_frame_max(self.project),
            current_frame=self.timeline_slider.value(),
            keyframes=[int(item.frame) for item in keyframes],
            interactive=bool(keyframes),
            message=message,
        )

    def _move_selected_polygon_keyframe(self, index: int, frame: int) -> None:
        kind, layer = self._selected_layer()
        if kind != "polygon" or layer is None:
            return

        keyframes = sorted(layer.keyframes, key=lambda item: int(item.frame))
        if not (0 <= index < len(keyframes)):
            return

        frame_values = [int(item.frame) for item in keyframes]
        clamped_frame = clamp_keyframe_frame(frame_values, index, frame, 0, project_frame_max(self.project))
        keyframes[index].frame = clamped_frame
        keyframes.sort(key=lambda item: int(item.frame))
        layer.keyframes = keyframes

        if clamped_frame != self.timeline_slider.value():
            self.timeline_slider.setValue(clamped_frame)
        else:
            self._refresh_keyframe_track()
            self._refresh_canvas_only()

    def _announce_keyframe_move(self, _index: int, frame: int) -> None:
        self.statusBar().showMessage(f"Polygon keyframe moved to frame {frame}", 2500)

    def _refresh_canvas_only(self) -> None:
        self.canvas.invalidate_cache()
        self.canvas.update()

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self._playback_start_time = None
            self.canvas.set_playback_active(False)
            self.play_button.setText("Play")
        else:
            self._playback_start_time = time.perf_counter()
            self._playback_start_frame = self.timeline_slider.value()
            self.canvas.set_playback_active(True)
            self.play_timer.start()
            self.play_button.setText("Pause")

    def _advance_frame(self) -> None:
        if self._playback_start_time is None:
            return
        max_frame = project_frame_max(self.project)
        elapsed = max(0.0, time.perf_counter() - self._playback_start_time)
        target_frame = self._playback_start_frame + int(elapsed * max(1, self.project.fps))
        if target_frame >= max_frame:
            self.play_timer.stop()
            self._playback_start_time = None
            self.canvas.set_playback_active(False)
            self.timeline_slider.setValue(max_frame)
            self.play_button.setText("Play")
            return
        if target_frame != self.timeline_slider.value():
            self.timeline_slider.setValue(target_frame)

    def _export_png(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export PNG", str(self.exports_root / "preview.png"), "PNG Files (*.png)")
        if not path:
            return
        export_frame_png(self.project, self.canvas.basemap_image, self.timeline_slider.value(), path, font_path=self.font_path)
        self.statusBar().showMessage(f"PNG exported: {path}", 4000)

    def _export_video(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export MP4", str(self.exports_root / "exploration_video.mp4"), "MP4 Files (*.mp4)")
        if not path:
            return
        total_frames = export_frame_count(self.project)
        progress = QProgressDialog("Exporting video...", None, 0, total_frames, self)
        progress.setWindowTitle("Export")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        try:
            export_video(
                self.project,
                self.canvas.basemap_image,
                path,
                font_path=self.font_path,
                progress_callback=lambda done, total: self._update_export_progress(progress, done, total),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return
        progress.setValue(total_frames)
        self.statusBar().showMessage(f"MP4 exported: {path}", 4000)

    def _update_export_progress(self, dialog: QProgressDialog, done: int, total: int) -> None:
        dialog.setMaximum(total)
        dialog.setValue(done)
        dialog.setLabelText(f"Exporting video... {done}/{total}")
        QApplication.processEvents()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._toggle_play()
            return
        if event.key() == Qt.Key.Key_H:
            self._set_final_preview_mode(not self.final_preview_button.isChecked())
            return
        if event.key() == Qt.Key.Key_Delete:
            if self.canvas.delete_selected_vertex():
                self.statusBar().showMessage("Polygon point deleted", 2500)
                return
        if event.key() == Qt.Key.Key_Escape:
            self.canvas.cancel_drawing()
            return
        super().keyPressEvent(event)



def project_frame_max(project: Project) -> int:
    return max(0, project.frame_count - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exploration_Editor")
    parser.add_argument("--project", default=None)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(initial_project_path=args.project)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
