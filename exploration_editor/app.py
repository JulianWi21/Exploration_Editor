from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from PIL import Image
from PyQt6.QtCore import QPointF, QTimer, Qt, pyqtSignal
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
from exploration_editor.export import export_frame_png, export_video
from exploration_editor.geometry import (
    clamp,
    lonlat_to_world,
    polygon_edit_points_at_frame,
    resample_path,
    rounded_closed_path,
    world_to_lonlat,
    unwrap_longitudes,
)
from exploration_editor.model import DEFAULT_BASEMAP_PATH, PolygonKeyframe, PolygonLayer, Project, RouteLayer, default_project, load_project, save_project
from exploration_editor.render import clamp_view_state, compute_map_layout, render_frame


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


def _pil_to_qimage(image) -> QImage:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return qimage.copy()


class MapCanvas(QWidget):
    drawingFinished = pyqtSignal(object)
    viewChanged = pyqtSignal()

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
        self.show_edit_overlays = True
        self.is_panning = False
        self.last_mouse_pos: tuple[float, float] | None = None
        self._last_layout: dict[str, float] | None = None
        self._cached_qimage: QImage | None = None
        self._cached_key: tuple[object, ...] | None = None
        self._last_render_time = 0.0
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
        self._render_now()

    def set_selected_polygon_layer(self, layer_id: str | None) -> None:
        self.selected_polygon_layer_id = layer_id
        self._dragging_vertex_index = None
        self._selected_vertex_index = None
        self.update()

    def set_show_edit_overlays(self, show_edit_overlays: bool) -> None:
        self.show_edit_overlays = bool(show_edit_overlays)
        self._dragging_vertex_index = None
        self._selected_vertex_index = None
        self.update()

    def invalidate_cache(self) -> None:
        self._interactive_render_timer.stop()
        self._cached_qimage = None
        self._cached_key = None

    def _build_preview_basemap_levels(self, image):
        levels = []
        for target_width in (512, 1024, 2048, 4096):
            if image.width <= target_width:
                continue
            target_height = max(1, int(round(image.height * (target_width / image.width))))
            levels.append(image.resize((target_width, target_height), resample=Image.Resampling.BILINEAR))
        levels.append(image)
        return levels

    def _select_preview_basemap(self, layout: dict[str, float]):
        desired_width = max(256, int(round(layout["draw_w"] * 1.35)))
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
        points = self._editable_polygon_points()
        if len(points) < 3:
            return None
        tolerance_px = max(10.0, self.height() * 0.014)
        best_match: tuple[int, float] | None = None
        best_dist_sq = tolerance_px * tolerance_px
        for path in self._screen_variants(points):
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
        self._sync_polygon_keyframe_topology(layer, len(keyframe.points))

        edge_index, edge_t = edge_match
        for polygon_keyframe in layer.keyframes:
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

        self._sync_polygon_keyframe_topology(layer, len(keyframe.points))
        delete_index = self._selected_vertex_index
        for polygon_keyframe in layer.keyframes:
            if len(polygon_keyframe.points) > 3 and delete_index < len(polygon_keyframe.points):
                polygon_keyframe.points.pop(delete_index)

        if len(keyframe.points) > 3:
            self._selected_vertex_index = min(delete_index, len(keyframe.points) - 1)
        else:
            self._selected_vertex_index = None
        self._dragging_vertex_index = None
        self._render_now()
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
        self.update()

    def paintEvent(self, _event) -> None:
        frame_size = (max(2, self.width()), max(2, self.height()))
        layout = compute_map_layout(frame_size, self.basemap_image.size, self.project.view)
        preview_basemap = self._select_preview_basemap(layout)
        key = (
            frame_size,
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
        if self._cached_qimage is None or self._cached_key != key:
            image = render_frame(
                self.project,
                basemap_image=self.basemap_image,
                display_basemap_image=preview_basemap,
                frame_index=self.frame_index,
                output_size=frame_size,
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
        outline = QColor(layer.color[0], layer.color[1], layer.color[2], 235)
        handle_outer = QColor(6, 8, 12, 220)
        handle_inner = QColor(245, 248, 252, 235)
        selected_handle = QColor(88, 170, 255, 245)
        active_handle = QColor(255, 208, 84, 245)

        pen = QPen(outline)
        pen.setWidth(3)
        if not is_exact_keyframe:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        for path in self._screen_variants(points):
            outline_path = rounded_closed_path(path, float(layer.rounding_px))
            polygon = QPolygonF([QPointF(x, y) for x, y in outline_path])
            painter.drawPolygon(polygon)
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
        if world_x < 0.0 or world_x > self.basemap_image.width:
            return None
        if world_y < 0.0 or world_y > self.basemap_image.height:
            return None
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
                    keyframe = self._ensure_polygon_keyframe(layer, self.frame_index)
                    if 0 <= handle_index < len(keyframe.points):
                        self._selected_vertex_index = handle_index
                        self._dragging_vertex_index = handle_index
                        point = self._screen_to_lonlat(pos.x(), pos.y())
                        if point is not None:
                            keyframe.points[handle_index] = point
                        self._schedule_interactive_render()
                        self.update()
                        return
            self._selected_vertex_index = None
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
                    self._schedule_interactive_render()
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
            self._render_now()
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
        self.font_path = self.repo_root / "fonts" / "Montserrat-ExtraBold.ttf"
        self.project_path: str | None = None
        self.project = default_project(str(self.repo_root / DEFAULT_BASEMAP_PATH))
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
        self.open_basemap_button = QPushButton("Open Basemap")
        self.new_polygon_button = QPushButton("New Polygon")
        self.add_keyframe_button = QPushButton("Add Polygon Keyframe")
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
            self.open_basemap_button,
            self.new_polygon_button,
            self.add_keyframe_button,
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

        timeline = QHBoxLayout()
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_label = QLabel("Frame 0 / 0")
        timeline.addWidget(QLabel("Timeline"))
        timeline.addWidget(self.timeline_slider, 1)
        timeline.addWidget(self.timeline_label)
        root_layout.addLayout(timeline)

        self.setCentralWidget(root)

    def _connect_signals(self) -> None:
        self.new_project_button.clicked.connect(self._new_project)
        self.open_project_button.clicked.connect(self._open_project)
        self.save_project_button.clicked.connect(self._save_project)
        self.open_basemap_button.clicked.connect(self._open_basemap)
        self.new_polygon_button.clicked.connect(lambda: self.canvas.begin_polygon_draw(None))
        self.add_keyframe_button.clicked.connect(self._begin_polygon_keyframe)
        self.new_route_button.clicked.connect(self.canvas.begin_route_draw)
        self.delete_layer_button.clicked.connect(self._delete_selected_layer)
        self.export_png_button.clicked.connect(self._export_png)
        self.export_video_button.clicked.connect(self._export_video)
        self.play_button.clicked.connect(self._toggle_play)
        self.final_preview_button.toggled.connect(self._set_final_preview_mode)
        self.timeline_slider.valueChanged.connect(self._on_frame_changed)
        self.layer_list.currentRowChanged.connect(self._populate_layer_form)
        self.color_button.clicked.connect(self._pick_color)
        self.layer_name_edit.editingFinished.connect(self._apply_layer_form)
        self.layer_visible_check.stateChanged.connect(self._apply_layer_form)
        self.feather_spin.valueChanged.connect(self._apply_layer_form)
        self.rounding_spin.valueChanged.connect(self._apply_layer_form)
        self.opacity_spin.valueChanged.connect(self._apply_layer_form)
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
        path, _filter = QFileDialog.getOpenFileName(self, "Open Project", str(self.repo_root / "examples"), "JSON Files (*.json)")
        if path:
            self._load_project_from_path(path)

    def _load_project_from_path(self, path: str) -> None:
        project = load_project(path)
        basemap_path = Path(project.basemap_path)
        if not basemap_path.is_absolute():
            project.basemap_path = str((Path(path).resolve().parent / basemap_path).resolve())
        self._reset_project(project, path)

    def _save_project(self) -> None:
        if self.project_path is None:
            default_name = self.project.title.lower().replace(" ", "_") or "exploration_project"
            path, _filter = QFileDialog.getSaveFileName(self, "Save Project", str(self.repo_root / "examples" / f"{default_name}.json"), "JSON Files (*.json)")
            if not path:
                return
            self.project_path = path
        save_project(self.project, self.project_path)
        self.statusBar().showMessage(f"Project saved: {self.project_path}", 4000)

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
        self._updating_form = True
        enabled = layer is not None
        for widget in [
            self.layer_name_edit,
            self.layer_visible_check,
            self.color_button,
            self.feather_spin,
            self.rounding_spin,
            self.opacity_spin,
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
            self._updating_form = False
            return
        self.layer_name_edit.setText(layer.name)
        self.layer_visible_check.setChecked(layer.visible)
        self._set_color_button(layer.color)
        is_polygon = kind == "polygon"
        self.feather_spin.setEnabled(is_polygon)
        self.rounding_spin.setEnabled(is_polygon)
        self.opacity_spin.setEnabled(is_polygon)
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
        else:
            self.route_width_spin.setValue(layer.width_px)
            self.route_reveal_spin.setValue(layer.reveal_px)
            self.route_start_spin.setValue(layer.start_frame)
            self.route_end_spin.setValue(layer.end_frame)
            self.route_label_edit.setText(layer.label)
            self.route_legend_check.setChecked(layer.show_in_legend)
        self._updating_form = False

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

    def _delete_selected_layer(self) -> None:
        kind, layer = self._selected_layer()
        if layer is None:
            return
        if kind == "polygon":
            self.project.polygon_layers = [item for item in self.project.polygon_layers if item.id != layer.id]
        else:
            self.project.route_layers = [item for item in self.project.route_layers if item.id != layer.id]
        self._refresh_layer_list()
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
        self._refresh_canvas_only()

    def _refresh_timeline(self) -> None:
        current = min(self.timeline_slider.value(), project_frame_max(self.project))
        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(project_frame_max(self.project))
        self.timeline_slider.setValue(current)
        self.timeline_slider.blockSignals(False)
        self.timeline_label.setText(f"Frame {current} / {project_frame_max(self.project)}")
        self.play_timer.setInterval(int(round(1000.0 / max(1, self.project.fps))))

    def _on_frame_changed(self, value: int) -> None:
        self.timeline_label.setText(f"Frame {value} / {project_frame_max(self.project)}")
        self.canvas.set_frame_index(value)
        self._populate_layer_form()

    def _refresh_canvas_only(self) -> None:
        self.canvas.invalidate_cache()
        self.canvas.update()

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("Play")
        else:
            self.play_timer.start()
            self.play_button.setText("Pause")

    def _advance_frame(self) -> None:
        current = self.timeline_slider.value()
        if current >= project_frame_max(self.project):
            self.play_timer.stop()
            self.play_button.setText("Play")
            return
        self.timeline_slider.setValue(current + 1)

    def _export_png(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export PNG", str(self.repo_root / "exports" / "preview.png"), "PNG Files (*.png)")
        if not path:
            return
        export_frame_png(self.project, self.canvas.basemap_image, self.timeline_slider.value(), path, font_path=self.font_path)
        self.statusBar().showMessage(f"PNG exported: {path}", 4000)

    def _export_video(self) -> None:
        path, _filter = QFileDialog.getSaveFileName(self, "Export MP4", str(self.repo_root / "exports" / "exploration_video.mp4"), "MP4 Files (*.mp4)")
        if not path:
            return
        progress = QProgressDialog("Exporting video...", None, 0, self.project.frame_count, self)
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
        progress.setValue(self.project.frame_count)
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
