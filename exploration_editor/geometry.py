from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from exploration_editor.model import PolygonLayer


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def lonlat_to_world(lon: float, lat: float, world_size: tuple[float, float], wrap: bool = True) -> tuple[float, float]:
    world_w, world_h = world_size
    lon_value = float(lon)
    if wrap:
        lon_value = ((lon_value + 180.0) % 360.0) - 180.0
    x = ((lon_value + 180.0) / 360.0) * world_w
    y = ((90.0 - float(lat)) / 180.0) * world_h
    return x, y


def world_to_lonlat(x: float, y: float, world_size: tuple[float, float]) -> list[float]:
    world_w, world_h = world_size
    lon = (float(x) / world_w) * 360.0 - 180.0
    lat = 90.0 - (float(y) / world_h) * 180.0
    while lon < -180.0:
        lon += 360.0
    while lon > 180.0:
        lon -= 360.0
    return [lon, clamp(lat, -90.0, 90.0)]


def unwrap_longitudes(points: Iterable[Iterable[float]]) -> list[list[float]]:
    pts = [[float(p[0]), float(p[1])] for p in points]
    if not pts:
        return []
    result = [pts[0][:]]
    for lon, lat in pts[1:]:
        prev_lon = result[-1][0]
        candidate = lon
        while candidate - prev_lon > 180.0:
            candidate -= 360.0
        while candidate - prev_lon < -180.0:
            candidate += 360.0
        result.append([candidate, lat])
    return result


def _path_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.array([0.0], dtype=np.float64)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(deltas * deltas, axis=1))
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def resample_path(points: list[list[float]], sample_count: int, closed: bool = False) -> list[list[float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0][:] for _ in range(max(1, sample_count))]

    unwrapped = unwrap_longitudes(points)
    if closed:
        unwrapped = unwrapped + [unwrapped[0][:]]

    path = np.asarray(unwrapped, dtype=np.float64)
    lengths = _path_lengths(path)
    total = float(lengths[-1])
    if total <= 1e-6:
        return [list(path[0]) for _ in range(max(1, sample_count))]

    targets = np.linspace(0.0, total, max(2, sample_count), endpoint=not closed)
    samples: list[list[float]] = []
    for target in targets:
        idx = int(np.searchsorted(lengths, target, side="right") - 1)
        idx = max(0, min(idx, len(path) - 2))
        start = path[idx]
        end = path[idx + 1]
        seg_len = max(1e-6, float(lengths[idx + 1] - lengths[idx]))
        t = (target - lengths[idx]) / seg_len
        point = start * (1.0 - t) + end * t
        samples.append([float(point[0]), float(point[1])])
    return samples


def interpolate_paths(points_a: list[list[float]], points_b: list[list[float]], t: float, closed: bool = False) -> list[list[float]]:
    if not points_a:
        return [point[:] for point in points_b]
    if not points_b:
        return [point[:] for point in points_a]

    t = clamp(float(t), 0.0, 1.0)
    sample_count = max(len(points_a), len(points_b), 96 if closed else 48)
    path_a = np.asarray(resample_path(points_a, sample_count, closed=closed), dtype=np.float64)
    path_b = np.asarray(resample_path(points_b, sample_count, closed=closed), dtype=np.float64)
    blended = path_a * (1.0 - t) + path_b * t
    return [[float(point[0]), float(point[1])] for point in blended]


def polygon_points_at_frame(layer: PolygonLayer, frame: int) -> list[list[float]]:
    if not layer.keyframes:
        return []

    keyframes = sorted(layer.keyframes, key=lambda item: item.frame)
    if frame <= keyframes[0].frame:
        return [point[:] for point in keyframes[0].points]
    if frame >= keyframes[-1].frame:
        return [point[:] for point in keyframes[-1].points]

    for index in range(len(keyframes) - 1):
        left = keyframes[index]
        right = keyframes[index + 1]
        if left.frame <= frame <= right.frame:
            if frame == left.frame:
                return [point[:] for point in left.points]
            if frame == right.frame:
                return [point[:] for point in right.points]
            span = max(1, right.frame - left.frame)
            t = (frame - left.frame) / span
            return interpolate_paths(left.points, right.points, t, closed=True)
    return [point[:] for point in keyframes[-1].points]


def path_prefix(points: list[list[float]], progress: float) -> list[list[float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0][:]]

    progress = clamp(float(progress), 0.0, 1.0)
    if progress <= 0.0:
        return [points[0][:]]
    if progress >= 1.0:
        return [point[:] for point in points]

    unwrapped = unwrap_longitudes(points)
    path = np.asarray(unwrapped, dtype=np.float64)
    lengths = _path_lengths(path)
    total = float(lengths[-1])
    if total <= 1e-6:
        return [points[0][:]]

    target = total * progress
    result: list[list[float]] = [unwrapped[0][:]]
    for index in range(len(path) - 1):
        start_len = float(lengths[index])
        end_len = float(lengths[index + 1])
        start = path[index]
        end = path[index + 1]
        if target >= end_len:
            result.append([float(end[0]), float(end[1])])
            continue
        seg_len = max(1e-6, end_len - start_len)
        t = (target - start_len) / seg_len
        point = start * (1.0 - t) + end * t
        result.append([float(point[0]), float(point[1])])
        break
    return result


def route_progress(frame: int, start_frame: int, end_frame: int) -> float:
    if end_frame <= start_frame:
        return 1.0 if frame >= end_frame else 0.0
    return clamp((frame - start_frame) / float(end_frame - start_frame), 0.0, 1.0)
