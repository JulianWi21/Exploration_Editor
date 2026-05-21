from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterable

import numpy as np

from exploration_editor.model import POLYGON_EASING_LINEAR, POLYGON_EASING_MODES, PolygonLayer


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def apply_easing(value: float, easing: str | None) -> float:
    t = clamp(float(value), 0.0, 1.0)
    mode = str(easing or POLYGON_EASING_LINEAR).strip().lower()
    if mode not in POLYGON_EASING_MODES or mode == POLYGON_EASING_LINEAR:
        return t
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        one_minus_t = 1.0 - t
        return 1.0 - one_minus_t * one_minus_t
    if mode == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return t


def _immutable_points(points: Iterable[Iterable[float]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in points)


def _polygon_area(points: Iterable[Iterable[float]]) -> float:
    path = unwrap_longitudes(points)
    if len(path) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(path):
        x2, y2 = path[(index + 1) % len(path)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


@lru_cache(maxsize=256)
def _build_constant_area_curve(
    points_a_key: tuple[tuple[float, float], ...],
    points_b_key: tuple[tuple[float, float], ...],
    sample_count: int,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    points_a = [list(point) for point in points_a_key]
    points_b = [list(point) for point in points_b_key]
    sample_total = max(9, int(sample_count))
    samples_t = np.linspace(0.0, 1.0, sample_total, dtype=np.float64)
    areas = np.empty(sample_total, dtype=np.float64)

    for index, sample_t in enumerate(samples_t):
        if index == 0:
            interpolated = points_a
        elif index == sample_total - 1:
            interpolated = points_b
        else:
            interpolated = interpolate_paths(points_a, points_b, float(sample_t), closed=True)
        areas[index] = _polygon_area(interpolated)

    delta = float(areas[-1] - areas[0])
    if abs(delta) <= 1e-6:
        identity = tuple(float(value) for value in samples_t)
        return identity, identity

    monotonic_areas = np.maximum.accumulate(areas) if delta > 0.0 else np.minimum.accumulate(areas)
    progress = np.clip((monotonic_areas - monotonic_areas[0]) / delta, 0.0, 1.0)
    progress[0] = 0.0
    progress[-1] = 1.0

    keep_indices = [0]
    last_progress = float(progress[0])
    for index in range(1, len(progress) - 1):
        current_progress = float(progress[index])
        if current_progress > last_progress + 1e-6:
            keep_indices.append(index)
            last_progress = current_progress
    keep_indices.append(len(progress) - 1)

    kept_progress = tuple(float(progress[index]) for index in keep_indices)
    kept_t = tuple(float(samples_t[index]) for index in keep_indices)
    if len(kept_progress) < 2 or kept_progress[-1] <= kept_progress[0] + 1e-6:
        identity = tuple(float(value) for value in samples_t)
        return identity, identity
    return kept_progress, kept_t


def remap_constant_area_progress(
    points_a: list[list[float]],
    points_b: list[list[float]],
    value: float,
    sample_count: int = 65,
) -> float:
    t = clamp(float(value), 0.0, 1.0)
    progress_curve, t_curve = _build_constant_area_curve(
        _immutable_points(points_a),
        _immutable_points(points_b),
        int(sample_count),
    )
    return float(np.interp(t, np.asarray(progress_curve, dtype=np.float64), np.asarray(t_curve, dtype=np.float64)))


def polygon_segment_progress(
    left_points: list[list[float]],
    right_points: list[list[float]],
    value: float,
    easing: str | None = None,
    constant_area: bool = False,
) -> float:
    t = clamp(float(value), 0.0, 1.0)
    if bool(constant_area):
        return remap_constant_area_progress(left_points, right_points, t)
    return apply_easing(t, easing)


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


def _wrap_longitude(lon: float) -> float:
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and float(lon) > 0.0:
        return 180.0
    return wrapped


def _interpolate_matched_points(points_a: list[list[float]], points_b: list[list[float]], t: float) -> list[list[float]]:
    left = unwrap_longitudes(points_a)
    right = unwrap_longitudes(points_b)
    blended: list[list[float]] = []
    for point_a, point_b in zip(left, right):
        lon_b = point_b[0]
        while lon_b - point_a[0] > 180.0:
            lon_b -= 360.0
        while lon_b - point_a[0] < -180.0:
            lon_b += 360.0
        blended.append(
            [
                _wrap_longitude(point_a[0] * (1.0 - t) + lon_b * t),
                float(point_a[1] * (1.0 - t) + point_b[1] * t),
            ]
        )
    return blended


def _point_distance_sq(point_a: list[float], point_b: list[float]) -> float:
    delta_lon = float(point_b[0]) - float(point_a[0])
    while delta_lon > 180.0:
        delta_lon -= 360.0
    while delta_lon < -180.0:
        delta_lon += 360.0
    delta_lat = float(point_b[1]) - float(point_a[1])
    return delta_lon * delta_lon + delta_lat * delta_lat


def _interpolate_edge_point(start: list[float], end: list[float], t: float) -> list[float]:
    pair = unwrap_longitudes([start, end])
    lon = pair[0][0] * (1.0 - t) + pair[1][0] * t
    lat = float(start[1] * (1.0 - t) + end[1] * t)
    return [_wrap_longitude(lon), lat]


def _select_subsequence_indices(points_small: list[list[float]], points_large: list[list[float]]) -> list[int]:
    small_count = len(points_small)
    large_count = len(points_large)
    if small_count == 0:
        return []
    if small_count >= large_count:
        return list(range(large_count))

    parents = [[-1] * large_count for _ in range(small_count)]
    previous = [math.inf] * large_count
    max_first_index = large_count - small_count
    for large_index in range(max_first_index + 1):
        previous[large_index] = _point_distance_sq(points_small[0], points_large[large_index])

    for small_index in range(1, small_count):
        current = [math.inf] * large_count
        prefix_best = math.inf
        prefix_index = -1
        min_large_index = small_index
        max_large_index = large_count - (small_count - small_index)
        for large_index in range(min_large_index, max_large_index + 1):
            candidate_index = large_index - 1
            candidate_cost = previous[candidate_index]
            if candidate_cost < prefix_best:
                prefix_best = candidate_cost
                prefix_index = candidate_index
            if prefix_index >= 0 and prefix_best < math.inf:
                current[large_index] = prefix_best + _point_distance_sq(points_small[small_index], points_large[large_index])
                parents[small_index][large_index] = prefix_index
        previous = current

    valid_last_indices = range(small_count - 1, large_count)
    best_last_index = min(valid_last_indices, key=lambda index: previous[index])
    if math.isinf(previous[best_last_index]):
        return list(range(small_count))

    selected = [0] * small_count
    selected[-1] = best_last_index
    for small_index in range(small_count - 1, 0, -1):
        selected[small_index - 1] = parents[small_index][selected[small_index]]
    return selected


def _cyclic_index_span(start_index: int, end_index: int, count: int) -> list[int]:
    indices = [start_index]
    current = start_index
    while current != end_index:
        current = (current + 1) % count
        indices.append(current)
    return indices


def _expand_closed_path_to_match(points_small: list[list[float]], points_large: list[list[float]]) -> list[list[float]]:
    if not points_small:
        return []
    if len(points_small) >= len(points_large):
        return [point[:] for point in points_small]

    selected = _select_subsequence_indices(points_small, points_large)
    expanded: list[list[float] | None] = [None] * len(points_large)
    for small_index, large_index in enumerate(selected):
        expanded[large_index] = points_small[small_index][:]

    for small_index, start_large_index in enumerate(selected):
        next_small_index = (small_index + 1) % len(points_small)
        end_large_index = selected[next_small_index]
        span_indices = _cyclic_index_span(start_large_index, end_large_index, len(points_large))
        if len(span_indices) <= 2:
            continue

        span_points = [points_large[index] for index in span_indices]
        span_path = np.asarray(unwrap_longitudes(span_points), dtype=np.float64)
        span_lengths = _path_lengths(span_path)
        span_total = float(span_lengths[-1])
        denominator = max(1, len(span_indices) - 1)
        for offset, large_index in enumerate(span_indices[1:-1], start=1):
            if span_total <= 1e-6:
                position = offset / denominator
            else:
                position = float(span_lengths[offset] / span_total)
            expanded[large_index] = _interpolate_edge_point(
                points_small[small_index],
                points_small[next_small_index],
                position,
            )

    return [point[:] if point is not None else points_large[index][:] for index, point in enumerate(expanded)]


def interpolate_paths(
    points_a: list[list[float]],
    points_b: list[list[float]],
    t: float,
    closed: bool = False,
    sample_count: int | None = None,
) -> list[list[float]]:
    if not points_a:
        return [point[:] for point in points_b]
    if not points_b:
        return [point[:] for point in points_a]

    t = clamp(float(t), 0.0, 1.0)
    if len(points_a) == len(points_b):
        requested_samples = len(points_a) if sample_count is None else int(sample_count)
        if requested_samples == len(points_a):
            return _interpolate_matched_points(points_a, points_b, t)
    if closed and len(points_a) != len(points_b):
        if len(points_a) < len(points_b):
            aligned_a = _expand_closed_path_to_match(points_a, points_b)
            blended = _interpolate_matched_points(aligned_a, points_b, t)
        else:
            aligned_b = _expand_closed_path_to_match(points_b, points_a)
            blended = _interpolate_matched_points(points_a, aligned_b, t)
        if sample_count is not None and int(sample_count) > len(blended):
            return resample_path(blended, int(sample_count), closed=True)
        return blended
    if sample_count is None:
        sample_count = max(len(points_a), len(points_b), 96 if closed else 48)
    else:
        sample_count = max(int(sample_count), 3 if closed else 2)
    path_a = np.asarray(resample_path(points_a, sample_count, closed=closed), dtype=np.float64)
    path_b = np.asarray(resample_path(points_b, sample_count, closed=closed), dtype=np.float64)
    blended = path_a * (1.0 - t) + path_b * t
    return [[float(point[0]), float(point[1])] for point in blended]


def polygon_edit_points_at_frame(layer: PolygonLayer, frame: int) -> list[list[float]]:
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
            t = polygon_segment_progress(
                left.points,
                right.points,
                (frame - left.frame) / span,
                easing=getattr(left, "outgoing_easing", POLYGON_EASING_LINEAR),
                constant_area=getattr(left, "outgoing_constant_area", False),
            )
            sample_count = max(len(left.points), len(right.points), 3)
            return interpolate_paths(
                left.points,
                right.points,
                t,
                closed=True,
                sample_count=sample_count,
            )
    return [point[:] for point in keyframes[-1].points]


def rounded_closed_path(points: Iterable[Iterable[float]], radius: float) -> list[tuple[float, float]]:
    path = [(float(point[0]), float(point[1])) for point in points]
    if len(path) < 3 or radius <= 0.0:
        return path

    def append_unique(target: list[tuple[float, float]], point: tuple[float, float]) -> None:
        if not target:
            target.append(point)
            return
        if math.hypot(target[-1][0] - point[0], target[-1][1] - point[1]) > 1e-6:
            target.append(point)

    rounded_segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float], int]] = []
    for index, current in enumerate(path):
        previous = path[index - 1]
        following = path[(index + 1) % len(path)]

        in_dx = previous[0] - current[0]
        in_dy = previous[1] - current[1]
        out_dx = following[0] - current[0]
        out_dy = following[1] - current[1]
        in_len = math.hypot(in_dx, in_dy)
        out_len = math.hypot(out_dx, out_dy)
        corner_radius = min(float(radius), in_len * 0.48, out_len * 0.48)

        if corner_radius <= 1e-6 or in_len <= 1e-6 or out_len <= 1e-6:
            rounded_segments.append((current, current, current, 1))
            continue

        start = (current[0] + (in_dx / in_len) * corner_radius, current[1] + (in_dy / in_len) * corner_radius)
        end = (current[0] + (out_dx / out_len) * corner_radius, current[1] + (out_dy / out_len) * corner_radius)
        steps = max(4, int(round(corner_radius / 4.0)))
        rounded_segments.append((start, current, end, steps))

    rounded_path: list[tuple[float, float]] = []
    for start, control, end, steps in rounded_segments:
        append_unique(rounded_path, start)
        for step in range(1, steps):
            t = step / float(steps)
            one_minus_t = 1.0 - t
            append_unique(
                rounded_path,
                (
                    one_minus_t * one_minus_t * start[0] + 2.0 * one_minus_t * t * control[0] + t * t * end[0],
                    one_minus_t * one_minus_t * start[1] + 2.0 * one_minus_t * t * control[1] + t * t * end[1],
                ),
            )
        append_unique(rounded_path, end)

    if len(rounded_path) > 1 and math.hypot(rounded_path[0][0] - rounded_path[-1][0], rounded_path[0][1] - rounded_path[-1][1]) <= 1e-6:
        rounded_path.pop()
    return rounded_path


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
            t = polygon_segment_progress(
                left.points,
                right.points,
                (frame - left.frame) / span,
                easing=getattr(left, "outgoing_easing", POLYGON_EASING_LINEAR),
                constant_area=getattr(left, "outgoing_constant_area", False),
            )
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
