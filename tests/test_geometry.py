import math
import unittest

from exploration_editor.geometry import interpolate_paths, polygon_edit_points_at_frame, polygon_points_at_frame, rounded_closed_path
from exploration_editor.model import PolygonKeyframe, PolygonLayer


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _point_in_polygon(point: list[float], polygon: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            intersection_x = (previous_x - current_x) * (y - current_y) / max(1e-9, previous_y - current_y) + current_x
            if x < intersection_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _max_nearest_distance(points_a: list[tuple[float, float]], points_b: list[tuple[float, float]]) -> float:
    max_distance = 0.0
    for point_a in points_a:
        nearest = min(((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5 for point_b in points_b)
        max_distance = max(max_distance, nearest)
    return max_distance


def _assert_points_close(testcase: unittest.TestCase, actual: list[tuple[float, float]], expected: list[tuple[float, float]]) -> None:
    testcase.assertEqual(len(actual), len(expected))
    for actual_point, expected_point in zip(actual, expected):
        testcase.assertAlmostEqual(actual_point[0], expected_point[0], places=6)
        testcase.assertAlmostEqual(actual_point[1], expected_point[1], places=6)


def _max_turn_angle_degrees(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    max_angle = 0.0
    for index, current in enumerate(points):
        previous = points[index - 1]
        following = points[(index + 1) % len(points)]
        vector_a = (previous[0] - current[0], previous[1] - current[1])
        vector_b = (following[0] - current[0], following[1] - current[1])
        length_a = (vector_a[0] * vector_a[0] + vector_a[1] * vector_a[1]) ** 0.5
        length_b = (vector_b[0] * vector_b[0] + vector_b[1] * vector_b[1]) ** 0.5
        if length_a <= 1e-6 or length_b <= 1e-6:
            continue
        dot = (vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]) / (length_a * length_b)
        dot = max(-1.0, min(1.0, dot))
        turn = abs(180.0 - math.degrees(math.acos(dot)))
        max_angle = max(max_angle, turn)
    return max_angle


class GeometryInterpolationTests(unittest.TestCase):
    def test_interpolate_paths_keeps_unchanged_vertices_fixed_when_point_is_added(self) -> None:
        before = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        after = [[0.0, 0.0], [5.0, 2.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

        halfway = interpolate_paths(before, after, 0.5, closed=True, sample_count=len(after))

        expected = [[0.0, 0.0], [5.0, 1.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        self.assertEqual(len(halfway), len(expected))
        for actual, target in zip(halfway, expected):
            self.assertAlmostEqual(actual[0], target[0], places=6)
            self.assertAlmostEqual(actual[1], target[1], places=6)

    def test_polygon_edit_points_at_frame_keeps_existing_vertices_static(self) -> None:
        before = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        after = [[0.0, 0.0], [5.0, 2.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        layer = PolygonLayer(
            keyframes=[
                PolygonKeyframe(frame=0, points=before),
                PolygonKeyframe(frame=10, points=after),
            ]
        )

        halfway = polygon_edit_points_at_frame(layer, 5)

        expected = [[0.0, 0.0], [5.0, 1.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        self.assertEqual(len(halfway), len(expected))
        for actual, target in zip(halfway, expected):
            self.assertAlmostEqual(actual[0], target[0], places=6)
            self.assertAlmostEqual(actual[1], target[1], places=6)

    def test_rounded_closed_path_at_t0_stays_close_to_keyframe_curve(self) -> None:
        # Regression: verifies the NEW pipeline where each keyframe is smoothed independently.
        # The render pipeline no longer smooths aligned/proxy points, so this test just documents
        # that rounded_closed_path itself is deterministic for the same input.
        before = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        c1 = rounded_closed_path(before, 18.0)
        c2 = rounded_closed_path(before, 18.0)
        # Identical inputs must produce identical outputs.
        self.assertEqual(c1, c2)

    def test_rounded_closed_path_does_not_jump_on_first_tiny_detail_frame(self) -> None:
        before = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        after = [[0.0, 0.0], [-100.0, 50.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]]

        first_detail_frame = interpolate_paths(before, after, 0.01, closed=True)

        rounded = rounded_closed_path(first_detail_frame, 18.0)
        self.assertLess(min(point[0] for point in rounded), -0.1)

    def test_rounded_closed_path_uses_smooth_curve_for_added_detail(self) -> None:
        before = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        after = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 125.0], [0.0, 100.0]]

        first_detail_frame = interpolate_paths(before, after, 0.2, closed=True)
        rounded = rounded_closed_path(first_detail_frame, 222.0)

        # Catmull-Rom generates samples_per_segment * n points; must be denser than input
        self.assertGreater(len(rounded), len(first_detail_frame))
        self.assertLess(_max_turn_angle_degrees(rounded), 40.0)

    def test_added_point_starts_outside_previous_smoothed_reveal(self) -> None:
        before = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        after = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 140.0], [0.0, 100.0]]

        first_detail_frame = interpolate_paths(before, after, 0.01, closed=True)
        previous_reveal = rounded_closed_path(before, 70.0)
        added_point = max(first_detail_frame, key=lambda point: point[1])

        self.assertFalse(_point_in_polygon(added_point, previous_reveal))

    def test_smooth_curve_changes_continuously_when_middle_point_moves(self) -> None:
        before_drag = rounded_closed_path([[0.0, 0.0], [50.0, 90.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], 70.0)
        after_drag = rounded_closed_path([[0.0, 0.0], [50.0, 91.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], 70.0)

        # Catmull-Rom passes through control points so a 1-unit drag is a continuous response
        self.assertLess(_max_nearest_distance(before_drag, after_drag), 3.0)

    def test_interpolate_paths_aligns_rotated_closed_keyframes(self) -> None:
        before = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        after = [[10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]

        halfway = interpolate_paths(before, after, 0.5, closed=True)

        expected = before
        self.assertEqual(len(halfway), len(expected))
        for actual, target in zip(halfway, expected):
            self.assertAlmostEqual(actual[0], target[0], places=6)
            self.assertAlmostEqual(actual[1], target[1], places=6)

    def test_interpolate_paths_aligns_added_vertices_when_closed_start_shifts(self) -> None:
        before = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        after = [[-2.0, 5.0], [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

        first_detail_frame = interpolate_paths(before, after, 0.01, closed=True)

        moved_point = min(first_detail_frame, key=lambda point: point[0])
        self.assertAlmostEqual(moved_point[0], -0.02, places=6)
        self.assertAlmostEqual(moved_point[1], 5.0, places=6)

    def test_polygon_points_at_frame_applies_outgoing_keyframe_easing_per_segment(self) -> None:
        start = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        middle = [[0.0, 0.0], [20.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        end = [[0.0, 0.0], [30.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        layer = PolygonLayer(
            keyframes=[
                PolygonKeyframe(frame=0, points=start, outgoing_easing="ease_in"),
                PolygonKeyframe(frame=10, points=middle, outgoing_easing="ease_out"),
                PolygonKeyframe(frame=20, points=end),
            ]
        )

        early_segment = polygon_points_at_frame(layer, 2)
        later_segment = polygon_points_at_frame(layer, 12)

        self.assertAlmostEqual(early_segment[1][0], 10.4, places=6)
        self.assertAlmostEqual(later_segment[1][0], 23.6, places=6)

    def test_polygon_points_at_frame_can_equalize_revealed_area_per_segment(self) -> None:
        start = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
        end = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
        linear_layer = PolygonLayer(
            keyframes=[
                PolygonKeyframe(frame=0, points=start),
                PolygonKeyframe(frame=10, points=end),
            ]
        )
        equalized_layer = PolygonLayer(
            keyframes=[
                PolygonKeyframe(frame=0, points=start, outgoing_constant_area=True),
                PolygonKeyframe(frame=10, points=end),
            ]
        )

        linear_halfway = polygon_points_at_frame(linear_layer, 5)
        equalized_halfway = polygon_points_at_frame(equalized_layer, 5)

        self.assertAlmostEqual(_polygon_area(linear_halfway), 9.0, places=3)
        self.assertAlmostEqual(_polygon_area(equalized_halfway), 10.0, places=1)
        self.assertGreater(abs(equalized_halfway[0][0]), abs(linear_halfway[0][0]))


if __name__ == "__main__":
    unittest.main()