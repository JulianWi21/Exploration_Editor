import unittest

from exploration_editor.geometry import interpolate_paths, polygon_edit_points_at_frame, polygon_points_at_frame
from exploration_editor.model import PolygonKeyframe, PolygonLayer


def _polygon_area(points: list[list[float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


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