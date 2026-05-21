import unittest

from exploration_editor.geometry import interpolate_paths, polygon_edit_points_at_frame
from exploration_editor.model import PolygonKeyframe, PolygonLayer


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


if __name__ == "__main__":
    unittest.main()