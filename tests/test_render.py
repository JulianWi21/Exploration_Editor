import unittest

from exploration_editor.model import PolygonKeyframe, PolygonLayer, ViewState
from exploration_editor.render import clamp_view_state, compute_map_layout, polygon_outline_screen_paths_at_frame


class RenderViewTests(unittest.TestCase):
    def test_clamp_view_state_wraps_horizontal_offset(self) -> None:
        frame_size = (512, 512)
        world_size = (8192, 4096)
        wrapped = clamp_view_state(frame_size, world_size, ViewState(zoom=1.0, offset_x=900.0, offset_y=0.0))

        layout = compute_map_layout(frame_size, world_size, wrapped)

        self.assertLessEqual(layout["offset_x"], 0.0)
        self.assertGreater(layout["offset_x"], -layout["draw_w"])

    def test_rounded_polygon_outline_avoids_first_frame_topology_jump(self) -> None:
        layout = {
            "frame_w": 360,
            "frame_h": 180,
            "world_w": 360,
            "world_h": 180,
            "scale": 1.0,
            "draw_w": 360.0,
            "draw_h": 180.0,
            "offset_x": -180.0,
            "offset_y": -90.0,
        }
        layer = PolygonLayer(
            rounding_px=222,
            keyframes=[
                PolygonKeyframe(frame=0, points=[[0.0, 0.0], [60.0, 0.0], [60.0, -60.0], [0.0, -60.0]]),
                PolygonKeyframe(frame=100, points=[[-30.0, -30.0], [0.0, 0.0], [60.0, 0.0], [60.0, -60.0], [0.0, -60.0]]),
            ],
        )

        exact_path = polygon_outline_screen_paths_at_frame(layer, 0, layout)[1]
        first_path = polygon_outline_screen_paths_at_frame(layer, 1, layout)[1]

        exact_min_x = min(point[0] for point in exact_path)
        first_min_x = min(point[0] for point in first_path)
        # With Catmull-Rom the proxy point (on the source edge) can slightly change the
        # overshoot at the corner, but the jump must stay well below the old Chaikin jump (~8px).
        self.assertLess(abs(first_min_x - exact_min_x), 6.0)

    def test_rounded_polygon_outline_keeps_unchanged_section_stable(self) -> None:
        layout = {
            "frame_w": 360,
            "frame_h": 180,
            "world_w": 360,
            "world_h": 180,
            "scale": 1.0,
            "draw_w": 360.0,
            "draw_h": 180.0,
            "offset_x": -180.0,
            "offset_y": -90.0,
        }
        layer = PolygonLayer(
            rounding_px=222,
            keyframes=[
                PolygonKeyframe(frame=0, points=[[0.0, 0.0], [80.0, 0.0], [80.0, -80.0], [0.0, -80.0]]),
                PolygonKeyframe(frame=100, points=[[-90.0, -35.0], [-60.0, -70.0], [0.0, 0.0], [80.0, 0.0], [80.0, -80.0], [0.0, -80.0]]),
            ],
        )

        exact_path = polygon_outline_screen_paths_at_frame(layer, 0, layout)[1]
        halfway_path = polygon_outline_screen_paths_at_frame(layer, 50, layout)[1]

        self.assertAlmostEqual(max(point[0] for point in halfway_path), max(point[0] for point in exact_path), delta=1.0)


if __name__ == "__main__":
    unittest.main()