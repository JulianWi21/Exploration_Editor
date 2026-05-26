import unittest

from PIL import Image

from exploration_editor.model import PolygonKeyframe, PolygonLayer, TextOverlayKeyframe, ViewState
from exploration_editor.render import clamp_view_state, compute_map_layout, polygon_outline_screen_paths_at_frame
from exploration_editor.model import Project, TextOverlayLayer
from exploration_editor.render import render_frame


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

    def test_render_frame_draws_text_overlay_in_expected_corner(self) -> None:
        project = Project(
            width=400,
            height=200,
            fog_opacity=0.0,
            text_layers=[
                TextOverlayLayer(
                    name="Year",
                    template="Year {time_label}",
                    anchor="top_right",
                    offset_x=-0.05,
                    offset_y=0.05,
                    alignment="right",
                    color=[0, 0, 0],
                    background_color=[255, 0, 0],
                    background_opacity=1.0,
                    border_opacity=0.0,
                    font_size=44,
                    padding_x=12,
                    padding_y=8,
                    frame_start=0,
                    frame_end=5,
                )
            ],
        )
        basemap = Image.new("RGB", (400, 200), (230, 230, 230))

        image = render_frame(project, basemap_image=basemap, frame_index=0, output_size=(400, 200), preview=True)
        pixels = image.load()
        red_pixels = 0
        for x in range(220, 390):
            for y in range(0, 90):
                if pixels[x, y][0] > 230 and pixels[x, y][1] < 80 and pixels[x, y][2] < 80:
                    red_pixels += 1

        self.assertGreater(red_pixels, 200)

    def test_render_frame_hides_text_overlay_outside_frame_range(self) -> None:
        project = Project(
            width=320,
            height=180,
            fog_opacity=0.0,
            text_layers=[
                TextOverlayLayer(
                    name="Year",
                    template="Overlay",
                    anchor="top_left",
                    offset_x=0.05,
                    offset_y=0.05,
                    color=[0, 0, 0],
                    background_color=[255, 0, 0],
                    background_opacity=1.0,
                    border_opacity=0.0,
                    frame_start=10,
                    frame_end=20,
                )
            ],
        )
        basemap = Image.new("RGB", (320, 180), (230, 230, 230))

        visible = render_frame(project, basemap_image=basemap, frame_index=10, output_size=(320, 180), preview=True)
        hidden = render_frame(project, basemap_image=basemap, frame_index=5, output_size=(320, 180), preview=True)

        self.assertNotEqual(list(visible.getdata()), list(hidden.getdata()))

    def test_render_frame_uses_text_keyframes_for_content_switches(self) -> None:
        project = Project(
            width=320,
            height=180,
            fog_opacity=0.0,
            text_layers=[
                TextOverlayLayer(
                    name="Label",
                    template="EARLY",
                    anchor="top_left",
                    offset_x=0.05,
                    offset_y=0.05,
                    color=[0, 0, 0],
                    background_color=[255, 0, 0],
                    background_opacity=1.0,
                    border_opacity=0.0,
                    text_keyframes=[
                        TextOverlayKeyframe(frame=10, template="LATE STAGE"),
                    ],
                )
            ],
        )
        basemap = Image.new("RGB", (320, 180), (230, 230, 230))

        early = render_frame(project, basemap_image=basemap, frame_index=0, output_size=(320, 180), preview=True)
        late = render_frame(project, basemap_image=basemap, frame_index=10, output_size=(320, 180), preview=True)

        self.assertNotEqual(list(early.getdata()), list(late.getdata()))


if __name__ == "__main__":
    unittest.main()