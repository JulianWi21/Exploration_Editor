import unittest

from exploration_editor.model import RouteKeyframe, RouteLayer, project_to_dict, route_layer_progress_at_frame


class RouteKeyframeTests(unittest.TestCase):
    def test_route_progress_uses_start_end_without_keyframes(self) -> None:
        layer = RouteLayer(start_frame=10, end_frame=30)

        self.assertEqual(route_layer_progress_at_frame(layer, 0), 0.0)
        self.assertEqual(route_layer_progress_at_frame(layer, 10), 0.0)
        self.assertAlmostEqual(route_layer_progress_at_frame(layer, 20), 0.5)
        self.assertEqual(route_layer_progress_at_frame(layer, 30), 1.0)

    def test_route_progress_interpolates_between_route_keyframes(self) -> None:
        layer = RouteLayer(
            keyframes=[
                RouteKeyframe(frame=0, progress=0.0),
                RouteKeyframe(frame=20, progress=0.25),
                RouteKeyframe(frame=60, progress=1.0),
            ]
        )

        self.assertEqual(route_layer_progress_at_frame(layer, -5), 0.0)
        self.assertAlmostEqual(route_layer_progress_at_frame(layer, 10), 0.125)
        self.assertAlmostEqual(route_layer_progress_at_frame(layer, 40), 0.625)
        self.assertEqual(route_layer_progress_at_frame(layer, 80), 1.0)

    def test_project_to_dict_sorts_route_keyframes_in_project(self) -> None:
        from exploration_editor.model import Project

        project = Project(route_layers=[
            RouteLayer(
                name="Voyage",
                keyframes=[
                    RouteKeyframe(frame=70, progress=1.0),
                    RouteKeyframe(frame=10, progress=0.0),
                ],
            )
        ])

        payload = project_to_dict(project)

        self.assertEqual([item["frame"] for item in payload["route_layers"][0]["keyframes"]], [10, 70])


if __name__ == "__main__":
    unittest.main()