import unittest

from exploration_editor.model import Project, TextOverlayKeyframe, TextOverlayLayer, TimeKeyframe, build_text_template_context, interpolate_project_year, project_time_label, project_to_dict, render_text_template, render_text_template_html, text_overlay_template_at_frame, text_template_is_rich


class TextOverlayTemplateTests(unittest.TestCase):
    def test_interpolate_project_year_uses_linear_segments(self) -> None:
        project = Project(
            fps=30,
            duration_sec=10.0,
            time_keyframes=[
                TimeKeyframe(frame=0, year=-200000),
                TimeKeyframe(frame=100, year=-100000),
                TimeKeyframe(frame=200, year=2026),
            ],
        )

        self.assertEqual(interpolate_project_year(project, -10), -200000.0)
        self.assertEqual(interpolate_project_year(project, 0), -200000.0)
        self.assertEqual(interpolate_project_year(project, 50), -150000.0)
        self.assertEqual(interpolate_project_year(project, 150), -48987.0)
        self.assertEqual(interpolate_project_year(project, 500), 2026.0)

    def test_project_time_label_prefers_exact_keyframe_label(self) -> None:
        project = Project(
            time_keyframes=[
                TimeKeyframe(frame=0, year=-200000, label="Origin"),
                TimeKeyframe(frame=100, year=2026),
            ],
        )

        self.assertEqual(project_time_label(project, 0), "Origin")
        self.assertEqual(project_time_label(project, 50), "98,987 BC")
        self.assertEqual(project_time_label(project, 100), "2,026")

    def test_render_text_template_substitutes_supported_placeholders(self) -> None:
        project = Project(
            title="Homo Sapiens Expansion",
            fps=24,
            duration_sec=10.0,
            time_keyframes=[
                TimeKeyframe(frame=0, year=-200000),
                TimeKeyframe(frame=239, year=2026),
            ],
        )

        rendered = render_text_template(
            "{project_title} | {frame}/{frame_max} | {progress_pct} | {year} | {time_label} | {unknown}",
            project,
            120,
        )

        self.assertIn("Homo Sapiens Expansion", rendered)
        self.assertIn("120/239", rendered)
        self.assertIn("50%", rendered)
        self.assertIn("-98564", rendered)
        self.assertIn("98,564 BC", rendered)
        self.assertIn("{unknown}", rendered)

    def test_render_text_template_html_escapes_placeholder_values(self) -> None:
        project = Project(
            title="A&B",
            time_keyframes=[TimeKeyframe(frame=0, year=2026, label="Now & <Soon>")],
        )

        rendered = render_text_template_html("<p>{project_title} | {time_label}</p>", project, 0)

        self.assertIn("A&amp;B", rendered)
        self.assertIn("Now &amp; &lt;Soon&gt;", rendered)

    def test_text_template_is_rich_detects_html_markup(self) -> None:
        self.assertTrue(text_template_is_rich("<p><span style='color: red;'>Styled</span></p>"))
        self.assertFalse(text_template_is_rich("Plain text only"))

    def test_template_context_exposes_blank_time_fields_without_time_track(self) -> None:
        project = Project(title="Untitled", fps=30, duration_sec=5.0)

        context = build_text_template_context(project, 15)

        self.assertEqual(context["year"], "")
        self.assertEqual(context["time_label"], "")
        self.assertEqual(context["progress_pct"], "10%")

    def test_text_overlay_template_at_frame_uses_latest_keyframe_override(self) -> None:
        layer = TextOverlayLayer(
            template="Year: {time_label}",
            text_keyframes=[
                TextOverlayKeyframe(frame=40, template="Migration Begins"),
                TextOverlayKeyframe(frame=80, template="Year: {time_label}"),
            ],
        )

        self.assertEqual(text_overlay_template_at_frame(layer, 0), "Year: {time_label}")
        self.assertEqual(text_overlay_template_at_frame(layer, 39), "Year: {time_label}")
        self.assertEqual(text_overlay_template_at_frame(layer, 40), "Migration Begins")
        self.assertEqual(text_overlay_template_at_frame(layer, 70), "Migration Begins")
        self.assertEqual(text_overlay_template_at_frame(layer, 80), "Year: {time_label}")

    def test_project_to_dict_sorts_text_keyframes_per_layer(self) -> None:
        project = Project(
            text_layers=[
                TextOverlayLayer(
                    name="Overlay",
                    text_keyframes=[
                        TextOverlayKeyframe(frame=80, template="Late"),
                        TextOverlayKeyframe(frame=20, template="Early"),
                    ],
                )
            ]
        )

        payload = project_to_dict(project)

        self.assertEqual([item["frame"] for item in payload["text_layers"][0]["text_keyframes"]], [20, 80])


if __name__ == "__main__":
    unittest.main()