import unittest

from exploration_editor.app import clamp_keyframe_frame


class KeyframeClampTests(unittest.TestCase):
    def test_middle_keyframe_stays_between_neighbors(self) -> None:
        keyframes = [10, 40, 80]

        self.assertEqual(clamp_keyframe_frame(keyframes, 1, 5, 0, 100), 11)
        self.assertEqual(clamp_keyframe_frame(keyframes, 1, 55, 0, 100), 55)
        self.assertEqual(clamp_keyframe_frame(keyframes, 1, 95, 0, 100), 79)

    def test_outer_keyframes_respect_timeline_bounds(self) -> None:
        keyframes = [10, 40, 80]

        self.assertEqual(clamp_keyframe_frame(keyframes, 0, -20, 0, 100), 0)
        self.assertEqual(clamp_keyframe_frame(keyframes, 0, 45, 0, 100), 39)
        self.assertEqual(clamp_keyframe_frame(keyframes, 2, 35, 0, 100), 41)
        self.assertEqual(clamp_keyframe_frame(keyframes, 2, 120, 0, 100), 100)

    def test_single_keyframe_can_move_freely(self) -> None:
        self.assertEqual(clamp_keyframe_frame([15], 0, 95, 0, 100), 95)


if __name__ == "__main__":
    unittest.main()