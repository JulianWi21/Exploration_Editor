from __future__ import annotations

import os
import random
from pathlib import Path
import subprocess
from typing import Callable

import numpy as np
from PIL import Image
from static_ffmpeg import run

from exploration_editor.basemap import repo_root
from exploration_editor.render import render_frame
from exploration_editor.model import Project


ffmpeg_path, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

VIDEO_FADE_DURATION_SEC = 2.0
VIDEO_HOLD_START_SEC = 0.0
VIDEO_HOLD_END_SEC = 3.0
AUDIO_FADE_IN_SEC = 2.0
AUDIO_FADE_OUT_SEC = 3.0


def export_frame_png(
    project: Project,
    basemap_image: Image.Image,
    frame_index: int,
    output_path: str | Path,
    font_path: str | Path | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = render_frame(
        project,
        basemap_image=basemap_image,
        frame_index=frame_index,
        output_size=(project.width, project.height),
        font_path=font_path,
        preview=False,
    )
    image.save(output)
    return output


def export_frame_count(project: Project) -> int:
    fade_in_frames = int(VIDEO_FADE_DURATION_SEC * int(project.fps))
    hold_start_frames = int(VIDEO_HOLD_START_SEC * int(project.fps))
    hold_end_frames = int(VIDEO_HOLD_END_SEC * int(project.fps))
    fade_out_frames = int(VIDEO_FADE_DURATION_SEC * int(project.fps))
    return fade_in_frames + hold_start_frames + int(project.frame_count) + hold_end_frames + fade_out_frames


def _soundtrack_dir() -> Path:
    env_override = os.environ.get("EXPLORATION_EDITOR_SOUNDTRACK_DIR")
    if env_override:
        return Path(env_override)
    return repo_root().parent.parent / "GEOPANDA" / "sound" / "soundtrack"


def _get_random_soundtrack(soundtrack_dir: Path) -> Path:
    candidates = sorted(soundtrack_dir.glob("*.mp3"))
    if not candidates:
        raise RuntimeError(f"No MP3 files found in soundtrack directory: {soundtrack_dir}")
    return random.choice(candidates)


def _get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def _apply_black_fade(image: Image.Image, opacity: float) -> Image.Image:
    alpha = max(0.0, min(1.0, float(opacity)))
    if alpha >= 1.0:
        return image
    if alpha <= 0.0:
        return Image.new("RGB", image.size, (0, 0, 0))
    faded = (np.asarray(image, dtype=np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(faded, "RGB")


def _combine_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    video_duration = _get_video_duration(video_path)
    fade_out_start = max(0.0, video_duration - AUDIO_FADE_OUT_SEC)
    audio_filter = (
        f"afade=t=in:st=0:d={AUDIO_FADE_IN_SEC},"
        f"afade=t=out:st={fade_out_start}:d={AUDIO_FADE_OUT_SEC}"
    )
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-stream_loop",
            "-1",
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a]atrim=0:{video_duration},{audio_filter}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def export_video(
    project: Project,
    basemap_image: Image.Image,
    output_path: str | Path,
    font_path: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent_output = output.with_name(f"{output.stem}.silent{output.suffix}")
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{project.width}x{project.height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(project.fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(silent_output),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    fade_in_frames = int(VIDEO_FADE_DURATION_SEC * int(project.fps))
    hold_start_frames = int(VIDEO_HOLD_START_SEC * int(project.fps))
    hold_end_frames = int(VIDEO_HOLD_END_SEC * int(project.fps))
    fade_out_frames = int(VIDEO_FADE_DURATION_SEC * int(project.fps))
    render_start = fade_in_frames + hold_start_frames
    render_end = render_start + int(project.frame_count)
    fade_out_start = render_end + hold_end_frames
    total = export_frame_count(project)
    last_frame_index = max(0, int(project.frame_count) - 1)

    try:
        for frame_index in range(total):
            if frame_index < render_start:
                source_frame_index = 0
            elif frame_index < render_end:
                source_frame_index = frame_index - render_start
            else:
                source_frame_index = last_frame_index

            image = render_frame(
                project,
                basemap_image=basemap_image,
                frame_index=source_frame_index,
                output_size=(project.width, project.height),
                font_path=font_path,
                preview=False,
            )
            if fade_in_frames > 0 and frame_index < fade_in_frames:
                image = _apply_black_fade(image, frame_index / fade_in_frames)
            elif fade_out_frames > 0 and frame_index >= fade_out_start:
                image = _apply_black_fade(image, max(0.0, 1.0 - ((frame_index - fade_out_start) / fade_out_frames)))
            proc.stdin.write(image.tobytes())
            if progress_callback is not None:
                progress_callback(frame_index + 1, total)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        if silent_output.exists():
            silent_output.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg export failed with exit code {proc.returncode}")

    soundtrack_dir = _soundtrack_dir()
    try:
        if soundtrack_dir.exists():
            soundtrack = _get_random_soundtrack(soundtrack_dir)
            _combine_video_audio(silent_output, soundtrack, output)
            silent_output.unlink(missing_ok=True)
        else:
            silent_output.replace(output)
    except Exception:
        if output.exists():
            output.unlink(missing_ok=True)
        if silent_output.exists():
            silent_output.unlink(missing_ok=True)
        raise

    return output
