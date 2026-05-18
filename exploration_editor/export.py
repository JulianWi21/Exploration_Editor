from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

from PIL import Image
from static_ffmpeg import run

from exploration_editor.render import render_frame
from exploration_editor.model import Project


ffmpeg_path, _ffprobe_path = run.get_or_fetch_platform_executables_else_raise()


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


def export_video(
    project: Project,
    basemap_image: Image.Image,
    output_path: str | Path,
    font_path: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        str(output),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    total = project.frame_count
    try:
        for frame_index in range(total):
            image = render_frame(
                project,
                basemap_image=basemap_image,
                frame_index=frame_index,
                output_size=(project.width, project.height),
                font_path=font_path,
                preview=False,
            )
            proc.stdin.write(image.tobytes())
            if progress_callback is not None:
                progress_callback(frame_index + 1, total)
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed with exit code {proc.returncode}")
    return output
