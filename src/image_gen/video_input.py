"""Operations that turn a user-supplied input video into generation inputs.

Today the only operation is `last_frame`: take the final frame of the uploaded video
and use it as the first frame of a new generation, which reuses the existing img2img
start-image path. The uploaded video is then prepended to the generated one so the
result plays as a single continuous clip. Additional operations (e.g. restyling or
continuing a whole clip) register here and are selected by
`ImageWorkflow.input_video_type`.
"""

import json
import os
import subprocess

from src.image_gen.ImageWorkflow import VideoInputType


def extract_last_frame(video_path: str) -> str:
    """Write the final frame of `video_path` to a PNG next to it and return its path."""
    out_path = os.path.splitext(video_path)[0] + "_last_frame.png"
    # -sseof seeks relative to the end; -update overwrites the output for each decoded
    # frame, so what survives is the last one. A plain -frames:v 1 would instead grab
    # the frame at the seek point rather than the true final frame.
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-3",
        "-i", video_path,
        "-update", "1",
        "-f", "image2",
        "-c:v", "png",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg produced no frame for {video_path}")
    return os.path.abspath(out_path)


# Each handler takes the saved video path and returns the image path to use as the
# generation's starting frame.
_VIDEO_INPUT_HANDLERS = {
    VideoInputType.last_frame: extract_last_frame,
}


def resolve_video_input(video_path: str, input_video_type: VideoInputType = VideoInputType.last_frame) -> str:
    """Run the requested video operation and return the resulting start-frame image."""
    handler = _VIDEO_INPUT_HANDLERS.get(input_video_type)
    if handler is None:
        raise ValueError(f"Unsupported input_video_type: {input_video_type}")
    return handler(video_path)


def _probe(video_path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", video_path],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def _video_stream(probe: dict) -> dict:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise RuntimeError("No video stream found")


def _has_audio(probe: dict) -> bool:
    return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))


def _frame_rate(stream: dict) -> float:
    num, _, den = stream.get("r_frame_rate", "24/1").partition("/")
    den = float(den or 1)
    return float(num) / den if den else 24.0


def concat_videos(first_path: str, second_path: str, out_path: str) -> str:
    """Join two videos into one, conforming the first to the second's format.

    The two clips come from different sources (a user upload and a fresh generation),
    so they can differ in resolution, frame rate, codec and audio layout. Everything is
    re-encoded through the concat filter against the generated clip's format rather than
    stream-copied, which is what makes a mismatched upload joinable at all. A clip
    without an audio track gets a silent one, since concat requires both inputs to carry
    the same streams.
    """
    first_probe = _probe(first_path)
    second_probe = _probe(second_path)

    target = _video_stream(second_probe)
    width, height = int(target["width"]), int(target["height"])
    fps = _frame_rate(target)

    inputs = ["-i", first_path, "-i", second_path]
    filters = []
    silent_input_index = 2

    for index, (path, probe) in enumerate([(first_path, first_probe), (second_path, second_probe)]):
        filters.append(
            f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},format=yuv420p[v{index}]"
        )
        if _has_audio(probe):
            audio_source = f"{index}:a"
        else:
            # A finite silent track: anullsrc is infinite, and an unbounded stream would
            # make the concat filter hang instead of ending with the clip.
            duration = _stream_duration(path)
            inputs.extend(["-f", "lavfi", "-t", f"{duration}", "-i", "anullsrc=r=48000:cl=stereo"])
            audio_source = f"{silent_input_index}:a"
            silent_input_index += 1
        filters.append(
            f"[{audio_source}]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{index}]"
        )

    filters.append("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filters),
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg produced no output joining {first_path} and {second_path}")
    return os.path.abspath(out_path)


def _stream_duration(video_path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())
