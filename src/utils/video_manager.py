import json
import subprocess
from pathlib import Path
from typing import List

from src.config.extenstions import redis_client


def stop_key(stream_id: str) -> str:
    return f"stream:stop:{stream_id}"


def stop_worker(stream_id: str):
    redis_client.set(stop_key(stream_id), True)


def should_stop(stream_id: str) -> bool:
    key = redis_client.get(stop_key(stream_id))
    return key is not None and key


def is_stream_source(src: str) -> bool:
    return src.startswith(("rtmp://", "rtsp://", "http://", "https://"))


def create_stream_input_command(src: str):
    return ["ffmpeg", "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay", "-vsync", "drop", "-i",
            src, "-f", "rawvideo", "-pix_fmt", "bgr24", "-", ]


def create_file_input_command(src: Path):
    return ["ffmpeg", "-loglevel", "error", "-re", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "bgr24", "-", ]


def create_hls_output_command(width: int, height: int, hls_dir: Path) -> List[str]:
    return ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "15", "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-f", "hls",
            "-hls_time", "2", "-hls_list_size", "10", "-hls_flags", "delete_segments+append_list",
            str(hls_dir.resolve()), ]


def create_h264_output_command(width: int, height: int, save_path: Path) -> List[str]:
    return ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-i", "-", "-c:v",
            "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(save_path.resolve()), ]


def probe_video_resolution(src: str) -> tuple[int, int]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json",
           src, ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, )

    info = json.loads(result.stdout)
    stream_info = info["streams"][0]

    return int(stream_info["width"]), int(stream_info["height"])
