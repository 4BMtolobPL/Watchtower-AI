import json
import os
import subprocess
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

models_folder = Path(os.environ.get("MODELS_FOLDER", "models"))


class BaseVideoDetector:
    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(model_src, **kwargs)

    @staticmethod
    def create_h264_output_command(
        width: int, height: int, save_path: Path
    ) -> List[str]:
        return [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-profile:v",
            "main",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(save_path.resolve()),
        ]

    @staticmethod
    def create_file_input_command(path: Path):
        return [
            "ffmpeg",
            "-loglevel",
            "error",
            "-re",
            "-i",
            str(path.resolve()),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]

    @staticmethod
    def probe_video_resolution(src: str) -> tuple[int, int]:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            src,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        info = json.loads(result.stdout)
        stream_info = info["streams"][0]

        return int(stream_info["width"]), int(stream_info["height"])

    @staticmethod
    def terminate_process(proc: subprocess.Popen):
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    def detect(self, src: Path, dest: Path, **kwargs):
        input_cmd = self.create_file_input_command(Path(src))
        width, height = self.probe_video_resolution(str(src))
        output_cmd = self.create_h264_output_command(width, height, dest)

        # Input, Output 스트림
        ffmpeg_in = subprocess.Popen(
            input_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        ffmpeg_out = subprocess.Popen(
            output_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        assert ffmpeg_in.stdout is not None
        assert ffmpeg_out.stdin is not None

        frame_size = width * height * 3

        try:
            while True:
                raw = ffmpeg_in.stdout.read(frame_size)
                if not raw:
                    break

                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))

                results = self.model(frame, **kwargs)
                annotated_frame = results[0].plot()

                ffmpeg_out.stdin.write(annotated_frame.tobytes())
                ffmpeg_out.stdin.flush()
            ffmpeg_out.stdin.close()
        finally:
            self.terminate_process(ffmpeg_in)
            self.terminate_process(ffmpeg_out)


class VideoDetectorYolo11n(BaseVideoDetector):
    def __init__(self):
        super().__init__(models_folder / "yolo11n.pt", verbose=False)


class VideoDetectorFireDetectV1(BaseVideoDetector):
    def __init__(self):
        super().__init__(models_folder / "fire_detect_v251205_1.pt", verbose=False)


# 갓길 정차 차량 감지 로직
class VideoDetectorShoulderStop:
    def __init__(self):
        self.model = YOLO(models_folder / "yolov8s.pt", verbose=False)

        self.STOP_DISTANCE = 3
        self.STOP_TIME = 0.05
        self.SKIP = 4

        # 고정좌표
        self.poly = np.array(
            [
                (459, 359),
                (519, 364),
                (283, 580),
                (110, 544),
            ],
            dtype=np.int32,
        )

        self.car_status = {}
        self.stopped_ids = set()
        self.stop_time_map = {}
        self.frame_count = 0
        self.last_boxes = []
        self.last_ids = []
        self.last_roi_occupied = False

    def detect(self, src: Path, dest: Path, **kwargs):
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(
            str(dest),
            cv2.VideoWriter.fourcc(*"avc1"),
            fps,
            (width, height),
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1

            if self.frame_count % self.SKIP != 0:
                annotated = frame.copy()

                for box, tid in zip(self.last_boxes, self.last_ids):
                    x1, y1, x2, y2 = map(int, box)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

                roi_color = (0, 0, 255) if self.last_roi_occupied else (0, 255, 0)
                cv2.polylines(annotated, [self.poly], True, roi_color, 2)

                out.write(annotated)
                continue

            annotated = frame.copy()
            current_time = time.time()
            roi_occupied = False

            results = self.model.track(
                frame,
                persist=True,
                conf=0.05,
                verbose=False,
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                ids = results[0].boxes.id.int().cpu().tolist()

                self.last_boxes = boxes
                self.last_ids = ids

                for box, tid in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    corners = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]

                    inside = any(
                        cv2.pointPolygonTest(self.poly, corner, False) >= 0
                        for corner in corners
                    )

                    if not inside:
                        continue

                    roi_occupied = True

                    # 정차 판단 부분
                    if tid not in self.car_status:
                        self.car_status[tid] = (cx, cy, current_time)
                    else:
                        px, py, last_time = self.car_status[tid]
                        move_dist = abs(cx - px) + abs(cy - py)

                        if move_dist < self.STOP_DISTANCE:
                            if current_time - last_time >= self.STOP_TIME:
                                self.stopped_ids.add(tid)
                                self.stop_time_map.setdefault(tid, current_time)
                        else:
                            self.car_status[tid] = (cx, cy, current_time)

                    color = (0, 0, 255) if tid in self.stopped_ids else (0, 255, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated,
                        f"ID:{tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )

            self.last_roi_occupied = roi_occupied

            roi_color = (0, 0, 255) if roi_occupied else (0, 255, 0)
            cv2.polylines(annotated, [self.poly], True, roi_color, 2)
            out.write(annotated)

        cap.release()
        out.release()
