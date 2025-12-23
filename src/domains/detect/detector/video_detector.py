import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

models_folder = Path(os.environ.get("MODELS_FOLDER", "models"))


class BaseVideoDetector:
    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(model_src, **kwargs)

    def detect(self, src: Path, dest: Path, **kwargs):
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        out = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, **kwargs)
            out.write(results[0].plot())

        out.release()
        cap.release()


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
