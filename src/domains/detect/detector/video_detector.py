import os
from pathlib import Path

import cv2
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
