import os
from pathlib import Path

from ultralytics import YOLO

models_folder = Path(os.environ.get("MODELS_FOLDER", "models"))


class BaseImageDetector:

    def __init__(self, model_src: Path, **kwargs):
        self.model = YOLO(model_src, **kwargs)

    def detect(self, src, **kwargs):
        return self.model(src, **kwargs)


class ImageDetectorYOLO11n(BaseImageDetector):
    def __init__(self):
        super().__init__(models_folder / "yolo11n.pt", verbose=False)


class ImageDetectorFireDetectV1(BaseImageDetector):
    def __init__(self):
        super().__init__(models_folder / "fire_detect_v251205_1.pt", verbose=False)
