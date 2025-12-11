from enum import Enum

from src.domains.detect.detector.image_detector import (
    DetectorYOLO,
    DetectorFireDetectV1,
)


class DetectorEnum(Enum):
    YOLO11n = "YOLO11n"
    FireDetectV1 = "fire_detect_v1"


detector_models = {
    DetectorEnum.YOLO11n: DetectorYOLO,
    DetectorEnum.FireDetectV1: DetectorFireDetectV1,
}
