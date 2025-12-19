from enum import Enum

from src.domains.detect.detector.image_detector import (
    BaseImageDetector,
    ImageDetectorYOLO11n,
    ImageDetectorFireDetectV1,
)
from src.domains.detect.detector.video_detector import (
    VideoDetectorFireDetectV1,
    BaseVideoDetector,
    VideoDetectorYolo11n,
    VideoDetectorShoulderStop,
)


class ImageDetectorEnum(Enum):
    YOLO11n = "yolo11n"
    FireDetectV1 = "fire_detect_v1"


image_detector_models = {
    ImageDetectorEnum.YOLO11n: ImageDetectorYOLO11n,
    ImageDetectorEnum.FireDetectV1: ImageDetectorFireDetectV1,
}


class VideoDetectorEnum(Enum):
    YOLO11n = "yolo11n"
    FireDetectV1 = "fire_detect_v1"
    ShoulderStop = "shoulder_stop"


video_detector_models = {
    VideoDetectorEnum.YOLO11n: VideoDetectorYolo11n,
    VideoDetectorEnum.FireDetectV1: VideoDetectorFireDetectV1,
    VideoDetectorEnum.ShoulderStop: VideoDetectorShoulderStop,
}
