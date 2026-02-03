from pathlib import Path

from celery import shared_task
from flask import current_app

from src.domains.detect.models import DetectionVideo, TaskStatus
from src.main import db
from src.utils.detector import video_detector_models, VideoDetectorEnum
from src.utils.video_manager import is_stream_source, create_stream_input_command, probe_video_resolution, \
    create_hls_output_command, create_file_input_command, create_h264_output_command


def change_detection_video_status(detection_video_id: int, status: TaskStatus):
    detection_video: DetectionVideo | None = DetectionVideo.query.get(detection_video_id)
    if detection_video:
        current_app.logger.info(f"Change detection video '{detection_video.id}' status to: {status}")
        detection_video.status = status
        db.session.commit()


@shared_task(ignore_result=False)
def create_detect_task(src: str, dest: str, selected_model: str, detection_video_id: int):
    current_app.logger.info(f"Start detecting videos: {src}")
    current_app.logger.info(f"Detection video id: {detection_video_id}")

    # detection_video: DetectionVideo | None = DetectionVideo.query.get(detection_video_id)
    # if detection_video:
    #     current_app.logger.info(f"Change detection video status to: {TaskStatus.STARTED}")
    #     detection_video.status = TaskStatus.STARTED
    #     db.session.commit()
    change_detection_video_status(detection_video_id, TaskStatus.STARTED)

    try:
        src_path = Path(src)
        dest_path = Path(dest)

        if is_stream_source(str(src_path)):
            input_cmd = create_stream_input_command(src_path)
            width, height = probe_video_resolution(str(src_path))
            output_cmd = create_hls_output_command(width, height, dest_path)
        else:
            input_cmd = create_file_input_command(src_path)
            width, height = probe_video_resolution(str(src_path))
            output_cmd = create_h264_output_command(width, height, dest_path)

        detector = video_detector_models[VideoDetectorEnum(selected_model)]()
        # if not torch.backends.mps.is_available():
        #     if not torch.backends.mps.is_built():
        #         print("MPS not available because the current PyTorch install was not "
        #               "built with MPS enabled.")
        #     else:
        #         print("MPS not available because the current MacOS version is not 12.3+ "
        #               "and/or you do not have an MPS-enabled device on this machine.")
        #     mps_device = torch.device("cpu")
        # else:
        #     mps_device = torch.device("mps")
        detector.detect(input_cmd, output_cmd, width, height, detection_video_id, conf=0.5, verbose=False)

        current_app.logger.info(f"End detecting videos: {src_path}")
        # if detection_video:
        #     current_app.logger.info(f"Change detection video status to: {TaskStatus.SUCCESS}")
        #     detection_video.status = TaskStatus.SUCCESS
        #     db.session.commit()
        change_detection_video_status(detection_video_id, TaskStatus.SUCCESS)
    except Exception as e:
        current_app.logger.error(f"Error detecting videos: {e}")
        # if detection_video:
        #     current_app.logger.info(f"Change detection video status to: {TaskStatus.FAILURE}")
        #     detection_video.status = TaskStatus.FAILURE
        #     db.session.commit()
        change_detection_video_status(detection_video_id, TaskStatus.FAILURE)
