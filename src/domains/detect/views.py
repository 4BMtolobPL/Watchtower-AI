import uuid
from pathlib import Path
from typing import List

import cv2
from celery import shared_task
from celery.result import AsyncResult
from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from ultralytics.engine.results import Results

from src.domains.detect.detector import (
    ImageDetectorEnum,
    VideoDetectorEnum,
    image_detector_models,
    video_detector_models,
)
from src.domains.detect.forms import (
    DetectImageForm,
    DetectVideoForm,
    UploadImageForm,
    UploadVideoForm,
)
from src.domains.detect.models import (
    DetectionImage,
    DetectionVideo,
    UserImage,
    UserVideo,
)
from src.main import db

detect_views = Blueprint(
    "detect", __name__, template_folder="templates", static_folder="static"
)


@detect_views.get("/images")
def image_dashboard():
    user_images: List[UserImage] = UserImage.query.order_by(UserImage.id.desc()).all()

    return render_template("detect/image_dashboard.html", user_images=user_images)


@detect_views.get("/images/<path:filename>")
def images(filename: str):
    return send_from_directory(
        Path(current_app.config["UPLOAD_FOLDER"], "images"), filename
    )


@detect_views.get("/images/<int:image_id>/status")
def image_status(image_id: int):
    model = request.args.get("model")

    detection_image: DetectionImage | None = (
        DetectionImage.query.filter(DetectionImage.user_image_id == image_id)
        .filter(DetectionImage.model == model)
        .order_by(DetectionImage.id.desc())
        .first()
    )

    if detection_image is None:
        return jsonify({"status": "NONE"})
    else:
        return jsonify({"status": "SUCCESS", "image_path": detection_image.image_path})


@detect_views.route("/upload/images", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()

    if form.validate_on_submit():
        file = form.image.data

        ext = Path(file.filename).suffix
        image_uuid_file_name = str(uuid.uuid4()) + ext
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"], "images", image_uuid_file_name
        )
        file.save(image_path)

        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("detect.image_dashboard"))
    return render_template("detect/upload_images.html", form=form)


@detect_views.route("/images/detail/<int:image_id>", methods=["GET", "POST"])
def image_detail(image_id: int):
    user_image: UserImage = db.get_or_404(UserImage, image_id)
    form = DetectImageForm()
    form.image_id.data = image_id

    if form.validate_on_submit():
        selected_model = form.model.data
        if selected_model not in [de.value for de in ImageDetectorEnum]:
            return "잘못된 모델입니다.", 400

        ext = Path(str(user_image.image_path)).suffix
        base = Path(Path(current_app.config["UPLOAD_FOLDER"], "images"))
        dest = str(uuid.uuid4()) + ext

        detector = image_detector_models[(ImageDetectorEnum(selected_model))]()
        results: List[Results] = detector.detect(base / user_image.image_path)
        results[0].save(str(base / dest))
        current_app.logger.info(f"File saved: {str(base / dest)}")

        detection_image = DetectionImage(
            model=selected_model,
            image_path=dest,
            user_image=user_image,
        )
        db.session.add(detection_image)
        db.session.commit()

        return jsonify({"image_path": dest})

    return render_template("detect/image_detail.html", user_image=user_image, form=form)


@detect_views.get("/videos")
def video_dashboard():
    user_videos: List[UserVideo] = UserVideo.query.order_by(UserVideo.id.desc()).all()

    return render_template("detect/video_dashboard.html", user_videos=user_videos)


@detect_views.get("/videos/<path:filename>")
def videos(filename: str):
    return send_from_directory(
        Path(current_app.config["UPLOAD_FOLDER"], "videos"), filename
    )


@detect_views.post("/videos/task")
def check_task():
    json = request.get_json()

    if not json:
        return jsonify({"error": "No JSON data provided"}), 400

    video_id = json.get("video_id")
    model = json.get("model")

    if not video_id or not model:
        return jsonify({"error": "Missing 'video_id' or 'model'"}), 400

    detection_video: DetectionVideo | None = (
        DetectionVideo.query.filter_by(user_video_id=video_id)
        .filter_by(model=model)
        .order_by(DetectionVideo.id.desc())
        .first()
    )

    if not detection_video:
        return jsonify({"status": "None"})

    return jsonify(
        {
            "status": "Ok",
            "task_id": detection_video.task_id,
            "video_path": detection_video.video_path,
        }
    )


@detect_views.get("/videos/thumbnail/<path:thumbnail>")
def thumbnails(thumbnail: str):
    return send_from_directory(
        Path(current_app.config["UPLOAD_FOLDER"], "videos"), thumbnail
    )


@detect_views.route("/upload/videos", methods=["GET", "POST"])
@login_required
def upload_video():
    form = UploadVideoForm()

    if form.validate_on_submit():
        file = form.video.data

        ext = Path(file.filename).suffix
        video_uuid_file_name = str(uuid.uuid4()) + ext
        video_path = Path(
            current_app.config["UPLOAD_FOLDER"], "videos", video_uuid_file_name
        )
        file.save(video_path)

        thumbnail_path = extract_thumbnail(video_path)

        user_video = UserVideo(
            user_id=current_user.id,
            video_path=video_uuid_file_name,
            thumbnail_path=thumbnail_path,
        )
        db.session.add(user_video)
        db.session.commit()

        return redirect(url_for("detect.video_dashboard"))
    return render_template("detect/upload_videos.html", form=form)


@detect_views.route("/videos/detail/<int:video_id>", methods=["GET", "POST"])
def video_detail(video_id: int):
    user_video: UserVideo = db.get_or_404(UserVideo, video_id)
    form = DetectVideoForm()
    form.video_id.data = user_video.id

    if form.validate_on_submit():
        selected_model = form.model.data
        if selected_model not in [de.value for de in VideoDetectorEnum]:
            return "잘못된 모델입니다.", 400

        dest = str(uuid.uuid4())

        result = detect_videos.delay(
            str(Path(current_app.config["UPLOAD_FOLDER"], "videos")),
            str(user_video.video_path),
            dest,
            selected_model,
        )
        current_app.logger.info(f"Task 추가: {result.id}")

        detection_video = DetectionVideo(
            model=selected_model,
            task_id=result.id,
            video_path=f"{dest}.mp4",
            user_video=user_video,
        )
        db.session.add(detection_video)
        db.session.commit()

        return jsonify(
            {
                "result_id": detection_video.task_id,
                "video_path": detection_video.video_path,
            }
        )

    return render_template("detect/video_detail.html", user_video=user_video, form=form)


def extract_thumbnail(video_path: Path) -> str | None:
    save_path = video_path.parent
    thumbnail_name = f"{video_path.name}.webp"
    cap = cv2.VideoCapture(str(video_path))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(save_path / thumbnail_name), frame)
    else:
        return None

    cap.release()
    return thumbnail_name


@shared_task(ignore_result=False)
def detect_videos(base: str, src: str, dest_name: str, selected_model: str):
    current_app.logger.info(f"Detecting videos: {src}")
    base_path: Path = Path(base)

    detector = video_detector_models[VideoDetectorEnum(selected_model)]()
    detector.detect(
        base_path / src, base_path / f"{dest_name}.mp4", conf=0.5, verbose=False
    )

    current_app.logger.info(f"End detecting videos: {dest_name}.mp4")
    return dest_name


@detect_views.get("/result/<string:result_id>")
def task_result(result_id: str):
    result = AsyncResult(result_id)

    if result.failed():
        current_app.logger.error(f"Task failed: {result_id}\n{result.info}")
        return (
            jsonify(
                {
                    "state": result.state,
                    "error": str(result.info),
                }
            ),
            500,
        )

    return jsonify(
        {
            "state": result.state,
            "result": result.result,
        }
    )
