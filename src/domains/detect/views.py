import uuid
from pathlib import Path

from flask import (
    Blueprint,
    render_template,
    current_app,
    send_from_directory,
    jsonify,
)
from ultralytics import YOLO

from src.domains.detect.forms import UploadImageForm

detect_views = Blueprint(
    "detect", __name__, template_folder="templates", static_folder="static"
)


@detect_views.get("/images/<path:filename>")
def images(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@detect_views.route("/images", methods=["GET", "POST"])
def detect_images():
    form = UploadImageForm()
    if form.validate_on_submit():

        file = form.image.data
        ext = Path(file.filename).suffix
        # 아래처럼 werkzeug의 secure_filename을 사용해도 될듯
        # image_secure_filename = secure_filename(file.filename)
        image_uuid_file_name = str(uuid.uuid4())
        image_path = Path(
            current_app.config["UPLOAD_FOLDER"], image_uuid_file_name + ext
        )
        file.save(image_path)

        model = YOLO(
            Path(current_app.config["MODELS_FOLDER"], "fire_detect_v251205_1.pt"),
            verbose=False,
        )
        results = model(image_path, conf=0.5, verbose=False)

        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     probs = result.probs  # Probs object for classification outputs
        #     obb = result.obb  # Oriented boxes object for OBB outputs
        #     result.show()  # display to screen
        detect_file_name = f"detect_{image_uuid_file_name}"
        detect_image_path = Path(
            current_app.config["UPLOAD_FOLDER"], detect_file_name + ext
        )
        results[0].save(filename=detect_image_path)  # save to disk

        return jsonify({"filename": detect_file_name + ext})
    return render_template("detect/detect_images.html", form=form)
