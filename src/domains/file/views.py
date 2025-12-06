import uuid
from pathlib import Path

from flask import (
    Blueprint,
    redirect,
    url_for,
    render_template,
    abort,
    current_app,
    send_from_directory,
)
from flask_login import login_required, current_user

from src.domains.file.forms import UploadImageForm
from src.domains.file.models import UserImage
from src.main import db

file_views = Blueprint(
    "file", __name__, template_folder="templates", static_folder="static"
)


@file_views.route("/images/upload", methods=["GET", "POST"])
@login_required
def upload_image():
    form = UploadImageForm()
    if form.validate_on_submit():
        file = form.image.data

        ext = Path(file.filename).suffix
        # 아래처럼 werkzeug의 secure_filename을 사용해도 될듯
        # image_secure_filename = secure_filename(file.filename)
        image_uuid_file_name = str(uuid.uuid4()) + ext
        image_path = Path(current_app.config["UPLOAD_FOLDER"], image_uuid_file_name)
        file.save(image_path)

        user_image = UserImage(user_id=current_user.id, image_path=image_uuid_file_name)
        db.session.add(user_image)
        db.session.commit()

        return redirect(url_for("root.index"))
    return render_template("file/image_upload.html", form=form)


@file_views.route("/images/<path:filename>")
def image_files(filename):
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)
