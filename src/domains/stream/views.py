import uuid
from pathlib import Path

from flask import Blueprint, jsonify, current_app, abort, render_template, send_from_directory, redirect, url_for

from src.domains.stream.forms import UserStreamForm
from src.domains.stream.models import UserStream
from src.main import db
from src.utils.task_manager import create_detect_task

stream_views = Blueprint("stream", __name__, template_folder="templates", static_folder="static")


@stream_views.route("/")
def index():
    user_streams = UserStream.query.order_by(UserStream.id.desc()).all()

    return render_template("stream/index.html", user_streams=user_streams)


@stream_views.route("/create", methods=["GET", "POST"])
def create():
    form = UserStreamForm()

    if form.validate_on_submit():
        stream_url = form.stream_url.data
        selected_model = form.model.data

        stream_key = str(uuid.uuid4())
        user_stream = UserStream(model=selected_model, stream_url=stream_url, stream_key=stream_key)
        db.session.add(user_stream)
        db.session.commit()

        save_dir = Path(current_app.config["STREAM_FOLDER"]) / stream_key
        save_dir.mkdir(parents=True, exist_ok=True)

        create_detect_task.delay(stream_url, str(save_dir / "index.m3u8"), selected_model)

        return redirect(url_for("stream.index"))  # return jsonify({"hls": f"/hls/{stream_key}/index.m3u8"})

    return render_template("stream/create.html", form=form)


@stream_views.get("/detail/<int:stream_id>")
def detail(stream_id: int):
    user_stream: UserStream = UserStream.query.get_or_404(stream_id)
    return render_template("stream/detail.html", user_stream=user_stream)


@stream_views.get("/key/<stream_id>")
def key(stream_id: int):
    user_stream: UserStream = UserStream.query.get_or_404(stream_id)
    return jsonify({"stream_id": user_stream.id, "stream_key": user_stream.stream_key})


@stream_views.get("/hls/<string:stream_key>/<path:filename>")
def hls(stream_key: str, filename: str):
    stream_dir = Path(current_app.config["STREAM_FOLDER"]) / stream_key

    if not stream_dir.exists():
        abort(404)

    file_path = stream_dir / filename
    if not file_path.exists():
        abort(404)

    # MIME 타입 분기
    if filename.endswith(".m3u8"):
        mimetype = "application/vnd.apple.mpegurl"
    elif filename.endswith(".ts"):
        mimetype = "video/mp2t"
    else:
        abort(404)

    response = send_from_directory(directory=stream_dir, path=filename, mimetype=mimetype, conditional=True,
                                   # Range 요청 처리
                                   )

    # 라이브 스트림용 캐시 방지
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response
