import uuid
from typing import Dict

from flask import Blueprint, jsonify, current_app, abort, request, render_template

stream_views = Blueprint("stream", __name__, template_folder="templates", static_folder="static")


@stream_views.route("/")
def index():
    return render_template("stream/index.html")


@stream_views.post("/create")
def create():
    request_json: Dict[str, str] = request.get_json()
    filename = request_json.get("filename")

    if filename is None:
        current_app.logger.warning("No filename provided")
        abort(400)

    stream_id = str(uuid.uuid4())

    yolo_worker.delay(filename, stream_id)

    return jsonify({"hls": f"/hls/{stream_id}/index.m3u8"})
