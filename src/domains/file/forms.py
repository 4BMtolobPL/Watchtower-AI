from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.fields.simple import SubmitField


class UploadImageForm(FlaskForm):
    image = FileField(
        validators=[
            FileRequired("Image file is required"),
            FileAllowed(
                ["jpg", "png", "jpeg", "gif", "webp", "bmp"],
                "지원하지 않는 파일입니다.",
            ),
        ]
    )
    submit = SubmitField("Upload")


class UploadVideoForm(FlaskForm):
    video = FileField(
        validators=[
            FileRequired("Video file is required"),
            FileAllowed(["mp4", "avi", "mkv"], "지원하지 않는 파일입니다."),
        ]
    )
    submit = SubmitField("Upload")
