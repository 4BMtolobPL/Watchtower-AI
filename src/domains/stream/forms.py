from flask_wtf import FlaskForm
from wtforms.fields.choices import SelectField
from wtforms.fields.simple import URLField, SubmitField
from wtforms.validators import DataRequired

from src.utils.detector import VideoDetectorEnum


class UserStreamForm(FlaskForm):
    stream_url = URLField(validators=[DataRequired("Stream URL is required")])
    model = SelectField("모델 선택", choices=[(de.value, de.name) for de in VideoDetectorEnum],
                        default=VideoDetectorEnum.YOLO11n.value, )
    submit = SubmitField("Submit")
