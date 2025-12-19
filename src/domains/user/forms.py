from flask_wtf import FlaskForm
from wtforms.fields.simple import PasswordField, SubmitField, StringField
from wtforms.validators import DataRequired, Email


class SignUpForm(FlaskForm):
    # username = StringField('Username', validators=[DataRequired("Username is required"), Length(min=2, max=20)])
    email = StringField(
        "Email",
        validators=[DataRequired("Email is required"), Email("Email is invalid")],
    )
    password = PasswordField(
        "Password", validators=[DataRequired("Password is required")]
    )
    submit = SubmitField("Sign Up")
