import os
from pathlib import Path

from celery import Celery, Task
from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_security import Security, SQLAlchemyUserDatastore
from flask_security.models import fsqla_v3
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect

from src.config.config import config

db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
security = Security()
login_manager = LoginManager()
login_manager.login_view = "security.login"
login_manager.login_message = ""


def create_app(env: str = "development") -> Flask:
    app = Flask(__name__)

    if env in config:
        app.config.from_object(config[env])
    else:
        raise KeyError(f"Environment {env} not found.")

    db.init_app(app)
    migrate.init_app(app, db)
    fsqla_v3.FsModels.set_db_info(db)

    celery_init_app(app)

    from src.domains.auth.models import Role
    from src.domains.user.models import User

    user_datastore = SQLAlchemyUserDatastore(db, User, Role)
    security.init_app(app, user_datastore)
    csrf.init_app(app)
    login_manager.init_app(app)

    # from src.domains.user.views import user_views
    # from src.domains.auth.views import auth_views
    from src.domains.detect.views import detect_views
    from src.domains.root.views import root_views

    app.register_blueprint(root_views, url_prefix="/")
    # app.register_blueprint(user_views, url_prefix="/user")
    # app.register_blueprint(auth_views, url_prefix="/auth")
    app.register_blueprint(detect_views, url_prefix="/detect")

    init_folder(Path(app.config["UPLOAD_FOLDER"]))
    init_folder(Path(app.config["UPLOAD_FOLDER"]) / "images")
    init_folder(Path(app.config["UPLOAD_FOLDER"]) / "videos")
    init_folder(Path(app.config["MODELS_FOLDER"]))

    return app


def init_folder(path: Path):
    if not os.path.isdir(path):
        os.makedirs(path)


def celery_init_app(app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app
