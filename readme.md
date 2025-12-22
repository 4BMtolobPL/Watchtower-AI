Watchtower-AI
=============

## Preparation

### Install uv

https://docs.astral.sh/uv/getting-started/installation/

## Install dependency

```bash
uv sync
```

## Run

### Database

```bash
uv run flask db upgrade
```

### Run flask, celery

```bash
uv run flask run
uv run celery -A src.make_celery worker --loglevel INFO -P gevent
```

## Environment

### Required

```dotenv
SECRET_KEY=
WTF_CSRF_SECRET_KEY=
SECURITY_PASSWORD_SALT=
CELERY_BROKER_URL=
CELERY_RESULT_BACKEND=
```

You can use the code below to generate the secret key and salt.

```python
import secrets

print(secrets.token_hex())
print(secrets.SystemRandom().getrandbits(128))
```

### Optional

```dotenv
FLASK_DEBUG=True
UPLOAD_FOLDER=
MODELS_FOLDER=
```

