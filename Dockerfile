# uv image
FROM ghcr.io/astral-sh/uv:python3.14-trixie AS builder

# Change the working directory to the `/app` directory
WORKDIR /app

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_NO_DEV=1
ENV UV_COMPILE_BYTECODE=1

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --group prod --locked --no-install-project --no-editable

# Copy the project into the intermediate image
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --group prod --locked --no-editable

FROM python:3.14-slim

WORKDIR /app

# 런타임에 필요한 최소 라이브러리
RUN apt update && apt install -y sqlite3 libsqlite3-dev libgl1 libglib2.0-0 ffmpeg && apt clean && rm -rf /root/.cache/pip && rm -rf /var/lib/apt/lists/*

# RUN useradd -m app
# USER app
# COPY --from=builder --chown=app:app /app /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.flaskenv /app/.flaskenv
COPY --from=builder /app/migrations /app/migrations
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python"]
CMD ["-m", "gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "src.main:create_app('production')"]