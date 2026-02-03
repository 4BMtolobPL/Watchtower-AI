import os

import redis

redis_client = redis.from_url(os.environ.get("CELERY_RESULT_BACKEND"))
