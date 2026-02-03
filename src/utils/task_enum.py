from enum import Enum


class TaskStatus(Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    FAILURE = "FAILURE"
    SUCCESS = "SUCCESS"
