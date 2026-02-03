from sqlalchemy.orm import Mapped, mapped_column

from src.main import db
from src.utils.task_enum import TaskStatus


class UserStream(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column()
    stream_url: Mapped[str] = mapped_column()
    stream_key: Mapped[str] = mapped_column()
    status: Mapped[TaskStatus] = mapped_column(default=TaskStatus.PENDING)
