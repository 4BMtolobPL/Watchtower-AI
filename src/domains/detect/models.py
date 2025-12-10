import datetime
from typing import List

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.main import db


class UserImage(db.Model):
    # __tablename__ = "user_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    image_path: Mapped[str] = mapped_column()
    # is_detected: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC),
        onupdate=datetime.datetime.now(datetime.UTC),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    user: Mapped["User"] = relationship(
        back_populates="images"
    )  # noqa: F821  # ty:ignore[unresolved-reference]


class UserVideo(db.Model):
    # __tablename__ = "user_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    video_path: Mapped[str] = mapped_column()
    thumbnail_path: Mapped[str] = mapped_column(nullable=True)
    # is_detected: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC)
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        default=datetime.datetime.now(datetime.UTC),
        onupdate=datetime.datetime.now(datetime.UTC),
    )

    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    user: Mapped["User"] = relationship(
        back_populates="videos"
    )  # noqa: F821  # ty:ignore[unresolved-reference]

    detection_videos: Mapped[List["DetectionVideo"]] = relationship(
        back_populates="user_video"
    )


class DetectionVideo(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    model: Mapped[str] = mapped_column()
    video_path: Mapped[str] = mapped_column()

    user_video_id: Mapped[int] = mapped_column(ForeignKey("user_video.id"))
    user_video: Mapped["UserVideo"] = relationship(back_populates="detection_videos")
