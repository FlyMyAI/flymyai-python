import enum


class EventType(str, enum.Enum):
    CANCELLING = "stream_cancelling"
    STREAM_ID = "id"
