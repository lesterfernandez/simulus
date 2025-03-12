from dataclasses import dataclass
from typing import Any

from .data import _Timed_Data


class STM_Tag:
    STM_DATA = 1
    CONNECTION_INIT_DATA = 2


@dataclass
class _Message_STM_Channels_Init:
    channels: list[str]
    source_rank: int


@dataclass
class _Message_STM_Shutdown:
    source_rank: int


@dataclass
class _Message_Channel_Put:
    ts: int
    item: Any
    source_rank: int
    channel_name: str


@dataclass
class _Message_Reader_Data:
    ts: int
    item: Any
    channel_name: str
