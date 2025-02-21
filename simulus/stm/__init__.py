from collections.abc import Callable
import threading
from mpi4py import MPI
from typing import Any, Literal
import numpy as np

from .log import logger
from .data import _Timed_Data
from .messaging import (
    _Message_Channel_Connection,
    _Message_Channel_Put,
    _Message_Connection_Init_Data,
    _Message_Reader_Data,
    _Message_STM_Init_Data,
    _Message_STM_Shutdown,
    STM_Tag,
)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class STMBuilder:
    def __init__(self):
        self._obj = _STM()

    def create_channels(self, channels: list[str]):
        for channel in channels:
            self._obj._local_channels[channel] = _Channel(channel)
            self._obj._channel_ranks[channel] = rank
        return self

    def build(self):
        if not self._obj:
            raise Exception("Builder cannot be reused")
        obj = self._obj
        self._obj = None
        return obj


class _STM:
    def __init__(self):
        self._channel_ranks: dict[str, int] = {}
        self._local_channels: dict[str, _Channel] = {}
        self._channel_readers: dict[str, list[_Reader]] = {}
        self._rank_shutdown = [False] * size

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, listening_mode: Literal["thread", "manual"] = "thread"):
        ready_message = _Message_STM_Init_Data(
            channels=list(self._local_channels.keys()), source_rank=rank
        )
        logger.debug(f"({rank}) gathering ready msgs")
        rank_ready_messages = comm.allgather(ready_message)
        logger.debug(f"({rank}) ready msgs = {rank_ready_messages}")
        for msg in rank_ready_messages:
            for chan in msg.channels:
                self._channel_ranks[chan] = msg.source_rank

        if listening_mode == "thread":
            self._listening_thread = threading.Thread(
                target=self._receive_message_loop,
                args=(STM_Tag.STM_DATA, self.handle_incoming),
            )
            self._listening_thread.start()
        elif listening_mode == "manual":
            pass
        else:
            raise ValueError("Invalid listening_mode")

    def stop(self):
        shutdown_msg = _Message_STM_Shutdown(source_rank=rank)
        for target in range(size):
            comm.send(obj=shutdown_msg, dest=target, tag=STM_Tag.STM_DATA)

    def _receive_message_loop(self, tag: int, process_msg: Callable[[Any], None]):
        while True:
            req = comm.irecv(tag=tag)
            msg = req.wait()
            if isinstance(msg, _Message_STM_Shutdown):
                self._rank_shutdown[msg.source_rank] = True
                logger.debug(
                    f"({rank}) received shutdown from {msg.source_rank}, {self._rank_shutdown}"
                )
            if self._rank_shutdown[rank] and all(self._rank_shutdown):
                logger.info(f"({rank}) shutting down")
                break
            process_msg(msg)

    def listen_manual(self):
        return comm.irecv(tag=STM_Tag.STM_DATA)

    def handle_incoming(self, msg):
        logger.info(f"({rank}) received {msg}")
        if isinstance(msg, _Message_Channel_Connection):
            channel = self._local_channels[msg.channel_name]
            channel.ranks_attached.add(msg.source_rank)
            channel.send_all_data(msg.source_rank)
        elif isinstance(msg, _Message_Channel_Put):
            channel = self._local_channels[msg.channel_name]
            self._put(msg.ts, msg.item, msg.channel_name)
        elif isinstance(msg, _Message_Reader_Data):
            for reader in self._channel_readers[msg.channel_name]:
                reader.data[msg.ts] = msg.item

    def attach_reader(self, channel_name: str):
        channel_rank = self._channel_ranks[channel_name]
        logger.debug(f"({rank}) creating reader for {channel_name} at {channel_rank}")
        reader = _Reader(channel_name, channel_rank)
        if channel_name in self._local_channels:
            channel = self._local_channels[channel_name]
            reader.data = channel.channel_data
            channel.locals_attached.add(reader)
        else:
            msg = _Message_Channel_Connection(rank, channel_name)
            comm.send(obj=msg, dest=channel_rank, tag=STM_Tag.STM_DATA)
            reader.data = comm.recv(tag=STM_Tag.CONNECTION_INIT_DATA).channel_data
            self._channel_readers.setdefault(channel_name, [])
            self._channel_readers[channel_name].append(reader)
        logger.debug(
            f"({rank}) reader finishied attachment to {channel_name} at {channel_rank}"
        )
        return reader

    def attach_writer(self, channel_name: str):
        return _Writer(channel_name)

    def _put(self, ts: int, item: Any, channel_name: str):
        if channel_name in self._local_channels:
            channel = self._local_channels[channel_name]
            channel.publish_data(ts, item)
        else:
            if channel_name not in self._channel_ranks:
                raise ValueError(f"Unknown channel {channel_name}")
            msg = _Message_Channel_Put(ts, item, rank, channel_name)
            channel_rank = self._channel_ranks[channel_name]
            comm.send(obj=msg, dest=channel_rank, tag=STM_Tag.STM_DATA)


class _Channel:
    def __init__(self, name: str):
        self.name = name
        self.channel_data = _Timed_Data()
        self.ranks_attached: set[int] = set()
        self.locals_attached: set[_Reader] = set()

    def send_all_data(self, dest_rank: int):
        msg = _Message_Connection_Init_Data(self.channel_data)
        comm.isend(obj=msg, dest=dest_rank, tag=STM_Tag.CONNECTION_INIT_DATA)

    def publish_data(self, ts: int, item: Any):
        self.channel_data[ts] = item
        for reader in self.locals_attached:
            reader.data[ts] = item
        reqs: list[MPI.Request] = []
        msg = _Message_Reader_Data(ts, item, self.name)
        for rank_attached in self.ranks_attached:
            req = comm.isend(obj=msg, dest=rank_attached, tag=STM_Tag.STM_DATA)
            reqs.append(req)
        logger.debug(f"({rank}) publishing item={item} ts={ts} reqs={reqs}")
        MPI.Request.waitall(reqs)


class _Reader:
    def __init__(self, channel_name: str, channel_rank: int):
        self.channel_name = channel_name
        self.channel_rank = channel_rank
        self.data: _Timed_Data = None

    def get(self, ts: int):
        return self.data[ts]


class _Writer:
    def __init__(self, channel_name: str):
        self.channel_name = channel_name

    def put(self, ts: int, item: Any, stm: _STM):
        stm._put(ts, item, self.channel_name)
