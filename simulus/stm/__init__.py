from collections.abc import Callable
import threading
from mpi4py import MPI
from typing import Any, Literal

from .data import _Timed_Data

from .messaging import (
    _Message_Channel_Connection,
    _Message_Channel_Creation,
    _Message_Channel_Put,
    _Message_Connection_Init_Data,
    _Message_Reader_Data,
    STM_Tag,
)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class STM:
    def __init__(self, listening_mode: Literal["thread", "manual"] = "thread"):
        self._channel_ranks: dict[str, int] = {}
        self._local_channels: dict[str, _Channel] = {}
        self._remote_readers: dict[str, list[Reader]] = {}
        self._setup_listening(listening_mode)

    def _setup_listening(self, listening_mode: Literal["thread", "manual"]):
        if listening_mode == "thread":
            self._listening_thread = threading.Thread(
                target=self._receive_process_loop,
                args=(STM_Tag.STM_DATA, self.handle_incoming),
            )
            self._listening_thread.start()
        elif listening_mode == "manual":
            pass
        else:
            raise ValueError("Invalid listening_mode")

    @staticmethod
    def _receive_process_loop(tag: int, callback: Callable[[Any], None]):
        while True:
            req = comm.irecv(tag=tag)
            data = req.wait()
            callback(data)

    def listen_manual(self):
        return comm.irecv(tag=STM_Tag.STM_DATA)

    def handle_incoming(self, msg):
        print(f"rank={rank} msg={msg}")
        if isinstance(msg, _Message_Channel_Creation):
            self._channel_ranks[msg.channel_name] = msg.source_rank
        elif isinstance(msg, _Message_Channel_Connection):
            channel = self._local_channels[msg.channel_name]
            channel.ranks_attached.add(msg.source_rank)
            channel.send_all_data(msg.source_rank)
        elif isinstance(msg, _Message_Channel_Put):
            channel = self._local_channels[msg.channel_name]
            self._put(msg.ts, msg.item, msg.channel_name)
        elif isinstance(msg, _Message_Reader_Data):
            for reader in self._remote_readers[msg.channel_name]:
                reader.data[msg.ts] = msg.item

    def create_channel(self, channel_name: str):
        if channel_name in self._channel_ranks:
            raise ValueError("Channel already exists!")
        self._local_channels[channel_name] = _Channel(channel_name)
        self._channel_ranks[channel_name] = rank

    def attach_reader(self, channel_name: str):
        channel_rank = self._channel_ranks[channel_name]
        reader = Reader(channel_name, channel_rank)
        if channel_name in self._local_channels:
            channel = self._local_channels[channel_name]
            channel.locals_attached.add(reader)
            reader.data = channel.channel_data
        else:
            self._remote_readers.setdefault(channel_name, [])
            self._remote_readers[channel_name].append(reader)
            msg = _Message_Channel_Connection(rank, channel_name)
            comm.send(obj=msg, dest=channel_rank, tag=STM_Tag.STM_DATA)
            reader.data = comm.recv(tag=STM_Tag.CONNECTION_INIT_DATA).channel_data
        return reader

    def attach_writer(self, channel_name: str):
        return Writer(channel_name)

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
        self.locals_attached: set[Reader] = set()
        creation_msg = _Message_Channel_Creation(rank, name)
        self.notify_ranks(creation_msg)

    def notify_ranks(self, msg: _Message_Channel_Creation):
        reqs: list[MPI.Request] = []
        for i in range(size):
            if i == rank:
                continue
            req = comm.isend(obj=msg, dest=i, tag=STM_Tag.STM_DATA)
            reqs.append(req)
        MPI.Request.waitall(reqs)

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
        MPI.Request.waitall(reqs)


class Reader:
    def __init__(self, channel_name: str, channel_rank: int):
        self.channel_name = channel_name
        self.channel_rank = channel_rank
        self.data: _Timed_Data = None

    def get(self, ts: int):
        return self.data[ts]


class Writer:
    def __init__(self, channel_name: str):
        self.channel_name = channel_name

    def put(self, ts: int, item: Any, stm: STM):
        stm._put(ts, item, self.channel_name)
