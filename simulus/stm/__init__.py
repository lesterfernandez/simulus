from collections.abc import Callable
import threading
from mpi4py import MPI
from typing import Any, Literal


from .log import logger
from .data import _Timed_Data
from .messaging import (
    _Message_Channel_Put,
    _Message_Reader_Data,
    _Message_STM_Channels_Init,
    _Message_STM_Shutdown,
    STM_Tag,
)

from simulus.utils import _PQDict_


COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


class STMBuilder:
    def __init__(self):
        self._obj = _STM()
        self._channel_reader_names: dict[str, list[str]] = {}

    def create_channels(self, channels: list[str]):
        for channel in channels:
            self._obj._local_channels[channel] = _Channel(channel)
            self._obj._channel_ranks[channel] = RANK
        return self

    def create_reader(self, channel_name: str, reader_name: str):
        channel_is_local = (
            channel_name in self._obj._local_channels
            and channel_name in self._obj._channel_ranks
        )
        if channel_is_local:
            channel_rank = self._obj._channel_ranks[channel_name]
            reader = _Reader(reader_name, channel_name, channel_rank)
            channel = self._obj._local_channels[channel_name]
            reader.data = channel.channel_data
            channel.locals_attached.add(reader)
            self._obj.readers[reader_name] = reader
        else:
            self._channel_reader_names.setdefault(channel_name, [])
            self._channel_reader_names[channel_name].append(reader_name)
        return self

    def create_writer(self, channel: str, name: str):
        pass

    def _create_connection(
        self, channel: str, name: str, type: Literal["reader", "writer"]
    ):
        pass

    def build(self):
        if not self._obj:
            raise Exception("Builder cannot be reused")

        # distribute all channel locations (source ranks)
        channel_msgs = _Message_STM_Channels_Init(
            channels=list(self._obj._local_channels.keys()), source_rank=RANK
        )
        logger.debug(f"({RANK}) gathering ready msgs")
        rank_ready_messages = COMM.allgather(channel_msgs)
        logger.debug(f"({RANK}) ready msgs = {rank_ready_messages}")
        for msg in rank_ready_messages:
            for channel_name in msg.channels:
                self._obj._channel_ranks[channel_name] = msg.source_rank

        # initialize readers that are attached to remote channels
        # each rank has a list of connections, which are represented as tuples
        rank_connections: list[list[tuple[str, str]]] = [[] for _ in range(SIZE)]
        for channel_name, reader_names in self._channel_reader_names.items():
            channel_rank = self._obj._channel_ranks[channel_name]
            # create reader objects
            for reader_name in reader_names:
                reader = _Reader(
                    reader_name, channel_name, self._obj._channel_ranks[channel_name]
                )
                self._obj.readers[reader_name] = reader
                self._obj._channel_readers.setdefault(channel_name, [])
                self._obj._channel_readers[channel_name].append(reader)
                # note the ranks that this rank has attachments to
                rank_connections[channel_rank].append((channel_name, reader_name))

        # distribute channel attachment information
        rank_connection_msgs = COMM.alltoall(rank_connections)
        logger.debug(f"({RANK}) rank connection msgs = {rank_connection_msgs}")
        for source_rank, connections in enumerate(rank_connection_msgs):
            if connections == []:
                continue
            for channel_name, reader_name in connections:
                channel = self._obj._local_channels[channel_name]
                channel.ranks_attached.add(source_rank)
                channel.reader_keeptime[reader_name] = 0

        logger.info(
            f"({RANK}) finished build with channel keeptimes = {
                [(key, channel.reader_keeptime[key]) 
                 for channel in self._obj._local_channels.values() 
                 for key in channel.reader_keeptime]
            }"
        )

        obj = self._obj
        self._obj = None
        return obj


class _STM:
    def __init__(self):
        self._channel_ranks: dict[str, int] = {}
        self._local_channels: dict[str, _Channel] = {}
        self._channel_readers: dict[str, list[_Reader]] = {}
        self._rank_shutdown = [False] * SIZE
        self.readers: dict[str, _Reader] = {}
        self.writers: dict[str, _Writer] = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, listening_mode: Literal["thread", "manual"] = "thread"):
        if listening_mode == "thread":
            self._listening_thread = threading.Thread(
                target=self._receive_message_loop,
                args=(self.process_message,),
            )
            self._listening_thread.start()
        elif listening_mode == "manual":
            pass
        else:
            raise ValueError("Invalid listening_mode")

    def stop(self):
        shutdown_msg = _Message_STM_Shutdown(source_rank=RANK)
        for target in range(SIZE):
            COMM.send(obj=shutdown_msg, dest=target, tag=STM_Tag.STM_DATA)

    def _receive_message_loop(self, handler: Callable[[Any], None]):
        while True:
            msg = self.receive_message().wait()
            if self.check_shutdown(msg):
                break
            handler(msg)

    def receive_message(self):
        return COMM.irecv(tag=STM_Tag.STM_DATA)

    def check_shutdown(self, msg: Any):
        if isinstance(msg, _Message_STM_Shutdown):
            self._rank_shutdown[msg.source_rank] = True
            logger.debug(
                f"({RANK}) received shutdown from {msg.source_rank}, {self._rank_shutdown}"
            )
        if self._rank_shutdown[RANK] and all(self._rank_shutdown):
            logger.info(f"({RANK}) shutting down")
            return True
        return False

    def process_message(self, msg):
        logger.info(f"({RANK}) received {msg}")
        if isinstance(msg, _Message_Channel_Put):
            self._put(msg.ts, msg.item, msg.channel_name)
        elif isinstance(msg, _Message_Reader_Data):
            for reader in self._channel_readers[msg.channel_name]:
                reader.data[msg.ts] = msg.item

    def attach_writer(self, channel_name: str):
        return _Writer(self, channel_name)

    def _put(self, ts: int, item: Any, channel_name: str):
        if channel_name in self._local_channels:
            channel = self._local_channels[channel_name]
            channel.publish_data(ts, item)
        else:
            if channel_name not in self._channel_ranks:
                raise ValueError(f"Unknown channel {channel_name}")
            msg = _Message_Channel_Put(ts, item, RANK, channel_name)
            channel_rank = self._channel_ranks[channel_name]
            COMM.send(obj=msg, dest=channel_rank, tag=STM_Tag.STM_DATA)


class _Channel:
    def __init__(self, name: str):
        self.name = name
        self.channel_data = _Timed_Data()
        self.ranks_attached: set[int] = set()
        self.reader_keeptime = _PQDict_()
        self.locals_attached: set[_Reader] = set()

    def publish_data(self, ts: int, item: Any):
        self.channel_data[ts] = item
        for reader in self.locals_attached:
            reader.data[ts] = item
        reqs: list[MPI.Request] = []
        msg = _Message_Reader_Data(ts, item, self.name)
        for rank_attached in self.ranks_attached:
            req = COMM.isend(obj=msg, dest=rank_attached, tag=STM_Tag.STM_DATA)
            reqs.append(req)
        logger.debug(f"({RANK}) publishing item={item} ts={ts} reqs={reqs}")
        MPI.Request.waitall(reqs)


class _Reader:
    def __init__(self, name: str, channel_name: str, channel_rank: int):
        self.name = name
        self.channel_name = channel_name
        self.channel_rank = channel_rank
        self.data = _Timed_Data()

    def get(self, ts: int):
        return self.data[ts]


class _Writer:
    def __init__(self, stm: _STM, channel_name: str):
        self.stm = stm
        self.channel_name = channel_name

    def put(self, ts: int, item: Any):
        self.stm._put(ts, item, self.channel_name)
