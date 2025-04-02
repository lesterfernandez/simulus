from collections.abc import Callable
import threading
from mpi4py import MPI
from typing import Any, Literal


from .log import logger
from .data import _Timed_Data
from .messaging import (
    _Message_Channel_Put,
    _Message_Reader_Consume,
    _Message_Reader_Data,
    _Message_STM_Channels_Init,
    _Message_STM_Shutdown,
    _Message_Writer_Advance,
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
        self._channel_writer_names: dict[str, list[str]] = {}

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
            reader = _Reader(reader_name, channel_name, RANK)
            channel = self._obj._local_channels[channel_name]
            channel.local_readers.add(reader)
            channel.set_reader_keeptime(reader_name, 0)
            self._obj._local_readers[reader_name] = reader
        else:
            # we don't know the rank at this point, so save for later
            self._channel_reader_names.setdefault(channel_name, [])
            self._channel_reader_names[channel_name].append(reader_name)
        return self

    def create_writer(self, channel_name: str, writer_name: str):
        channel_is_local = (
            channel_name in self._obj._local_channels
            and channel_name in self._obj._channel_ranks
        )
        if channel_is_local:
            channel = self._obj._local_channels[channel_name]
            channel.set_writer_advancetime(writer_name, 0)
            writer = _Writer(self._obj, writer_name, channel_name, RANK)
            self._obj._local_writers[writer_name] = writer
        else:
            # we don't know the rank at this point, so save for later
            self._channel_writer_names.setdefault(channel_name, [])
            self._channel_writer_names[channel_name].append(writer_name)
        return self

    def build(self):
        if not self._obj:
            raise Exception("Builder cannot be reused")

        # share all channel locations (source ranks)
        channel_msgs = _Message_STM_Channels_Init(
            channels=list(self._obj._local_channels.keys()), source_rank=RANK
        )
        rank_ready_messages = COMM.allgather(channel_msgs)
        logger.debug(f"({RANK}) ready msgs = {rank_ready_messages}")
        for msg in rank_ready_messages:
            for channel_name in msg.channels:
                self._obj._channel_ranks[channel_name] = msg.source_rank

        # initialize readers that are attached to remote channels
        # (this is a bit more involved since channels push data to readers)
        # each rank has a list of attached readers, which are each represented as a tuple
        # we will be distributing attachment data with alltoall,
        #   each rank is assigned a list of reader tuples.
        #   we declare a reader by putting its metadata in the list for the rank we want to send it to
        #   after alltoall, each channels will know the rank where each reader is located
        reader_rank_attachments: list[list[tuple[str, str]]] = [[] for _ in range(SIZE)]
        for channel_name, reader_names in self._channel_reader_names.items():
            channel_rank = self._obj._channel_ranks[channel_name]
            # create reader objects
            for reader_name in reader_names:
                reader = _Reader(reader_name, channel_name, channel_rank)
                self._obj._local_readers[reader_name] = reader
                self._obj._channel_readers.setdefault(channel_name, [])
                self._obj._channel_readers[channel_name].append(reader)
                # note the ranks that this reader has attachments to
                reader_rank_attachments[channel_rank].append(
                    (channel_name, reader_name)
                )
        # distribute reader attachment information
        reader_connection_msgs = COMM.alltoall(reader_rank_attachments)
        logger.debug(f"({RANK}) reader connection msgs = {reader_connection_msgs}")
        for source_rank, connections in enumerate(reader_connection_msgs):
            for channel_name, reader_name in connections:
                channel = self._obj._local_channels[channel_name]
                channel.reader_ranks.add(source_rank)
                channel._readers_keeptime[reader_name] = 0

        # a similar setup for writers, but for a different reason
        # each channel needs to know the advance time for each of its writers
        writer_rank_attachments: list[list[tuple[str, str]]] = [[] for _ in range(SIZE)]
        for channel_name, writer_names in self._channel_writer_names.items():
            channel_rank = self._obj._channel_ranks[channel_name]
            for writer_name in writer_names:
                writer = _Writer(self._obj, writer_name, channel_name, channel_rank)
                self._obj._local_writers[writer_name] = writer
                writer_rank_attachments[channel_rank].append(
                    (channel_name, writer_name)
                )
        # distribute writer attachment information
        writer_connection_msgs = COMM.alltoall(writer_rank_attachments)
        logger.debug(f"({RANK}) writer connection msgs = {writer_connection_msgs}")
        for source_rank, connections in enumerate(writer_connection_msgs):
            for channel_name, writer_name in connections:
                channel = self._obj._local_channels[channel_name]
                channel._writers_advancetime[writer_name] = 0

        logger.info(
            f"({RANK}) finished build with channel keeptimes = {
                [(key, channel._readers_keeptime[key]) 
                 for channel in self._obj._local_channels.values() 
                 for key in channel._readers_keeptime]
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
        self._local_readers: dict[str, _Reader] = {}
        self._local_writers: dict[str, _Writer] = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_reader(self, name: str):
        return self._local_readers[name]

    def get_writer(self, name: str):
        return self._local_writers[name]

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
            for reader in self._channel_readers.get(msg.channel_name, []):
                reader.data[msg.ts] = msg.item
        elif isinstance(msg, _Message_Reader_Consume):
            channel = self._local_channels[msg.channel_name]
            channel.handle_consume_until(msg.reader_name, msg.until)
        elif isinstance(msg, _Message_Writer_Advance):
            if msg.channel_name in self._local_channels:
                channel = self._local_channels[msg.channel_name]
                channel.handle_advance_until(msg.writer_name, msg.until)
                return
            for reader in self._channel_readers[msg.channel_name]:
                reader.channel_advancetime = max(reader.channel_advancetime, msg.until)

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
        self.reader_ranks: set[int] = set()
        self.local_readers: set[_Reader] = set()
        self._readers_keeptime = _PQDict_()
        self._writers_advancetime = _PQDict_()

    def publish_data(self, ts: int, item: Any):
        self.channel_data[ts] = item
        for reader in self.local_readers:
            reader.data[ts] = item
        reqs: list[MPI.Request] = []
        msg = _Message_Reader_Data(ts, item, self.name)
        for rank_attached in self.reader_ranks:
            req = COMM.isend(obj=msg, dest=rank_attached, tag=STM_Tag.STM_DATA)
            reqs.append(req)
        logger.debug(f"({RANK}) publishing item={item} ts={ts} to {len(reqs)} ranks")
        MPI.Request.waitall(reqs)

    def keeptime(self) -> int:
        _, ts = self._readers_keeptime.peek()
        return ts

    def set_reader_keeptime(self, reader_name: str, ts: int):
        self._readers_keeptime[reader_name] = ts

    def handle_consume_until(self, reader_name: str, ts: int):
        prev_chan_keeptime = self.keeptime()
        self.set_reader_keeptime(reader_name, ts)
        new_chan_keeptime = self.keeptime()
        logger.info(
            f"({RANK}) {self.name} consume until {ts}, keeptime={new_chan_keeptime}"
        )
        for ts in range(prev_chan_keeptime, new_chan_keeptime):
            logger.debug(f"({RANK}) {self.name} deleting item at {ts}")
            del self.channel_data[ts]

    def advancetime(self) -> int:
        _, ts = self._writers_advancetime.peek()
        return ts

    def set_writer_advancetime(self, writer_name: str, ts: int):
        self._writers_advancetime[writer_name] = ts

    def handle_advance_until(self, writer: str, ts: int):
        prev_chan_advancetime = self.advancetime()
        self.set_writer_advancetime(writer, ts)
        new_chan_advancetime = self.advancetime()
        if new_chan_advancetime <= prev_chan_advancetime:
            return
        for reader in self.local_readers:
            reader.channel_advancetime = ts
        reqs: list[MPI.Request] = []
        msg = _Message_Writer_Advance(
            until=ts, writer_name=writer, channel_name=self.name
        )
        for rank_attached in self.reader_ranks:
            req = COMM.isend(obj=msg, dest=rank_attached, tag=STM_Tag.STM_DATA)
            reqs.append(req)
        logger.debug(
            f"({RANK}) publishing writer advancetime={new_chan_advancetime} to {len(reqs)} ranks"
        )


class _Reader:
    def __init__(self, name: str, channel_name: str, channel_rank: int):
        self.name = name
        self.data = _Timed_Data()
        self.keeptime = 0
        self.channel_name = channel_name
        self.channel_rank = channel_rank
        self.channel_advancetime = 0

    def get(self, ts: int):
        if ts <= self.keeptime or ts < self.channel_advancetime:
            return None, False
        item = self.data[ts]
        return item, True
        # if not item and not wait:
        # return item
        # todo: block until item becomes available OR channel_advancetime reaches ts
        # return item

    def consume_until(self, time: int):
        if time >= self.keeptime:
            for ts in range(self.keeptime, time + 1):
                del self.data[ts]
            self.keeptime = time
            msg = _Message_Reader_Consume(time, self.name, self.channel_name)
            COMM.isend(obj=msg, dest=self.channel_rank, tag=STM_Tag.STM_DATA)


class _Writer:
    def __init__(self, stm: _STM, name: str, channel_name: str, channel_rank: int):
        self.stm = stm
        self.name = name
        self.channel_name = channel_name
        self.channel_rank = channel_rank
        self.advancetime = 0

    def put(self, ts: int, item: Any):
        self.stm._put(ts, item, self.channel_name)

    def advance_until(self, ts: int):
        if ts > self.advancetime:
            self.advancetime = ts
            msg = _Message_Writer_Advance(ts, self.name, self.channel_name)
            COMM.isend(obj=msg, dest=self.channel_rank, tag=STM_Tag.STM_DATA)
