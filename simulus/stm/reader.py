from mpi4py import MPI

from .messaging import _Message_Reader_Consume, STM_Tag
from .data import _Timed_Data

COMM = MPI.COMM_WORLD


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
