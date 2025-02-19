from typing import Any


class _Timed_Data:
    def __init__(self):
        self.data = {}

    def __getitem__(self, ts: int):
        return self.data.get(ts, None)

    def __setitem__(self, ts: int, item: Any):
        self.data[ts] = item
