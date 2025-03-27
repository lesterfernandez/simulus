# FILE INFO ###################################################
# Author: Jason Liu <jasonxliu2010@gmail.com>
# Created on July 2, 2019
# Last Update: Time-stamp: <2019-09-07 09:18:15 liux>
###############################################################

import math, re
from collections.abc import MutableMapping

__all__ = [
    "QDIS",
    "WelfordStats",
    "TimeMarks",
    "DataSeries",
    "TimeSeries",
    "DataCollector",
]

import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class QDIS:
    """Queuing disciplines used by semaphores and resources."""

    FIFO = 0  # first in first out
    LIFO = 1  # last in first out
    SIRO = 2  # service in random order
    PRIORITY = 3  # priority based


class WelfordStats(object):
    """Welford's one-pass algorithm to get simple statistics (including
    the mean and variance) from a series of data."""

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._varsum = 0.0
        self._max = float("-inf")
        self._min = float("inf")

    def __len__(self):
        return self._n

    def push(self, x):
        """Add data to the series."""
        self._n += 1
        if x > self._max:
            self._max = x
        if x < self._min:
            self._min = x
        d = x - self._mean
        self._varsum += d * d * (self._n - 1) / self._n
        self._mean += d / self._n

    def min(self):
        return self._min

    def max(self):
        return self._max

    def mean(self):
        return self._mean

    def stdev(self):
        return math.sqrt(self._varsum / self._n)

    def var(self):
        return self._varsum / self._n


class TimeMarks(object):
    """A series of (increasing) time instances."""

    def __init__(self, keep_data=False):
        if keep_data:
            self._data = []
        else:
            self._data = None
        self._n = 0

    def __len__(self):
        """Return the number of collected samples."""
        return self._n

    def push(self, t):
        if self._n == 0:
            self._last = t
        elif t < self._last:
            errmsg = "timemarks.push(%g) earlier than last entry (%g)" % (t, self._last)
            log.error(errmsg)
            raise ValueError(errmsg)
        if self._data is not None:
            self._data.append(t)
        self._n += 1
        self._last = t

    def data(self):
        """Return all samples if keep_data has been set when the timemarks was
        initialized; otherwise, return None."""
        return self._data

    def rate(self, t=None):
        """Return the arrival rate, which is the averge number of sameples up
        to the given time. If t is ignored, it's up to the time of the
        last entry."""
        if self._n > 0:
            if t is None:
                t = self._last
            elif t < self._last:
                errmsg = "timemarks.rate(t=%g) earlier than last entry (%g)" % (
                    t,
                    self._last,
                )
                log.error(errmsg)
                raise ValueError(errmsg)
            return self._n / t
        else:
            return 0


class DataSeries(object):
    """A series of numbers."""

    def __init__(self, keep_data=False):
        if keep_data:
            self._data = []
        else:
            self._data = None
        self._rs = WelfordStats()

    def __len__(self):
        """Return the number of collected samples."""
        return len(self._rs)

    def push(self, d):
        if self._data is not None:
            self._data.append(d)
        self._rs.push(d)

    def data(self):
        """Return all samples if keep_data has been set when the dataseries
        has been initialized; otherwise, return None."""
        return self._data

    def mean(self):
        """Return the sample mean."""
        if len(self._rs) > 0:
            return self._rs.mean()
        else:
            return 0

    def stdev(self):
        """Return the sample standard deviation."""
        if len(self._rs) > 1:
            return self._rs.stdev()
        else:
            return float("inf")

    def var(self):
        """Return the sample variance."""
        if len(self._rs) > 1:
            return self._rs.var()
        else:
            return float("inf")

    def min(self):
        """Return the minimum of all samples."""
        if len(self._rs) > 0:
            return self._rs.min()
        else:
            return float("-inf")

    def max(self):
        """Return the maximum of all samples."""
        if len(self._rs) > 0:
            return self._rs.max()
        else:
            return float("inf")


class TimeSeries(object):
    """A series of time-value pairs."""

    def __init__(self, keep_data=False):
        if keep_data:
            self._data = []
        else:
            self._data = None
        self._rs = WelfordStats()
        self._area = 0

    def __len__(self):
        """Return the number of collected samples."""
        return len(self._rs)

    def push(self, d):
        t, v = d
        if len(self._rs) == 0:
            self._last_t = t
            self._last_v = v
        elif t < self._last_t:
            errmsg = "timeseries.push(%r) earlier than last entry (%g)" % (
                d,
                self._last_t,
            )
            log.error(errmsg)
            raise ValueError(errmsg)

        if self._data is not None:
            self._data.append(d)
        self._rs.push(v)
        self._area += (t - self._last_t) * self._last_v
        self._last_t = t
        self._last_v = v

    def data(self):
        """Return all samples if keep_data has been set when the timeseries
        was initialized; otherwise, return None."""
        return self._data

    def rate(self, t=None):
        """Return the arrival rate, which is the averge number of samples up
        to the given time. If t is ignored, it's up to the time of the
        last entry."""
        if len(self._rs) > 0:
            if t is None:
                t = self._last_t
            elif t < self._last_t:
                errmsg = "timeseries.rate(t=%g) earlier than last entry (%g)" % (
                    t,
                    self._last_t,
                )
                log.error(errmsg)
                raise ValueError(errmsg)
            return len(self._rs) / t
        else:
            return 0

    def mean(self):
        """Return the sample mean."""
        if len(self._rs) > 0:
            return self._rs.mean()
        else:
            return 0

    def stdev(self):
        """Return the sample standard deviation."""
        if len(self._rs) > 1:
            return self._rs.stdev()
        else:
            return float("inf")

    def var(self):
        """Return the sample variance."""
        if len(self._rs) > 1:
            return self._rs.var()
        else:
            return float("inf")

    def min(self):
        """Return the minimum of all samples."""
        if len(self._rs) > 0:
            return self._rs.min()
        else:
            return float("-inf")

    def max(self):
        """Return the maximum of all samples."""
        if len(self._rs) > 0:
            return self._rs.max()
        else:
            return float("inf")

    def avg_over_time(self, t=None):
        """Return the average value over time. If t is ignored, it's the
        average up to the time of the last entry."""
        if len(self._rs) > 0:
            if t is None:
                t = self._last_t
            if t < self._last_t:
                errmsg = (
                    "timeseries.avg_over_time(t=%g) earlier than last entry (%g)"
                    % (t, self._last_t)
                )
                log.error(errmsg)
                raise ValueError(errmsg)
            return (self._area + (t - self._last_t) * self._last_v) / t
        else:
            return 0


class DataCollector(object):
    """Statistics collection for resources, stores, buckets, and mailboxes."""

    def __init__(self, **kwargs):
        """Initialize the data collector. kwargs is the keyworded arguments;
        it's a dictionary containing all attributes allowed to be
        collected at the corresponding resource or facility."""
        log.info("creating data collector:")
        self._attrs = kwargs
        patterns = {
            re.compile(r"timemarks\s*(\(\s*(all)?\s*\))?"): TimeMarks,
            re.compile(r"dataseries\s*(\(\s*(all)?\s*\))?"): DataSeries,
            re.compile(r"timeseries\s*(\(\s*(all)?\s*\))?"): TimeSeries,
        }
        for k, v in self._attrs.items():
            if hasattr(self, k):
                errmsg = "datacollector attribute %s already exists" % k
                log.error(errmsg)
                raise ValueError(errmsg)
            for pat, cls in patterns.items():
                m = pat.match(v)
                if m is not None:
                    if m.group(2):
                        v = cls(True)
                        log.info("  %s: %s(keep_data=True)" % (k, cls.__name__))
                    else:
                        v = cls(False)
                        log.info("  %s: %s(keep_data=False)" % (k, cls.__name__))
                    setattr(self, k, v)
                    self._attrs[k] = v
                    break
            else:
                errmsg = "datacollector %r has unknown value (%r)" % (k, v)
                log.error(errmsg)
                raise ValueError(errmsg)

    def _sample(self, k, v):
        if k in self._attrs:
            getattr(self, k).push(v)

    def report(self, t=None):
        """Print out the collected statistics nicely. If t is provided, it's
        expected to the the simulation end time; if t is ignored, the
        statistics are up to the time of the last sample."""

        for k, v in self._attrs.items():
            if isinstance(v, TimeMarks):
                print("%s (timemarks): samples=%d" % (k, len(v)))
                if len(v) > 0:
                    d = v.data()
                    if d is not None:
                        print("  data=%r ..." % d[:3])
                    print("  rate = %g" % v.rate(t))
            elif isinstance(v, DataSeries):
                print("%s (dataseries): samples=%d" % (k, len(v)))
                if len(v) > 0:
                    d = v.data()
                    if d is not None:
                        print("  data=%r ..." % d[:3])
                    print("  mean = %g" % v.mean())
                    if len(v) > 1:
                        print("  stdev = %g" % v.stdev())
                        print("  var = %g" % v.var())
                    print("  min = %g" % v.min())
                    print("  max = %g" % v.max())
            elif isinstance(v, TimeSeries):
                print("%s (timeseries): samples=%d" % (k, len(v)))
                if len(v) > 0:
                    d = v.data()
                    if d is not None:
                        print("  data=%r ..." % d[:3])
                    print("  rate = %g" % v.rate(t))
                    print("  mean = %g" % v.mean())
                    if len(v) > 1:
                        print("  stdev = %g" % v.stdev())
                        print("  var = %g" % v.var())
                    print("  min = %g" % v.min())
                    print("  max = %g" % v.max())
                    print("  avg_over_time=%g" % v.avg_over_time(t))


# PQDict is PRIORITY QUEUE DICTIONARY (PYTHON RECIPE)
# Created by Nezar Abdennur
# Python recipes (4591):
# http://code.activestate.com/recipes/578643-priority-queue-dictionary/
#
# An indexed priority queue implemented in pure python as a dict-like
# class. It is a stripped-down version of pqdict. A Priority Queue
# Dictionary maps dictionary keys (dkeys) to updatable priority keys
# (pkeys).
#
# The priority queue is implemented as a binary heap, which supports:
#       O(1) access to the top priority element
#       O(log n) removal of the top priority element
#       O(log n) insertion of a new element
#
# In addition, an internal dictionary or "index" maps dictionary keys
# to the position of their entry in the heap. This index is maintained
# as the heap is manipulated. As a result, a PQ-dict also supports:
#       O(1) lookup of an arbitrary element's priority key
#       O(log n) removal of an arbitrary element
#       O(log n) updating of an arbitrary element's priority key
# PQDict is modified to be used for our event list


class _MinEntry_(object):
    """
    Mutable entries for a Min-PQ dictionary.

    """

    def __init__(self, dkey, pkey):
        self.dkey = dkey  # dictionary key
        self.pkey = pkey  # priority key

    def __lt__(self, other):
        return self.pkey < other.pkey


class _MaxEntry_(object):
    """
    Mutable entries for a Max-PQ dictionary.

    """

    def __init__(self, dkey, pkey):
        self.dkey = dkey
        self.pkey = pkey

    def __lt__(self, other):
        return self.pkey > other.pkey


class _PQDict_(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._heap = []
        self._position = {}
        self.update(*args, **kwargs)

    create_entry = _MinEntry_  # defaults to a min-pq

    @classmethod
    def maxpq(cls, *args, **kwargs):
        pq = cls()
        pq.create_entry = _MaxEntry_
        pq.__init__(*args, **kwargs)
        return pq

    def __len__(self):
        return len(self._heap)

    def __iter__(self):
        for entry in self._heap:
            yield entry.dkey

    def __getitem__(self, dkey):
        return self._heap[self._position[dkey]].pkey

    def __setitem__(self, dkey, pkey):
        heap = self._heap
        position = self._position

        try:
            pos = position[dkey]
        except KeyError:
            # Add a new entry:
            # put the new entry at the end and let it bubble up
            pos = len(self._heap)
            heap.append(self.create_entry(dkey, pkey))
            position[dkey] = pos
            self.swim(pos)
        else:
            # Update an existing entry:
            # bubble up or down depending on pkeys of parent and children
            heap[pos].pkey = pkey
            parent_pos = (pos - 1) >> 1
            child_pos = 2 * pos + 1
            if parent_pos > -1 and heap[pos] < heap[parent_pos]:
                self.swim(pos)
            elif child_pos < len(heap):
                other_pos = child_pos + 1
                if other_pos < len(heap) and not heap[child_pos] < heap[other_pos]:
                    child_pos = other_pos
                if heap[child_pos] < heap[pos]:
                    self.sink(pos)

    def __delitem__(self, dkey):
        heap = self._heap
        position = self._position

        pos = position.pop(dkey)
        entry_to_delete = heap[pos]

        # Take the very last entry and place it in the vacated spot. Let it
        # sink or swim until it reaches its new resting place.
        end = heap.pop(-1)
        if end is not entry_to_delete:
            heap[pos] = end
            position[end.dkey] = pos
            parent_pos = (pos - 1) >> 1
            child_pos = 2 * pos + 1
            if parent_pos > -1 and heap[pos] < heap[parent_pos]:
                self.swim(pos)
            elif child_pos < len(heap):
                other_pos = child_pos + 1
                if other_pos < len(heap) and not heap[child_pos] < heap[other_pos]:
                    child_pos = other_pos
                if heap[child_pos] < heap[pos]:
                    self.sink(pos)
        del entry_to_delete

    def peek(self):
        try:
            entry = self._heap[0]
        except IndexError:
            raise KeyError
        return entry.dkey, entry.pkey

    def popitem(self):
        heap = self._heap
        position = self._position

        try:
            end = heap.pop(-1)
        except IndexError:
            raise KeyError

        if heap:
            entry = heap[0]
            heap[0] = end
            position[end.dkey] = 0
            self.sink(0)
        else:
            entry = end
        del position[entry.dkey]
        return entry.dkey, entry.pkey

    def iteritems(self):
        # destructive heapsort iterator
        try:
            while True:
                yield self.popitem()
        except KeyError:
            return

    def sink(self, top=0):
        # "Sink-to-the-bottom-then-swim" algorithm (Floyd, 1964)
        # Tends to reduce the number of comparisons when inserting "heavy" items
        # at the top, e.g. during a heap pop
        heap = self._heap
        position = self._position

        # Grab the top entry
        pos = top
        entry = heap[pos]
        # Sift up a chain of child nodes
        child_pos = 2 * pos + 1
        while child_pos < len(heap):
            # choose the smaller child
            other_pos = child_pos + 1
            if other_pos < len(heap) and not heap[child_pos] < heap[other_pos]:
                child_pos = other_pos
            child_entry = heap[child_pos]
            # move it up one level
            heap[pos] = child_entry
            position[child_entry.dkey] = pos
            # next level
            pos = child_pos
            child_pos = 2 * pos + 1
        # We are left with a "vacant" leaf. Put our entry there and let it swim
        # until it reaches its new resting place.
        heap[pos] = entry
        position[entry.dkey] = pos
        self.swim(pos, top)

    def swim(self, pos, top=0):
        heap = self._heap
        position = self._position

        # Grab the entry from its place
        entry = heap[pos]
        # Sift parents down until we find a place where the entry fits.
        while pos > top:
            parent_pos = (pos - 1) >> 1
            parent_entry = heap[parent_pos]
            if not entry < parent_entry:
                break
            heap[pos] = parent_entry
            position[parent_entry.dkey] = pos
            pos = parent_pos
        # Put entry in its new place
        heap[pos] = entry
        position[entry.dkey] = pos
