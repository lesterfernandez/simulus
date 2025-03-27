# mpiexec -np 3 python examples/stm/consume-spmd.py

from time import sleep
from simulus.stm import STMBuilder
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

b = STMBuilder()
if rank == 0:
    b.create_channels(["ch1"])
else:
    b.create_reader("ch1", f"ch1_reader_{rank}")

with b.build() as stm:
    if rank == 0:
        writer = stm.attach_writer("ch1")
        writer.put(1, f"data({1})")
        writer.put(2, f"data({2})")
        writer.put(3, f"data({3})")
        writer.put(4, f"data({4})")
        writer.put(5, f"data({5})")
    elif rank == 1:
        sleep(0.1)
        reader = stm.readers[f"ch1_reader_{rank}"]
        print(f"({rank}) {reader.data._data}")
        print(f"({rank}) consume_until(4)")
        reader.consume_until(4)
        print(f"({rank}) {reader.data._data}")
    elif rank == 2:
        sleep(0.5)
        reader = stm.readers[f"ch1_reader_{rank}"]
        print(f"({rank}) {reader.data._data}")
        print(f"({rank}) consume_until(3)")
        reader.consume_until(3)
        print(f"({rank}) {reader.data._data}")
        reader.consume_until(5)
