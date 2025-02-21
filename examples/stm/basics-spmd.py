# mpiexec -np 3 python examples/stm/basics-spmd.py

import time
from simulus.stm import STMBuilder
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

s = STMBuilder()
if rank == 0:
    s.create_channels(["ch1"])

with s.build() as s:
    if rank == 0:
        # time.sleep(0.001) # delay
        writer = s.attach_writer("ch1")
        writer.put(1, "HELLO, THIS IS DATA", s)
    else:
        time.sleep(0.3) # delay
        reader = s.attach_reader("ch1")
        print(f"({rank}) {reader.get(1)}")
        print(f"({rank}) {reader.get(2)}")