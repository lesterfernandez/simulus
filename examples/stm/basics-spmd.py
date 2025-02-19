# mpiexec -np 2 python examples/stm/basics-spmd.py

import time
from simulus.stm import STM
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

s = STM()


if rank == 0:
    s.create_channel("ch1")
    writer = s.attach_writer("ch1")
    writer.put(1, "HELLO, THIS IS DATA", s)
elif rank == 1:
    time.sleep(1)
    reader = s.attach_reader("ch1")
    print(f"{rank}: {reader.get(1)}")
elif rank == 2:
    time.sleep(1)
    reader = s.attach_reader("ch1")
    print(f"{rank}: {reader.get(1)}")
