# mpiexec -np 3 python examples/stm/basics-spmd.py

from simulus.stm import STMBuilder
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

b = STMBuilder()
if rank == 0:
    b.create_channels(["ch1"])

with b.build() as stm:
    if rank == 0:
        writer = stm.attach_writer("ch1")
        writer.put(1, "HELLO, THIS IS DATA")
    else:
        reader = stm.attach_reader("ch1")
        print(f"({rank}) {reader.get(1)}")
        print(f"({rank}) {reader.get(2)}")
