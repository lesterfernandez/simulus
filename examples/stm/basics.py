# mpiexec -np 1 python examples/stm/basics.py

from simulus.stm import STMBuilder

stm = STMBuilder().create_channels(["ch1"]).create_reader("ch1", "ch1_reader").build()
stm.start()

reader = stm._local_readers["ch1_reader"]
writer = stm.attach_writer("ch1")

writer.put(1, "HELLO, THIS IS DATA")
print(reader.get(1))

stm.stop()
