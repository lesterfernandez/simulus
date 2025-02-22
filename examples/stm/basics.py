# mpiexec -np 1 python examples/stm/basics.py

from simulus.stm import STMBuilder

stm = STMBuilder().create_channels(["ch1"]).build()
stm.start()

reader = stm.attach_reader("ch1")
writer = stm.attach_writer("ch1")

writer.put(1, "HELLO, THIS IS DATA")
print(reader.get(1))

stm.stop()