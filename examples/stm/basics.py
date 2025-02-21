# mpiexec -np 1 python examples/stm/basics.py

from simulus.stm import STMBuilder

s = STMBuilder().create_channels(["ch1"]).build()
s.start()

reader = s.attach_reader("ch1")
writer = s.attach_writer("ch1")

writer.put(1, "HELLO, THIS IS DATA", s)
print(reader.get(1))

s.stop()