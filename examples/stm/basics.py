# mpiexec -np 1 python examples/stm/basics.py

from simulus.stm import STM

s = STM()
s.create_channel("ch1")

reader = s.attach_reader("ch1")
writer = s.attach_writer("ch1")

writer.put(1, "HELLO, THIS IS DATA", s)
print(reader.get(1))