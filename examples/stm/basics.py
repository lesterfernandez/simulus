# mpiexec -np 1 python examples/stm/basics.py

from simulus.stm import STMBuilder

stm = (
    STMBuilder()
    .create_channels(["ch1"])
    .create_reader("ch1", "ch1_reader")
    .create_writer("ch1", "ch1_writer")
    .build()
)
stm.start()

reader = stm.get_reader("ch1_reader")
writer = stm.get_writer("ch1_writer")

writer.put(1, "HELLO, THIS IS DATA")
print(reader.get(1))

stm.stop()
