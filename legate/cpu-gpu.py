import cupy
import cupynumeric
import numpy
import legate
from legate.core import get_legate_runtime, get_machine, TaskTarget
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import int8

CPU = legate.core.VariantCode.CPU
GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()
machine = get_machine()


@task(variants=(CPU,))
def initialize(dst: OutputStore) -> None:
    numpy.asarray(dst)[:] = 1


shape = (2**31,)
x = runtime.create_store(int8, shape)

# task can only run on CPU
# so the Store starts out on RAM
initialize(x)

# cuPyNumeric can handle both CPU and GPU execution
# but we explicitly force GPU execution
# Legate emits (potentially inter-node) h2d copies
with machine.only(TaskTarget.GPU):
    y = cupynumeric.add(x, 1)
