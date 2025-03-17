import cupy
import legate
import cupynumeric
from legate.core import get_legate_runtime, get_machine, TaskTarget
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import int8

GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()
machine = get_machine()

@task(variants=(GPU,))
def increment(
    dst: OutputStore, src: InputStore
) -> None:
    cupy.asarray(dst)[:] = cupy.asarray(src) + 1

shape = (2**31,)
x = runtime.create_store(int8, shape)
runtime.issue_fill(x, 1)

with machine.only(TaskTarget.GPU)[0]:
    # full data is pulled to GPU 0, task runs there
    increment(x, x)

with machine.only(TaskTarget.GPU)[1]:
    # full data is pulled to GPU 1, task runs there
    increment(x, x)

assert(cupynumeric.asarray(x).max() == 3)
