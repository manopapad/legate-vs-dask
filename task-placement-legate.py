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

shape = (2**20,)
x = runtime.create_store(int8, shape)
runtime.issue_fill(x, 1)

# full data is pulled to GPU 0, work happens there
with machine.only(TaskTarget.GPU)[0]:
    increment(x, x)

assert(cupynumeric.asarray(x).max() == 2)
