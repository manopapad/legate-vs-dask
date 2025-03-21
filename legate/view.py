import cupy
import legate
import numpy
from legate.core import get_legate_runtime
from legate.core.task import OutputStore, task
from legate.core.types import int64

CPU = legate.core.VariantCode.CPU
GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()


@task(variants=(CPU, GPU))
def fill(to_fill: OutputStore, val: int) -> None:
    try:
        arr = numpy.asarray(to_fill)
    except ValueError:
        arr = cupy.asarray(to_fill)
    arr[:] = val


array = runtime.create_array(int64, (9, 9))

fill(array, 0)

fill(array[1:-1, 1:-1], 1)

fill(array[1:-1, 1:-1][1:-1, 1:-1], 2)

assert numpy.sum(array.get_physical_array()) == 74
