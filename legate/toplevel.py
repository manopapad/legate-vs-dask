import cupy
import legate
from legate.core import get_legate_runtime
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import int32

GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()


@task(variants=(GPU,))
def initialize(dst: OutputStore) -> None:
    cupy.asarray(dst)[:] = 1


@task(variants=(GPU,))
def increment(dst: OutputStore, src: InputStore) -> None:
    cupy.asarray(dst)[:] = cupy.asarray(src) + 1


# initialize data scattered across GPUs
x = runtime.create_store(int32, (2**20,))
initialize(x)

# pull the data onto first local GPU memory
y = cupy.asarray(x)
assert y.sum() == 2**20

# remote data and local view are kept in sync
increment(x, x)
assert y.sum() == 2**21
