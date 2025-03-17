import legate
from legate.core import get_legate_runtime
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import int8

GPU = legate.core.VariantCode.GPU
SYSMEM = legate.core.StoreTarget.SYSMEM
runtime = get_legate_runtime()

@task(variants=(GPU,))
def increment(
    dst: OutputStore, src: InputStore
) -> None:
    cupy.asarray(dst)[:] = cupy.asarray(src) + 1

shape = (2**31,)
x = runtime.create_store(int8, shape)
runtime.issue_fill(x, 42)
y = runtime.create_store(int8, shape)
z = runtime.create_store(int8, shape)

# intermediate values are materialized
# but obey typical reference counting rules
increment(y, x)
del x

# can also force offload to another memory
increment(z, y)
y.offload_to(SYSMEM)

# in-place updates save on intermediates
increment(z, z)
