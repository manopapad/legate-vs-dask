import cupy
import legate
from legate.core import align, broadcast, get_legate_runtime
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import complex64

GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()


@task(
    variants=(GPU,),
    constraints=(
        # partition these in the same way
        align("dst", "src"),
        # don't split the first dimension
        broadcast("src", (0,)),
    ),
)
def batched_fft(
    dst: OutputStore,
    src: InputStore,
):
    cp_dst = cupy.asarray(dst)
    cp_src = cupy.asarray(src)
    cp_dst[:] = cupy.fft.fftn(cp_src, axes=(0,))


shape = (1024, 1024)
x = runtime.create_store(complex64, shape)
y = runtime.create_store(complex64, shape)
runtime.issue_fill(y, 42)

batched_fft(x, y)
