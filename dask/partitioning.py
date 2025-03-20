import cupy
import dask
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from numpy import complex64

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)


def batched_fft(src: cupy.ndarray) -> cupy.ndarray:
    return cupy.fft.fftn(src, axes=(0,))


if __name__ == "__main__":

    shape = (1024, 1024)
    # manually make sure we don't split the first dimension
    chunks = (1024, 512)

    with dask.config.set({"array.backend": "cupy"}):
        x = dask.array.full(shape, 42, chunks=chunks, dtype=complex64)

    y = x.map_blocks(batched_fft)
    wait(y.persist())
