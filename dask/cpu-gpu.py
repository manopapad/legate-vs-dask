import cupy
import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from numpy import int8

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    shape = (2**31,)
    chunks = (2**30,)

    # starts out as collection of NumPy chunks, on RAM
    x = dask.array.ones(shape, chunks=chunks, dtype=int8)

    # move each chunk to GPU memory, in parallel
    y = x.map_blocks(cupy.asarray)

    # back to NumPy
    z = y.map_blocks(cupy.asnumpy)

    assert z.max().compute() == 1
