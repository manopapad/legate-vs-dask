from numpy import int32
import cupy
import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # array instantiated on the first local GPU
    x = cupy.arange(2**21, dtype=int32)

    # tiled and scattered across the cluster's 2 GPUs
    # Dask-CUDA recognizes that source and destination
    # are both GPU memories, and uses NVLink/GPUDirectRDMA
    chunks = (2**20,)
    with dask.config.set({"array.backend": "cupy"}):
        y = dask.array.from_array(x, chunks)

    # force the data back onto local GPU memory
    z = y.compute()
    assert cupy.all(x == z)
