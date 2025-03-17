import dask
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from numpy import int8

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

def increment(arr: cupy.ndarray) -> cupy.ndarray:
    return arr + 1

if __name__ == "__main__":

    shape = (2**31,)
    chunks = (2**30,)

    with dask.config.set({"array.backend": "cupy"}):
        x = dask.array.full(
            shape, 42, chunks=chunks, dtype=int8
        )

    # only the persisted values are kept around
    # refs to non-persisted values don't matter
    y = x.map_blocks(increment)
    z = y.map_blocks(increment)
    wait(z.persist())

    # dask-cuda heuristically spills objects to RAM if necessary
    # when memory utilization is high
