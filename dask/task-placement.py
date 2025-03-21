import cupy as cp
import numpy as np
from numpy import int8

import dask
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # one worker per GPU
    w0, w1 = client.scheduler_info()["workers"].keys()

    shape = (2**21,)
    chunks = (2**20,)

    with dask.config.set({"array.backend": "cupy"}):
        # data is initialized in two chunks
        x = dask.array.ones(shape, chunks=chunks, dtype=int8)

    # full data is pulled to GPU 0, work happens there
    with dask.annotate(workers=(w0,)):
        y = x + 1

    assert y.max().compute() == 2
