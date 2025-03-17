from numpy import int32
from dask_cuda import LocalCUDACluster
import dask
from dask.distributed import Client, wait

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    shape = (1024,)
    ary = dask.array.ones(shape, dtype=int32)

    # dataframe knows to interpret array's partitioning
    # so it can build a task graph to convert from it
    # in this case, no repartitioning needs to happen
    # but the series is a copy of the original array
    series = dask.dataframe.from_dask_array(ary)
    assert series.sum().compute() == 1024

    # each collection must know the internals of any other
    # that it wants to import from / export to
