import cupy
import dask
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    # dask.array.ones((1000, 3000), chunks=(500, 3000))
    row_wise = (
        dask.delayed(cupy.ones)((500, 3000)),
        dask.delayed(cupy.ones)((500, 3000)),
    )

    # dask.array.rechunk(row_wise, (1000, 3000))
    top = dask.delayed(cupy.hsplit)(row_wise[0], 2)
    top_l = dask.delayed(lambda x: x[0])(top)
    top_r = dask.delayed(lambda x: x[1])(top)
    bot = dask.delayed(cupy.hsplit)(row_wise[1], 2)
    bot_l = dask.delayed(lambda x: x[0])(bot)
    bot_r = dask.delayed(lambda x: x[1])(bot)
    col_wise = (
        dask.delayed(cupy.vstack)((top_l, bot_l)),
        dask.delayed(cupy.vstack)((top_r, bot_r)),
    )

    assert dask.delayed(cupy.shape)(col_wise[0]).compute() == (1000, 1500)
    assert dask.delayed(cupy.shape)(col_wise[1]).compute() == (1000, 1500)
