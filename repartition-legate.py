import cupy
import legate
from legate.core import get_legate_runtime, broadcast
from legate.core.task import InputStore, OutputStore, task
from legate.core.types import int8

GPU = legate.core.VariantCode.GPU
runtime = get_legate_runtime()

@task(variants=(GPU,),
      constraints=(broadcast("x", (1,)),))
def row_wise(x: InputStore) -> None:
    assert cupy.asarray(x).shape == (500, 3000)

@task(variants=(GPU,),
      constraints=(broadcast("x", (0,)),))
def col_wise(x: InputStore) -> None:
    assert cupy.asarray(x).shape == (1000, 1500)

store = runtime.create_store(int8, (1000, 3000))
runtime.issue_fill(store, 1)

# Store starts out row-partitioned
row_wise(store)

# Legate recognizes the store should be repartitioned
col_wise(store)
