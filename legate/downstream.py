import cupynumeric
from legate_dataframe import LogicalColumn
from legate_dataframe.lib.reduction import reduce
from numpy import int32
from pylibcudf import aggregation

shape = (1024,)
ary = cupynumeric.ones(shape, int32)

# cuPyNumeric and Legate-Dataframe can share data
# through the Legate Data Interface
# in this case, no repartitioning needs to happen
col = LogicalColumn(ary)
print(reduce(col, aggregation.sum(), int32))  # 1024

# the column and array refer to the same data
# whether or not repartitioning needed to happen
cupynumeric.add(ary, 1, out=ary)
print(reduce(col, aggregation.sum(), int32))  # 2048
