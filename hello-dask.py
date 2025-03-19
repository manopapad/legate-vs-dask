import dask

@dask.delayed
def hello():
    print("Hello, world")

hello.compute()()
