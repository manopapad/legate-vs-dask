from legate.core.task import task

@task
def hello():
    print("Hello, world")

hello()
