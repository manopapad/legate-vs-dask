This repository contains the examples accompanying this talk: https://www.nvidia.com/gtc/session-catalog/?#/session/1727453996006001bDEq.

These examples were tested on an x86 workstation with 2 GPUs. The may not work on a machine with a different number of GPUs.

## Legate examples

Create a conda environment as follows:

```
conda create -n legate -c legate/label/experimental -c legate/label/ucc140 -c conda-forge cupynumeric=25.05.00.dev11
```

These example use some features from an unreleased version of Legate (as of the date of the talk), so nightly builds are used. Later releases will probably continue to work.

Activate the environment:

```
conda activate legate
```

Run any of the examples in the `legate/` directory as follows:

```
LEGATE_TEST=1 legate --fbmem 5000 --sysmem 5000 <prog.py>
```

`LEGATE_TEST=1` forces Legate to always partition using one piece per GPU. Without this, Legate will most likely opt to use fewer GPUs for the small input sizes used in some of these examples.

`--fbmem 5000` forces Legate to leave some device memory for cupy's internal uses.

`--sysmem 5000` assigns enough memory to the host memory pool, to satisfy some example tasks that are running on the CPU.

### Downstream library interoperability example

The `downstream.py` example requires a different environment, that contains `legate-dataframe`:

```
conda create -n legate-2 -c legate/label/experimental -c rapidsai -c conda-forge legate-dataframe cupynumeric
```

### NCCL interoperability example

Use the `legate` environment from above, and run the `./build.sh` script inside the `nccl-interop/` directory to build the example. A recent version of cmake is required.

Run the resulting executable as follows:

```
LEGATE_TEST=1 ./test_nccl
```

## Dask examples

Create a conda environment as follows:

```
conda create -n dask -c conda-forge dask=2023.9.2 dask-cuda=23.10.0 cupy python=3.10
```

Python 3.10 was necessary at the time of the talk because of https://github.com/dask/dask/issues/11038. More recent versions of these package will most likely work.

Active the environment:

```
conda activate dask
```

Run any of the examples in the `dask/` directory as follows:

```
python <prog.py>
```