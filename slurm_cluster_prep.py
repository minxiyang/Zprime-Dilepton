import sys
sys.path.append("copperhead/")

import pytest
import dask
from dask.distributed import Client
from dask.distributed import Scheduler, Worker
from dask_jobqueue import SLURMCluster
from coffea.processor import dask_executor
from processNano.dimuon_processor import DimuonProcessor

dask.config.set({"temporary-directory": "/tmp/dask-temp/"})
dask.config.set({"distributed.worker.timeouts.connect": "60s"})

__all__ = [
    "pytest",
    "dask",
    "Client",
    "Scheduler",
    "Worker",
    "SLURMCluster",
    "dask_executor",
    "DimuonProcessor",
]

print("Dask version:", dask.__version__)
