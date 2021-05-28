import os
import sys
[sys.path.append(i) for i in ['.', '..']]
import time

import coffea.processor as processor
from coffea.processor import dask_executor, run_uproot_job
from python.dielectron_processor import DielectronProcessor
from python.samples_info import SamplesInfo

import dask
from dask.distributed import Client

__all__ = ['Client']


def almost_equal(a, b):
    return (abs(a - b) < 10e-8)


if __name__ == "__main__":
    tick = time.time()

    client = dask.distributed.Client(
        processes=True,
        n_workers=1,
        threads_per_worker=1,
        memory_limit='2.9GB',
    )
    print('Client created')

    file_name = "ztoee_file_NanoAODv6.root"
    file_path = f"{os.getcwd()}/tests/samples/{file_name}"
    dataset = {'test': file_path}

    samp_info = SamplesInfo(xrootd=False, datasets_from='el', year='2018')
    samp_info.paths = dataset
    samp_info.load('test', use_dask=False)
    samp_info.lumi_weights['test'] = 1.
    executor = dask_executor
    executor_args = {
        'client': client,
        'schema': processor.NanoAODSchema,
        'use_dataframes': True,
        'retries': 0
    }
    processor_args = {
        'samp_info': samp_info,
        'do_timer': False,
        'do_btag_syst': False,
    }
    print(samp_info.fileset)
    output = run_uproot_job(samp_info.fileset, 'Events',
                            DielectronProcessor(**processor_args),
                            executor, executor_args=executor_args,
                            chunksize=10000)
    df = output.compute()
    print(df)

    elapsed = round(time.time() - tick, 3)
    print(f'Finished everything in {elapsed} s.')

    diele_mass = df.loc[df.event == 8001, 'dielectron_mass'].values[0]
    wgt = df.loc[df.event == 8001, 'wgt_nominal'].values[0]
    assert(df.shape[0] == 5035)
    assert(almost_equal(diele_mass, 2473.15505815))
    assert(almost_equal(wgt, 0.000108786717997054))
