# import time
import subprocess
import glob

import uproot

from config.parameters import parameters
from config.cross_sections import cross_sections


def read_via_xrootd(server, path):
    #command = f"xrdfs {server} ls -R {path} | grep '.root'"
    command = 'dasgoclient --query=="file dataset= %s" '% path
    #print(command)
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, shell=True)
    result = proc.stdout.readlines()
    #print(result)
    #print(proc.stderr.readlines())
    #print(proc.stdout.readlines())
    if proc.stderr.readlines():
        print("Loading error! Check VOMS proxy.")
    result = [server + r.rstrip().decode("utf-8") for r in result]
    return result


class SamplesInfo(object):
    def __init__(self, **kwargs):
        self.year = kwargs.pop('year', '2016')
        self.out_path = kwargs.pop('out_path', '/output/')
        self.xrootd = kwargs.pop('xrootd', True)
        self.server = kwargs.pop('server', 'root://xrootd.rcac.purdue.edu/')
        self.timeout = kwargs.pop('timeout', 60)
        self.debug = kwargs.pop('debug', False)
        datasets_from = kwargs.pop('datasets_from', 'Zprime')

        self.parameters = {k: v[self.year] for k, v in parameters.items()}

        self.is_mc = True

        if 'mu' in datasets_from:
            from config.datasets_muon import datasets
        elif 'el' in datasets_from:
            from config.datasets_electron import datasets
        #print(self.year)
        #print(datasets_from)
        #print(datasets)
        self.paths = datasets[self.year]

        if '2016' in self.year:
            self.lumi = 35900.
        elif '2017' in self.year:
            self.lumi = 41530.
        elif '2018' in self.year:
            self.lumi = 59970.
 

        self.data_entries = 0
        self.sample = ''
        self.samples = []

        self.fileset = {}
        self.metadata = {}

        self.lumi_weights = {}

    def load(self, sample, use_dask, client=None):
        if 'data' in sample:
            self.is_mc = False

        res = self.load_sample(sample, use_dask, client)

        self.sample = sample
        self.samples = [sample]
        self.fileset = {sample: res['files']}

        self.metadata = res['metadata']
        self.data_entries = res['data_entries']


    def load_sample(self, sample, use_dask=False, client=None):
        if sample not in self.paths:
            print(f"Couldn't load {sample}! Skipping.")
            return {'sample': sample, 'metadata': {},
                    'files': {}, 'data_entries': 0, 'is_missing': True}

        all_files = []
        metadata = {}
        data_entries = 0
        #print(self.xroot)
        if self.xrootd:
            all_files = read_via_xrootd(self.server, self.paths[sample])
            #all_files = [self.server + _file for _file in self.paths[sample]]
        elif self.paths[sample].endswith('.root'):
            all_files = [self.paths[sample]]
        else:
            all_files = [self.server + f for f in
                         glob.glob(self.paths[sample]+'/**/**/*.root')]

        if self.debug:
            all_files = [all_files[0]]

        # print(f"Loading {sample}: {len(all_files)} files")

        sumGenWgts = 0
        nGenEvts = 0

        if use_dask:
            from dask.distributed import get_client
            if not client:
                client = get_client()
            if 'data' in sample:
                work = client.map(self.get_data, all_files, priority=100)
            else:
                work = client.map(self.get_mc, all_files, priority=100)
            for w in work:
                ret = w.result()
                if 'data' in sample:
                    data_entries += ret['data_entries']
                else:
                    sumGenWgts += ret['sumGenWgts']
                    nGenEvts += ret['nGenEvts']
        else:
            for f in all_files:
                if 'data' in sample:
                    tree = uproot.open(f, timeout=self.timeout)['Events']
                    data_entries += tree.num_entries
                else:
                    tree = uproot.open(f, timeout=self.timeout)['Runs']
                    if (('NanoAODv6' in self.paths[sample]) or
                            ('NANOV10' in self.paths[sample])):
                        sumGenWgts += tree['genEventSumw_'].array()[0]
                        nGenEvts += tree['genEventCount_'].array()[0]
                    else:
                        sumGenWgts += tree['genEventSumw'].array()[0]
                        nGenEvts += tree['genEventCount'].array()[0]
        metadata['sumGenWgts'] = sumGenWgts
        metadata['nGenEvts'] = nGenEvts

        files = {
            'files': all_files,
            'treename': 'Events'
        }
        return {'sample': sample, 'metadata': metadata, 'files': files,
                'data_entries': data_entries, 'is_missing': False}

    def get_data(self, f):
        ret = {}
        file = uproot.open(f, timeout=self.timeout)
        tree = file['Events']
        ret['data_entries'] = tree.num_entries
        return ret

    def get_mc(self, f):
        ret = {}
        tree = uproot.open(f, timeout=self.timeout)['Runs']
        if ('NanoAODv6' in f) or ('NANOV10' in f):
            ret['sumGenWgts'] = tree['genEventSumw_'].array()[0]
            ret['nGenEvts'] = tree['genEventCount_'].array()[0]
        else:
            ret['sumGenWgts'] = tree['genEventSumw'].array()[0]
            ret['nGenEvts'] = tree['genEventCount'].array()[0]
        return ret

    def finalize(self):
        if self.is_mc:
            N = self.metadata['sumGenWgts']
            numevents = self.metadata['nGenEvts']
            if isinstance(cross_sections[self.sample], dict):
                xsec = cross_sections[self.sample][self.year]
            else:
                xsec = cross_sections[self.sample]
            if N > 0:
                self.lumi_weights[self.sample] = xsec * self.lumi / N
            else:
                self.lumi_weights[self.sample] = 0
            # print(f"{self.sample}: events={numevents}")
            return numevents
        else:
            return self.data_entries
