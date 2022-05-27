import sys
sys.path.append("copperhead/")
import glob
import tqdm
import argparse
import dask
from dask.distributed import Client
import dask.dataframe as dd

from copperhead.python.io import load_dataframe
from doAnalysis.postprocessor import process_partitions

from copperhead.config.mva_bins import mva_bins
from config.variables import variables_lookup

__all__ = ["dask"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "-y", "--years", nargs="+", help="Years to process", default=["2018"]
)
parser.add_argument(
    "-sl",
    "--slurm",
    dest="slurm_port",
    default=None,
    action="store",
    help="Slurm cluster port (if not specified, will create a local cluster)",
)
args = parser.parse_args()

# Dask client settings
use_local_cluster = args.slurm_port is None
node_ip = "128.211.148.60"

if use_local_cluster:
    ncpus_local = 40
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "global_path": "/depot/cms/users/schul105/Zprime-Dilepton/output/",
    "years": args.years,
    "label": "correctionFix",
    "channels": ["inclusive", "0b", "1b", "2b"],
    "regions": ["bb", "be"],
    "syst_variations": ["nominal","resUnc","scaleUncUp","scaleUncDown"],
    # "custom_npartitions": {
    #     "vbf_powheg_dipole": 1,
    # },
    #
    # < settings for histograms >
    "hist_vars": ["min_bl_mass","bmmj1_mass","dimuon_mass", "dimuon_mass_gen",'njets','nbjets'],
    "hist_vars_2d": [["dimuon_mass", "met"]],
    "variables_lookup": variables_lookup,
    "save_hists": True,
    #
    # < settings for unbinned output>
    "tosave_unbinned": {
        "bb": ["dimuon_mass", "event", "wgt_nominal"],
        "be": ["dimuon_mass", "event", "wgt_nominal"],
    },
    "save_unbinned": True,
    #
    # < MVA settings >
    "models_path": "data/trained_models/",
    "dnn_models": {},
    "bdt_models": {},
    "mva_bins_original": mva_bins,
}
parameters["datasets"] = [
#    "data_A",
#    "data_B",
#    "data_C",
#    "data_D",
#    "data_E",
#    "data_F",
#    "data_G",
#    "data_H",
    "dy120to200",
    "dy200to400",
    "dy400to800",
    "dy800to1400",
    "dy1400to2300",
    "dy2300to3500",
    "dy3500to4500",
    "dy4500to6000",
    "dy6000toInf",
    "dyInclusive50",
    "ttbar_lep_inclusive",
    "ttbar_lep_M500to800",
    "ttbar_lep_M800to1200",
    "ttbar_lep_M1200to1800",
    "ttbar_lep_M1800toInf",
    "tW",
    "Wantitop",
    "WWinclusive",
    "WW200to600",
    "WW600to1200",
    "WW1200to2500",
    "WW2500",
    "WZ2L2Q",
    "WZ3LNu",
    "ZZ2L2Nu",
    "ZZ4L",
    "bbll_4TeV_M1000_negLL",
    "bbll_4TeV_M1000_negLR",
    "bbll_4TeV_M1000_posLL",
    "bbll_4TeV_M1000_posLR",
    "bbll_4TeV_M400_negLL",
    "bbll_4TeV_M400_negLR",
    "bbll_4TeV_M400_posLL",
    "bbll_4TeV_M400_posLR",
    "bbll_8TeV_M1000_negLL",
    "bbll_8TeV_M1000_negLR",
    "bbll_8TeV_M1000_posLL",
    "bbll_8TeV_M1000_posLR",
    "bbll_8TeV_M400_negLL",
    "bbll_8TeV_M400_negLR",
    "bbll_8TeV_M400_posLL",
    "bbll_8TeV_M400_posLR",
]

# using one small dataset for debugging
# parameters["datasets"] = ["vbf_powheg_dipole"]

if __name__ == "__main__":
    # prepare Dask client
    if use_local_cluster:
        print(
            f"Creating local cluster with {ncpus_local} workers."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(
            processes=True,
            #dashboard_address=dashboard_address,
            n_workers=ncpus_local,
            threads_per_worker=1,
            memory_limit="4GB",
        )
    else:
        print(
            f"Connecting to Slurm cluster at {slurm_cluster_ip}."
            f" Dashboard address: {dashboard_address}"
        )
        client = Client(parameters["slurm_cluster_ip"])
    parameters["ncpus"] = len(client.scheduler_info()["workers"])
    print(f"Connected to cluster! #CPUs = {parameters['ncpus']}")

    # add MVA scores to the list of variables to create histograms from
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["hist_vars"] += ["score_" + model]

    # prepare lists of paths to parquet files (stage1 output) for each year and dataset
    all_paths = {}
    for year in parameters["years"]:
        all_paths[year] = {}
        for dataset in parameters["datasets"]:
            paths = glob.glob(
                f"{parameters['global_path']}/"
                f"{parameters['label']}/stage1_output/{year}/"
                f"{dataset}/*.parquet"
            )
            all_paths[year][dataset] = paths

    # run postprocessing
    for year in parameters["years"]:
        print(f"Processing {year}")
        for dataset, path in tqdm.tqdm(all_paths[year].items()):
            if len(path) == 0:
                continue

            # read stage1 outputs
            df = load_dataframe(client, parameters, inputs=[path], dataset=dataset)
            if not isinstance(df, dd.DataFrame):
                continue

            # run processing sequence (categorization, mva, histograms)
            info = process_partitions(client, parameters, df)
            #print(info)
