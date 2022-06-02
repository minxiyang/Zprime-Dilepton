import argparse
import dask
from dask.distributed import Client

from config.variables import variables_lookup
from produceResults.plotter import plotter, plotter2D
from produceResults.make_templates import to_templates

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
    ncpus_local = 1
    slurm_cluster_ip = ""
    dashboard_address = f"{node_ip}:34875"
else:
    slurm_cluster_ip = f"{node_ip}:{args.slurm_port}"
    dashboard_address = f"{node_ip}:8787"

# global parameters
parameters = {
    # < general settings >
    "slurm_cluster_ip": slurm_cluster_ip,
    "years": args.years,
    "global_path": "/depot/cms/users/schul105/Zprime-Dilepton/output/",
    "label": "correctionFix",
    "channels": ["inclusive", "0b", "1b", "2b"],
    "regions": ["bb", "be"],
    "syst_variations": ["nominal"],
    #
    # < plotting settings >
    #"plot_vars": ["dimuon_mass", "dimuon_mass_gen", "dimuon_mass_resUnc", "dimuon_mass_scaleUncUp", "dimuon_mass_scaleUncDown"],  # "dimuon_mass"],
    #"plot_vars": ["bmmj1_mass","dimuon_mass","njets","nbjets"],  # "dimuon_mass"],
    #"plot_vars": ["min_bl_mass"],  # "dimuon_mass"],
    "plot_vars": ["dimuon_mass","dimuon_mass_gen"],  # "dimuon_mass"],
    "plot_vars_2d": [["dimuon_mass","met"]],  # "dimuon_mass"],
    "variables_lookup": variables_lookup,
    "save_plots": True,
    "plot_ratio": True,
    "plots_path": "./plots/2022may27/",
    "dnn_models": {},
    "bdt_models": {},
    #
    # < templates and datacards >
    "save_templates": True,
    "templates_vars": ["dimuon_mass","dimuon_mass_gen"],  # "dimuon_mass"],
}

parameters["grouping"] = {
    "data_A": "Data",
    "data_B": "Data",
    "data_C": "Data",
    "data_D": "Data",
    "data_E": "Data",
    "data_F": "Data",
    "data_G": "Data",
    "data_H": "Data",
    "dy120to200" : "DY",
    "dy200to400" : "DY",
    "dy400to800" : "DY",
    "dy800to1400" : "DY",
    "dy1400to2300" : "DY",
    "dy2300to3500" : "DY",
    "dy3500to4500" : "DY",
    "dy4500to6000" : "DY",
    "dy6000toInf" : "DY",

#    "dy120to200" : "DY120to200",
#    "dy200to400" : "DY200to400",
#    "dy400to800" : "DY400to800",
#    "dy800to1400" : "DY800to1400",
#    "dy1400to2300" : "DY1400to2300",
#    "dy2300to3500" : "DY2300to3500",
#    "dy3500to4500" : "DY3500to4500",
#    "dy4500to6000" : "DY4500to6000",
#    "dy6000toInf" : "DY6000toInf",


    "ttbar_lep_inclusive" : "Other",
    "ttbar_lep_M500to800" : "Other",
    "ttbar_lep_M800to1200" : "Other",
    "ttbar_lep_M1200to1800" : "Other",
    "ttbar_lep_M1800toInf" : "Other",
    #"ttbar_lep_inclusive" : "ttbar_inclusive",
    #"ttbar_lep_M500to800" : "ttbar_500to800",
    #"ttbar_lep_M800to1200" : "ttbar_800to1200",
    #"ttbar_lep_M1200to1800" : "ttbar_1200to1800",
    #"ttbar_lep_M1800toInf" : "ttbar_1800toInf",
    "tW" : "Other",
    "Wantitop" : "Other",
    "WWinclusive" : "Other",
    "WW200to600" : "Other",
    "WW600to1200" : "Other",
    "WW1200to2500" : "Other",
    "WW2500" : "Other",
    "WZ2L2Q" : "Other",
    "WZ3LNu" : "Other",
    "ZZ2L2Nu" : "Other",
    "ZZ4L" : "Other",
    "dyInclusive50" : "Other",
    "bbll_4TeV_M400_posLL" : "bbll_4TeV_posLL",
    "bbll_4TeV_M1000_posLL" : "bbll_4TeV_posLL",
    "bbll_8TeV_M400_posLL" : "bbll_8TeV_posLL",
    "bbll_8TeV_M1000_posLL" : "bbll_8TeV_posLL",

}
# parameters["grouping"] = {"vbf_powheg_dipole": "VBF",}

parameters["plot_groups"] = {
    "stack": ["DY", "Other"],
#   "stack": ["DY120to200","DY200to400","DY400to800","DY800to1400","DY1400to2300","DY2300to3500","DY3500to4500","DY4500to60000","DY6000toInf","ttbar_inclusive", "ttbar_500to800", "ttbar_800to1200", "ttbar_1200to1800", "ttbar_1800toInf"],
    #"stack": [],
    "step": ["bbll_4TeV_posLL", "bbll_8TeV_posLL"],
    #"step": ["DY","ttbar","Other","bbll_4TeV_posLL", "bbll_8TeV_posLL"],
    "errorbar": [],
    #"2D": ["Data","DY","Other","bbll_4TeV_posLL","bbll_8TeV_posLL"],
    #"2D": ["DY","Other"],
}


if __name__ == "__main__":
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

    # add MVA scores to the list of variables to plot
    dnn_models = list(parameters["dnn_models"].values())
    bdt_models = list(parameters["bdt_models"].values())
    for models in dnn_models + bdt_models:
        for model in models:
            parameters["plot_vars"] += ["score_" + model]
            parameters["templates_vars"] += ["score_" + model]

    parameters["datasets"] = parameters["grouping"].keys()

    # make 1D plots
    yields = plotter(client, parameters)

    # make 1D plots
    #yields2D = plotter2D(client, parameters)

    # save templates to ROOT files
    yield_df = to_templates(client, parameters)

    # make datacards
    #build_datacards("score_pytorch_test", yield_df, parameters)
