import pandas as pd
import awkward as ak
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools
from concurrent.futures import ProcessPoolExecutor
import dask.dataframe as dd
import mplhep as hep
from matplotlib import gridspec
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator


variables_plot = [
    "mu1_genPartFlav",
    "mu2_genPartFlav",
    "mu1_sip3d",
    "mu2_sip3d",
    "mu1_pt",
    "mu2_pt",
    "dimuon_mass",
]

path_ttbar_inclusive = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/ttbar_inclusive/*.parquet"
path_dy = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/DY/*/*.parquet"
path_ttbar = (
    "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/ttbar/*/*.parquet"
)
path_save = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/"

file_dy = glob.glob(path_dy)
with ProcessPoolExecutor(max_workers=48) as executor:
    dfs = list(executor.map(dd.read_parquet, file_dy))
dy = dd.concat(dfs)

file_ttbar = glob.glob(path_ttbar)
with ProcessPoolExecutor(max_workers=48) as executor:
    dfs = list(executor.map(dd.read_parquet, file_ttbar))
ttbar_par = dd.concat(dfs)

file_ttbar_inclusive = glob.glob(path_ttbar_inclusive)
with ProcessPoolExecutor(max_workers=48) as executor:
    dfs = list(executor.map(dd.read_parquet, file_ttbar_inclusive))
ttbar_inclusive = dd.concat(dfs)


ttbar_inclusive_mass = (ttbar_inclusive["dimuon_mass"].compute()).values
# ttbar_inclusive_wgt=(ttbar_inclusive[ttbar_inclusive_mass<500]['wgt_nominal'].compute()).values

ttbar_inclusive = ttbar_inclusive[ttbar_inclusive["dimuon_mass"] < 500]
ttbar = dd.concat([ttbar_inclusive, ttbar_par])
ttbar_mass = ttbar["dimuon_mass"].compute()
ttbar_wgt = (ttbar["wgt_nominal"].compute()).values
dy_mass = dy["dimuon_mass"].compute()
dy_wgt = (dy["wgt_nominal"].compute()).values


for mu in ["mu1", "mu2"]:

    idx = mu + "_genPartFlav"
    sip = mu + "_sip3d"
    sip = mu + "_dz"
    pt = mu + "_pt"
    bins = np.linspace(-10, 10, 100)
    print("load the variables for %s" % mu)
    ttbar_sip = ttbar[sip].compute()
    ttbar_id = ttbar[idx].compute()
    ttbar_pt = ttbar[pt].compute()
    dy_sip = dy[sip].compute()
    dy_id = dy[idx].compute()
    dy_pt = dy[pt].compute()

    print("plot sip3d for %s" % mu)
    ttbar_sip_b = ttbar_sip[ttbar_id == 5]
    ttbar_sip_t = ttbar_sip[ttbar_id == 15]
    ttbar_pt_b = ttbar_pt[ttbar_id == 5]
    ttbar_pt_t = ttbar_pt[ttbar_id == 15]
    ttbar_sip_other = ttbar_sip[(ttbar_id != 5) & (ttbar_id != 15)]
    ttbar_pt_other = ttbar_pt[(ttbar_id != 5) & (ttbar_id != 15)]
    ttbar_wgt_b = ttbar_wgt[ttbar_id == 5]
    ttbar_wgt_t = ttbar_wgt[ttbar_id == 15]
    ttbar_wgt_other = ttbar_wgt[(ttbar_id != 5) & (ttbar_id != 15)]
    # bins=np.linspace(0,25,100)
    plt.hist(
        (ttbar_sip_b, ttbar_sip_t, ttbar_sip_other, dy_sip),
        bins,
        histtype="bar",
        stacked=True,
        color=("r", "y", "b", "g"),
        label=("ttbar b", "ttbar tau", "ttbar other", "DY"),
        weights=(ttbar_wgt_b, ttbar_wgt_t, ttbar_wgt_other, dy_wgt),
    )
    plt.xlabel(sip)
    plt.ylabel("event")
    plt.yscale("log")
    plt.legend()
    plt.savefig(path_save + sip + ".pdf")
    plt.clf()
    cuts = [400, 600, 800, 1000, 1400]
    for cut in cuts:

        print("plot sip3d with invariant mass greater than %s GeV" % str(cut))
        ttbar_sip1 = ttbar_sip[ttbar_mass > cut]
        ttbar_id1 = ttbar_id[ttbar_mass > cut]
        dy_sip1 = dy_sip[dy_mass > cut]
        dy_id1 = dy_id[dy_mass > cut]
        ttbar_wgt1 = ttbar_wgt[ttbar_mass > cut]
        dy_wgt1 = dy_wgt[dy_mass > cut]

        ttbar_sip1_b = ttbar_sip1[ttbar_id1 == 5]
        ttbar_sip1_t = ttbar_sip1[ttbar_id1 == 15]
        ttbar_sip1_other = ttbar_sip1[(ttbar_id1 != 5) & (ttbar_id1 != 15)]
        ttbar_wgt1_b = ttbar_wgt1[ttbar_id1 == 5]
        ttbar_wgt1_t = ttbar_wgt1[ttbar_id1 == 15]
        ttbar_wgt1_other = ttbar_wgt1[(ttbar_id1 != 5) & (ttbar_id1 != 15)]

        plt.hist(
            (ttbar_sip1_b, ttbar_sip1_t, ttbar_sip1_other, dy_sip1),
            bins,
            histtype="bar",
            stacked=True,
            color=("r", "y", "b", "g"),
            label=("ttbar b", "ttbar tau", "ttbar other", "DY"),
            weights=(ttbar_wgt1_b, ttbar_wgt1_t, ttbar_wgt1_other, dy_wgt1),
        )
        plt.xlabel(sip)
        plt.ylabel("event")
        plt.yscale("log")
        # plt.ylim([1e-11, 1e2])
        plt.legend()
        plt.savefig(path_save + sip + str(cut) + "_pt.pdf")
        plt.clf()


print("plot mu1_sip3d vs mu2_sip3d")

ttbar_sipa = ttbar["mu1_dz"].compute()
ttbar_sipb = ttbar["mu2_dz"].compute()
dy_sipa = dy["mu1_dz"].compute()
dy_sipb = dy["mu2_dz"].compute()

plt.hist2d(
    ttbar_sipa,
    ttbar_sipb,
    bins=[bins, bins],
    weights=ttbar_wgt,
    norm=mpl.colors.LogNorm(),
)
plt.xlabel("mu1_dz")
plt.ylabel("mu2_dz")
plt.savefig(path_save + "ttbar_mu1dxy_vs_mu2dz.pdf")
plt.clf()

plt.hist2d(
    dy_sipa, dy_sipb, bins=[bins, bins], weights=dy_wgt, norm=mpl.colors.LogNorm()
)
plt.xlabel("mu1_dz")
plt.ylabel("mu2_dz")
plt.savefig(path_save + "dy_mu1dxy_vs_mu2dz.pdf")
plt.clf()


for cut in cuts:
    if cut == 400:
        edge = 3.0
    if cut == 600:
        edge = 2.0
    if cut == 800:
        edge = 1.0
    if cut == 1000:
        edge = 0.5
    if cut == 1400:
        edge = 0.3
    bins = np.linspace(-edge, edge, 100)
    ttbar_sipa1 = ttbar_sipa[ttbar_mass > cut]
    ttbar_sipb1 = ttbar_sipb[ttbar_mass > cut]
    ttbar_wgt1 = ttbar_wgt[ttbar_mass > cut]
    dy_sipa1 = dy_sipa[dy_mass > cut]
    dy_sipb1 = dy_sipb[dy_mass > cut]
    dy_wgt1 = dy_wgt[dy_mass > cut]

    plt.hist2d(
        ttbar_sipa1,
        ttbar_sipb1,
        bins=[bins, bins],
        weights=ttbar_wgt1,
        norm=mpl.colors.LogNorm(),
    )
    plt.xlabel("mu1_dz")
    plt.ylabel("mu2_dz")
    plt.savefig(path_save + "ttbar_mu1dz_vs_mu2dz" + str(cut) + ".pdf")
    plt.clf()

    plt.hist2d(
        dy_sipa1,
        dy_sipb1,
        bins=[bins, bins],
        weights=dy_wgt1,
        norm=mpl.colors.LogNorm(),
    )
    plt.xlabel("mu1_dz")
    plt.ylabel("mu2_dz")
    plt.savefig(path_save + "dy_mu1dz_vs_mu2dz" + str(cut) + ".pdf")
    plt.clf()
