import numpy as np
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import glob
import math
from itertools import repeat
from setFrame import setFrame
import mplhep as hep
import time
import matplotlib.pyplot as plt
import copy
import random as rand

def load2df(files):

    df = dd.read_parquet(files)
    field = [
        "dimuon_mass",
        # "dimuon_cos_theta_cs",
        "njets",
        "dimuon_mass_gen",
        "wgt_nominal",
        "met",
    ]
    out = df[field]
    out.compute()
    return out


def chunk(files, size):

    size = math.ceil(len(files) / float(size))
    file_bag = [
        files[i : min(i + size, len(files))] for i in range(0, len(files), size)
    ]
    return file_bag

def df2hist(var, df, bins, masscut, njets=-1, iscut=False, iswgt=True, scale=True):
    cut = 120
    if var=="met":
        cut = 400

    if njets == -1:
        
        var_array = df.loc[df["dimuon_mass"] > cut, var].compute()
        if iswgt:

            wgt = df.loc[(df["dimuon_mass"] > cut), "wgt_nominal"].compute()
            wgt[wgt < 0] = 0
            if iscut:
                genmass = df.loc[(df["dimuon_mass"] > cut), "dimuon_mass_gen"].compute()
                wgt[genmass > masscut] = 0
            vals, bins = np.histogram(var_array, bins=bins, weights=wgt)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:
            vals, bins = np.histogram(var_array, bins=bins)
            errs = np.sqrt(vals)

    else:
        if njets == 1:
            var_array = df.loc[
                (df["njets"] == njets) & (df["dimuon_mass"] > cut), var
            ].compute()
        else:
            var_array = df.loc[
                (df["njets"] > 1) & (df["dimuon_mass"] > cut), var
            ].compute()

        if iswgt:
            if njets == 1:
                wgt = df.loc[
                    (df["njets"] == njets) & (df["dimuon_mass"] > cut), "wgt_nominal"
                ].compute()
            else:
                wgt = df.loc[
                    (df["njets"] > 1) & (df["dimuon_mass"] > cut), "wgt_nominal"
                ].compute()

            wgt[wgt < 0] = 0
            if iscut:
                if njets == 1:
                    genmass = df.loc[
                        (df["dimuon_mass"] > cut) & (df["njets"] == njets),
                        "dimuon_mass_gen",
                    ].compute()
                else:
                    genmass = df.loc[
                        (df["dimuon_mass"] > cut) & (df["njets"] > 1), "dimuon_mass_gen"
                    ].compute()

                wgt[genmass > masscut] = 0
            vals, bins = np.histogram(var_array, bins=bins, weights=wgt)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:
            vals, bins = np.histogram(var_array, bins=bins)
            errs = np.sqrt(vals)
    if scale:
        binSize = np.diff(bins)
        vals = vals / binSize
        errs = errs / binSize

    return [vals, errs, bins]


def plots(axes, MCs, labels, colors, name):

    bins = MCs[0][2]
    MCs_vals = [MC[0] for MC in MCs]

    hep.histplot(
        MCs_vals[0],
        bins,
        ax=axes[0],
        color=colors[0],
        histtype="fill",
        label=labels[0],
        alpha=0.7,
    )

    hep.histplot(
        MCs_vals[1:],
        bins,
        ax=axes[1],
        color=colors[1:],
        label=labels[1:],
    )

    axes[0].legend(loc=(0.55, 0.75), fontsize="medium")
    axes[1].legend(loc=(0.55, 0.55), fontsize="x-small")
    axes[2].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}.pdf"
    )
    axes[2].clf()


def plot_sen(axes, MC_sig, MC_bkg, colors, name):

    bins = MC_sig[2]
    MC_sig_vals = np.cumsum(MC_sig[0][::-1])[::-1]
    MC_bkg_vals = np.cumsum(MC_bkg[0][::-1])[::-1]
    MC_sig_vals = MC_sig_vals[0] - MC_sig_vals
    MC_bkg_vals = MC_bkg_vals[0] - MC_bkg_vals
    sen = MC_sig_vals/(MC_sig_vals + MC_bkg_vals)
    MC_sig_vals = MC_sig_vals/MC_sig_vals[-1]
    MC_bkg_vals = MC_bkg_vals/MC_bkg_vals[-1]
    hep.histplot(
        [MC_sig_vals, MC_bkg_vals, sen],
        bins,
        ax=axes[0],
        color=colors,
        label=["signal efficiency", "ttbar", "sensitivity"],
    )

    axes[0].legend(loc=(0.55, 0.55), fontsize="medium")
    axes[1].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}.pdf"
    )
    axes[1].clf()


if __name__ == "__main__":

    client_args = {
        "n_workers": 48,
        "memory_limit": "3.0GB",
        "timeout": 120,
    }

    bins_mass = (
        [j for j in range(400, 600, 20)]
        + [j for j in range(600, 900, 30)]
        + [j for j in range(900, 1250, 50)]
        + [j for j in range(1250, 1610, 60)]
        + [j for j in range(1610, 1890, 70)]
        + [j for j in range(1890, 3970, 80)]
    )
    bins_met = [j for j in range(0, 820, 10)]
    path = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/"
    
    path_tt_inclusive = path + "ttbar/ttbar_lep/*.parquet"
    tt_inclusive_files = glob.glob(path_tt_inclusive)
    path_tt = path + "ttbar/ttbar_lep_*/*.parquet"
    tt_files = glob.glob(path_tt)
    tt_files = [file_ for file_ in tt_files if "ext" not in file_]

    CI_list = [
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
    path_CI = {}
    for sig_name in CI_list:
        files = glob.glob(path + "CI/" + sig_name + "/*")
        path_CI[sig_name] = files

    path_CI["tt_inclu"] = tt_inclusive_files
    path_CI["tt"] = tt_files
    client = Client(LocalCluster(**client_args))
    mass_inclu = {}
    mass_0j = {}
    mass_1j = {}
    mass_2j = {}
    met_inclu = {}
    met_0j = {}
    met_1j = {}
    met_2j = {}
    df_dict={}
    for key in path_CI.keys():

        if key == "tt_inclu":
            masscut = 500.0
            iscut = True
        else:
            masscut = 0
            iscut = False
        file_bag = chunk(path_CI[key], client_args["n_workers"])
        if len(file_bag) == 1:
            df = load2df_mc(file_dict[key])
        else:
            results = client.map(load2df, file_bag)

            dfs = client.gather(results)
            df = dd.concat(dfs)
        iswgt = True
        df_dict[key] = copy.deepcopy(df)
        mass_inclu[key] = df2hist(
            "dimuon_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=-1,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_0j[key] = df2hist(
            "dimuon_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=0,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_1j[key] = df2hist(
            "dimuon_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=1,
            iscut=iscut,
            iswgt=iswgt,
        )
        mass_2j[key] = df2hist(
            "dimuon_mass",
            df,
            bins_mass,
            masscut=masscut,
            njets=2,
            iscut=iscut,
            iswgt=iswgt,
        )
        met_inclu[key] = df2hist(
            "met",
            df,
            bins_met,
            masscut=masscut,
            njets=-1,
            iscut=iscut,
            iswgt=iswgt,
        )
        met_0j[key] = df2hist(
            "met",
            df,
            bins_met,
            masscut=masscut,
            njets=0,
            iscut=iscut,
            iswgt=iswgt,
        )
        met_1j[key] = df2hist(
            "met",
            df,
            bins_met,
            masscut=masscut,
            njets=1,
            iscut=iscut,
            iswgt=iswgt,
        )
        met_2j[key] = df2hist(
            "met",
            df,
            bins_met,
            masscut=masscut,
            njets=2,
            iscut=iscut,
            iswgt=iswgt,
        )

    client.close()
    labels_8TeV = [
        "$t\\bar{t}$",
        "LL pos $\Lambda$ 8 TeV",
        "LR pos $\Lambda$ 8 TeV",
        "LL neg $\Lambda$ 8 TeV",
        "LR neg $\Lambda$ 8 TeV",
    ]
    labels_4TeV = [
        "$t\\bar{t}$",
        "LL pos $\Lambda$ 4 TeV",
        "LR pos $\Lambda$ 4 TeV",
        "LL neg $\Lambda$ 4 TeV",
        "LR neg $\Lambda$ 4 TeV",
    ]

    colors = ["red", "darkred", "brown", "blue", "skyblue"]

    mass_2j["tt"][0] += mass_2j["tt_inclu"][0]
    mass_2j["tt"][1] = np.sqrt(mass_2j["tt"][1] ** 2 + mass_2j["tt_inclu"][1] ** 2)
    mass_1j["tt"][0] += mass_1j["tt_inclu"][0]
    mass_1j["tt"][1] = np.sqrt(mass_1j["tt"][1] ** 2 + mass_1j["tt_inclu"][1] ** 2)
    mass_0j["tt"][0] += mass_0j["tt_inclu"][0]
    mass_0j["tt"][1] = np.sqrt(mass_0j["tt"][1] ** 2 + mass_0j["tt_inclu"][1] ** 2)
    mass_inclu["tt"][0] += mass_inclu["tt_inclu"][0]
    mass_inclu["tt"][1] = np.sqrt(
        mass_inclu["tt"][1] ** 2 + mass_inclu["tt_inclu"][1] ** 2
    )
    met_2j["tt"][0] += met_2j["tt_inclu"][0]
    met_2j["tt"][1] = np.sqrt(met_2j["tt"][1] ** 2 + met_2j["tt_inclu"][1] ** 2)
    met_1j["tt"][0] += met_1j["tt_inclu"][0]
    met_1j["tt"][1] = np.sqrt(met_1j["tt"][1] ** 2 + met_1j["tt_inclu"][1] ** 2)
    met_0j["tt"][0] += met_0j["tt_inclu"][0]
    met_0j["tt"][1] = np.sqrt(met_0j["tt"][1] ** 2 + met_0j["tt_inclu"][1] ** 2)
    met_inclu["tt"][0] += met_inclu["tt_inclu"][0]
    met_inclu["tt"][1] = np.sqrt(
        met_inclu["tt"][1] ** 2 + met_inclu["tt_inclu"][1] ** 2
    )

    for sample in CI_list:

        if "1000" in sample:
            sample_b = sample.replace("1000", "400")
            mass_2j[sample_b][0] += mass_2j[sample][0]
            mass_2j[sample_b][1] = np.sqrt(
                mass_2j[sample_b][1] ** 2 + mass_2j[sample][1] ** 2
            )
            mass_1j[sample_b][0] += mass_1j[sample][0]
            mass_1j[sample_b][1] = np.sqrt(
                mass_1j[sample_b][1] ** 2 + mass_1j[sample][1] ** 2
            )
            mass_0j[sample_b][0] += mass_0j[sample][0]
            mass_0j[sample_b][1] = np.sqrt(
                mass_0j[sample_b][1] ** 2 + mass_0j[sample][1] ** 2
            )
            mass_inclu[sample_b][0] += mass_inclu[sample][0]
            mass_inclu[sample_b][1] = np.sqrt(
                mass_inclu[sample_b][1] ** 2 + mass_inclu[sample][1] ** 2
            )

            met_2j[sample_b][0] += met_2j[sample][0]
            met_2j[sample_b][1] = np.sqrt(
                met_2j[sample_b][1] ** 2 + met_2j[sample][1] ** 2
            )
            met_1j[sample_b][0] += met_1j[sample][0]
            met_1j[sample_b][1] = np.sqrt(
                met_1j[sample_b][1] ** 2 + met_1j[sample][1] ** 2
            )
            met_0j[sample_b][0] += met_0j[sample][0]
            met_0j[sample_b][1] = np.sqrt(
                met_0j[sample_b][1] ** 2 + met_0j[sample][1] ** 2
            )
            met_inclu[sample_b][0] += met_inclu[sample][0]
            met_inclu[sample_b][1] = np.sqrt(
                met_inclu[sample_b][1] ** 2 + met_inclu[sample][1] ** 2
            )

    MCs8 = [
        mass_inclu["tt"],
        mass_inclu["bbll_8TeV_M400_posLL"],
        mass_inclu["bbll_8TeV_M400_posLR"],
        mass_inclu["bbll_8TeV_M400_negLL"],
        mass_inclu["bbll_8TeV_M400_negLR"],
    ]
    MCs4 = [
        mass_inclu["tt"],
        mass_inclu["bbll_4TeV_M400_posLL"],
        mass_inclu["bbll_4TeV_M400_posLR"],
        mass_inclu["bbll_4TeV_M400_negLL"],
        mass_inclu["bbll_4TeV_M400_negLR"],
    ]

    MCs8_1j = [
        mass_1j["tt"],
        mass_1j["bbll_8TeV_M400_posLL"],
        mass_1j["bbll_8TeV_M400_posLR"],
        mass_1j["bbll_8TeV_M400_negLL"],
        mass_1j["bbll_8TeV_M400_negLR"],
    ]
    MCs4_1j = [
        mass_1j["tt"],
        mass_1j["bbll_4TeV_M400_posLL"],
        mass_1j["bbll_4TeV_M400_posLR"],
        mass_1j["bbll_4TeV_M400_negLL"],
        mass_1j["bbll_4TeV_M400_negLR"],
    ]
    MCs8_2j = [
        mass_2j["tt"],
        mass_2j["bbll_8TeV_M400_posLL"],
        mass_2j["bbll_8TeV_M400_posLR"],
        mass_2j["bbll_8TeV_M400_negLL"],
        mass_2j["bbll_8TeV_M400_negLR"],
    ]
    MCs4_2j = [
        mass_2j["tt"],
        mass_2j["bbll_4TeV_M400_posLL"],
        mass_2j["bbll_4TeV_M400_posLR"],
        mass_2j["bbll_4TeV_M400_negLL"],
        mass_2j["bbll_4TeV_M400_negLR"],
    ]
    
    axes_mass8 = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-5, 1e6],
        yRange=[0,40],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass8, MCs8, labels_8TeV, colors, "bbll_muon_mass_CI8TeV_inclu_nolog")

    axes_mass4 = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-5, 1e6],
        yRange=[0,40],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass4, MCs4, labels_4TeV, colors, "bbll_muon_mass_CI4TeV_inclu_nolog")

    axes_mass8_1j = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-6, 1e4],
        yRange=[0,20],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass8_1j, MCs8_1j, labels_8TeV, colors, "bbll_muon_mass_CI8TeV_1j_nolog")

    axes_mass4_1j = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-6, 1e4],
        yRange=[0,20],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass4_1j, MCs4_1j, labels_4TeV, colors, "bbll_muon_mass_CI4TeV_1j_nolog")

    axes_mass8_2j = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-6, 1e3],
        yRange=[0,10],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass8_2j, MCs8_2j, labels_8TeV, colors, "bbll_muon_mass_CI8TeV_2j_nolog")

    axes_mass4_2j = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[400, 3970],
        #yRange=[1e-6, 1e3],
        yRange=[0,10],
        flavor="mu",
        year="2018",
    )

    plots(axes_mass4_2j, MCs4_2j, labels_4TeV, colors, "bbll_muon_mass_CI4TeV_2j_nolog")

    MCs_met8 = [
        met_inclu["tt"],
        met_inclu["bbll_8TeV_M400_posLL"],
        met_inclu["bbll_8TeV_M400_posLR"],
        met_inclu["bbll_8TeV_M400_negLL"],
        met_inclu["bbll_8TeV_M400_negLR"],
    ]
    MCs_met4 = [
        met_inclu["tt"],
        met_inclu["bbll_4TeV_M400_posLL"],
        met_inclu["bbll_4TeV_M400_posLR"],
        met_inclu["bbll_4TeV_M400_negLL"],
        met_inclu["bbll_4TeV_M400_negLR"],
    ]

    MCs_met8_1j = [
        met_1j["tt"],
        met_1j["bbll_8TeV_M400_posLL"],
        met_1j["bbll_8TeV_M400_posLR"],
        met_1j["bbll_8TeV_M400_negLL"],
        met_1j["bbll_8TeV_M400_negLR"],
    ]
    MCs_met4_1j = [
        met_1j["tt"],
        met_1j["bbll_4TeV_M400_posLL"],
        met_1j["bbll_4TeV_M400_posLR"],
        met_1j["bbll_4TeV_M400_negLL"],
        met_1j["bbll_4TeV_M400_negLR"],
    ]
    MCs_met8_2j = [
        met_2j["tt"],
        met_2j["bbll_8TeV_M400_posLL"],
        met_2j["bbll_8TeV_M400_posLR"],
        met_2j["bbll_8TeV_M400_negLL"],
        met_2j["bbll_8TeV_M400_negLR"],
    ]
    MCs_met4_2j = [
        met_2j["tt"],
        met_2j["bbll_4TeV_M400_posLL"],
        met_2j["bbll_4TeV_M400_posLR"],
        met_2j["bbll_4TeV_M400_negLL"],
        met_2j["bbll_4TeV_M400_negLR"],
    ]

    for sample in CI_list:

        if "1000" in sample:
            continue
        #if "4TeV" in sample:
        #    MCs1j = MCs_met4_1j
        #    MCs2j = MCs_met4_2j
        #else:
        #    MCs1j = MCs_met8_1j
        #    MCs2j = MCs_met8_2j


        axes = setFrame(
            "MET [GeV]",
            "percentage",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[10, 800],
            yRange=[0,1.7],
            flavor="mu",
            year="2018",
            )

        plot_sen(axes, met_1j[sample], met_1j["tt"], ["red", "green", "blue"], "signi_1j_"+sample.replace("400", ""))
        axes = setFrame(
            "MET [GeV]",
            "percentage",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[10, 800],
            yRange=[0,1.7],
            flavor="mu",
            year="2018",
            )

        plot_sen(axes, met_2j[sample], met_2j["tt"], ["red", "green", "blue"], "signi_2j_"+sample.replace("400", ""))



   
    axes_met8 = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,50],
        flavor="mu",
        year="2018",
    )

    plots(axes_met8, MCs_met8, labels_8TeV, colors, "bbll_muon_met_CI8TeV_inclu_nolog")

    axes_met4 = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,50],
        flavor="mu",
        year="2018",
    )

    plots(axes_met4, MCs_met4, labels_4TeV, colors, "bbll_muon_met_CI4TeV_inclu_nolog")

    axes_met8_1j = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,20],
        flavor="mu",
        year="2018",
    )

    plots(axes_met8_1j, MCs_met8_1j, labels_8TeV, colors, "bbll_muon_met_CI8TeV_1j_nolog")

    axes_met4_1j = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,20],
        flavor="mu",
        year="2018",
    )

    plots(axes_met4_1j, MCs_met4_1j, labels_4TeV, colors, "bbll_muon_met_CI4TeV_1j_nolog")

    axes_met8_2j = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,10],
        flavor="mu",
        year="2018",
    )

    plots(axes_met8_2j, MCs_met8_2j, labels_8TeV, colors, "bbll_muon_met_CI8TeV_2j_nolog")

    axes_met4_2j = setFrame(
        "MET [GeV]",
        "Events/GeV",
        ratio=False,
        signal=True,
        logx=False,
        logy=False,
        xRange=[20, 820],
        #yRange=[1e-4, 1e7],
        yRange=[0,10],
        flavor="mu",
        year="2018",
    )

    plots(axes_met4_2j, MCs_met4_2j, labels_4TeV, colors, "bbll_muon_met_CI4TeV_2j_nolog")


    for sample in CI_list:

        if "1000" in sample:
            if "LL" in sample:
                CI_label = "LL"
            elif "LR" in sample:
                CI_label = "LR"
            if "pos" in sample:
                CI_label += " pos"
            elif "neg" in sample:
                CI_label += " neg"
            CI_label += " $\Lambda$"  
            if "4TeV" in sample: 
                CI_label += " 4 TeV"
            elif "8TeV" in sample:
                CI_label += " 8 TeV"
            sample_b = sample.replace("1000", "400")
            df_tt = df_dict["tt"].sample(frac=0.01)
            df_inclu = df_dict["tt_inclu"].sample(frac=0.01)
            CI400 = df_dict[sample_b].sample(frac=0.5)
            CI1000 = df_dict[sample].sample(frac=0.5)
            plt.xlabel("mll")
            plt.ylabel("MET")
            plt.xlim([0, 3000])
            plt.ylim([0, 700]) 
            plt.scatter(df_tt["dimuon_mass"].compute(), df_tt["met"].compute(), 100.*df_tt["wgt_nominal"].compute(), c="r", alpha=0.5)
            plt.scatter(df_inclu["dimuon_mass"].compute(), df_inclu["met"].compute(), 100.*df_inclu["wgt_nominal"].compute(), c="r", label="$t\\bar{t}$", alpha=0.5)
            plt.scatter(CI400["dimuon_mass"].compute(), CI400["met"].compute(), 100.*CI400["wgt_nominal"].compute(), c="b", alpha=0.5)
            plt.scatter(CI1000["dimuon_mass"].compute(), CI1000["met"].compute(), 100.*CI1000["wgt_nominal"].compute(), c="b", label = CI_label, alpha=0.5)
            leg = plt.legend(loc="upper right")
            leg.legendHandles[0]._sizes = [30]
            leg.legendHandles[1]._sizes = [30]
            plt_name = sample.replace("1000", "")+"_MllVsMet_inclu.pdf"
            plt.savefig("plots/"+plt_name)
            plt.clf()
            plt.xlabel("mll")
            plt.ylabel("MET")
            plt.xlim([0, 3000])
            plt.ylim([0, 700])

            plt.scatter(df_tt.loc[df_tt["njets"]==1 ,"dimuon_mass"].compute(), df_tt.loc[df_tt["njets"]==1 ,"met"].compute(), 100.*df_tt.loc[df_tt["njets"]==1,"wgt_nominal"].compute(), c="r", alpha=0.5)
            plt.scatter(df_inclu.loc[df_inclu["njets"]==1, "dimuon_mass"].compute(), df_inclu.loc[df_inclu["njets"]==1, "met"].compute(), 100.*df_inclu.loc[df_inclu["njets"]==1, "wgt_nominal"].compute(), c="r", label="$t\\bar{t}$", alpha=0.5)
            plt.scatter(CI400.loc[CI400["njets"]==1, "dimuon_mass"].compute(), CI400.loc[CI400["njets"]==1, "met"].compute(), 100.*CI400.loc[CI400["njets"]==1, "wgt_nominal"].compute(), c="b", alpha=0.5)
            plt.scatter(CI1000.loc[CI1000["njets"]==1 ,"dimuon_mass"].compute(), CI1000.loc[CI1000["njets"]==1, "met"].compute(), 100.*CI1000.loc[CI1000["njets"]==1 ,"wgt_nominal"].compute(), c="b", label = CI_label, alpha=0.5)

            leg = plt.legend(loc="upper right")
            leg.legendHandles[0]._sizes = [30]
            leg.legendHandles[1]._sizes = [30]

            plt_name = sample.replace("1000", "")+"_MllVsMet_1j.pdf"
            plt.savefig("plots/"+plt_name)
            plt.clf()
            plt.xlabel("mll")
            plt.ylabel("MET")
            plt.xlim([0, 3000])
            plt.ylim([0, 700])

            plt.scatter(df_tt.loc[df_tt["njets"]>1 ,"dimuon_mass"].compute(), df_tt.loc[df_tt["njets"]>1 ,"met"].compute(), 300.*df_tt.loc[df_tt["njets"]>1,"wgt_nominal"].compute(), c="r", alpha=0.5)
            plt.scatter(df_inclu.loc[df_inclu["njets"]>1, "dimuon_mass"].compute(), df_inclu.loc[df_inclu["njets"]>1, "met"].compute(), 300.*df_inclu.loc[df_inclu["njets"]>1, "wgt_nominal"].compute(), c="r", label="$t\\bar{t}$", alpha=0.5)
            plt.scatter(CI400.loc[CI400["njets"]>1, "dimuon_mass"].compute(), CI400.loc[CI400["njets"]>1, "met"].compute(), 300.*CI400.loc[CI400["njets"]>1, "wgt_nominal"].compute(), c="b", alpha=0.5)
            plt.scatter(CI1000.loc[CI1000["njets"]>1 ,"dimuon_mass"].compute(), CI1000.loc[CI1000["njets"]>1, "met"].compute(), 300.**CI1000.loc[CI1000["njets"]>1 ,"wgt_nominal"].compute(), c="b", label = CI_label, alpha=0.5)
           
            leg = plt.legend(loc="upper right")
            leg.legendHandles[0]._sizes = [30]
            leg.legendHandles[1]._sizes = [30]

            plt_name = sample.replace("1000", "")+"_MllVsMet_2j.pdf"
            plt.savefig("plots/"+plt_name)
            plt.clf()



           

