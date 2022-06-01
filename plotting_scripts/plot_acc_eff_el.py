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
import pandas as pd
from functools import reduce

def load2df(files):

    df = dd.read_parquet(files)
    field = [
        "dielectron_mass",
        "pt_pass",
        "nJets",
        "nJets_accepted",
        "nJets_ID",
        "nJets_btag",
        "pu_wgt",
        "pt",
        "eta",
        "match",
        "ID",
        "hlt",
        "accepted",
        "reco",
        "ID_pass",
        "Jet_pt",
        "Jet_eta",
        "Jet_ID",
        "btag",
    ]
    out = df[field]
    #out.compute()
    
    return out


def chunk(files, size):

    size = math.ceil(len(files) / float(size))
    file_bag = [
        files[i : min(i + size, len(files))] for i in range(0, len(files), size)
    ]
    return file_bag

def df2hist(var, df, bins, masscut, iscut=False, flavor="el", iswgt=True, scale=True, nj=-1):
    cut = 120
    if var=="met":
        cut = 400
    df = df.compute()
    df_local = df.copy()
    if nj!=-1 and nj!=2:
        df_local = df_local[df_local["nJets"]==nj]
    elif nj == 2:
        df_local = df_local[df_local["nJets"]>=nj]
    if flavor == "el":
        df_acc = df_local[df_local["accepted"]]
        df_reco = df_acc[df_acc["reco"]]
        df_ID = df_reco[df_reco["ID_pass"]]
        df_trig = df_ID[df_ID["hlt"]>0]
        dfs = [df_local, df_acc, df_reco, df_ID, df_trig]
        vals_list = [] 
        errs_list = []

        for df_ in dfs:
            if var == "dielectron_mass":
                df_ = df_[df_["pt_pass"]]
                df_.drop_duplicates(subset=["dielectron_mass"], inplace=True)
             
            var_array = df_.loc[df_["dielectron_mass"] > cut, var]
            
            if iswgt:

                wgt = df_.loc[(df_["dielectron_mass"] > cut), "pu_wgt"]
                wgt[wgt < 0] = 0
            if iscut:
                genmass = df_.loc[(df_["dielectron_mass"] > cut), "dielectron_mass"]
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

            vals_list.append(vals)
            errs_list.append(errs)
    elif flavor=="Jet":

        df.drop_duplicates(subset=["dielectron_mass"], inplace=True)
        var_array = df.loc[df["dielectron_mass"] > cut, var]

        if iswgt:

            wgt = df.loc[(df["dielectron_mass"] > cut), "pu_wgt"]
            nJets = df.loc[(df["dielectron_mass"] > cut), "nJets"]
            nJets_acc = df.loc[(df["dielectron_mass"] > cut), "nJets_accepted"]
            nJets_ID = df.loc[(df["dielectron_mass"] > cut), "nJets_ID"]
            nJets_btag = df.loc[(df["dielectron_mass"] > cut), "nJets_btag"]
            wgt[wgt < 0] = 0
            wgtn = wgt * nJets
            wgtn_acc = wgt * nJets_acc
            wgtn_ID = wgt * nJets_ID
            wgtn_btag = wgt * nJets_btag

            vals, bins = np.histogram(var_array, bins=bins, weights=wgtn)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgtn ** 2)
            errs = np.sqrt(vals2)
            
            vals_acc, bins = np.histogram(var_array, bins=bins, weights=wgtn_acc)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgtn_acc ** 2)
            errs_acc = np.sqrt(vals2)
    
            vals_ID, bins = np.histogram(var_array, bins=bins, weights=wgtn_ID)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgtn_ID ** 2)
            errs_ID = np.sqrt(vals2)
    
            vals_btag, bins = np.histogram(var_array, bins=bins, weights=wgtn_btag)
            vals2, bins = np.histogram(var_array, bins=bins, weights=wgtn_btag ** 2)
            errs_btag = np.sqrt(vals2)
            vals_list = [vals, vals_acc, vals_ID, vals_btag]
            errs_list = [errs, errs_acc, errs_ID, errs_btag]

            if scale:
                binSize = np.diff(bins)
                for i in range(len(vals_list)):
                    vals_list[i] = vals_list[i] / binSize
                    errs_list[i] = errs_list[i] / binSize

           
    else:
        #df_local = df_local[df_local["accepted"]]
        dfs = [df_local]
        for i in range(nj+1):
            if i==2:  
                dfs.append(df_local[(df_local["nJets_accepted"]>=i)&(df_local["accepted"])])
            else:
                dfs.append(df_local[(df_local["nJets_accepted"]==i)&(df_local["accepted"])])
        vals_list = []
        errs_list = []

        for df_ in dfs:
            if var == "dielectron_mass":
                df_ = df_[df_["pt_pass"]]
                df_.drop_duplicates(subset=["dielectron_mass"], inplace=True)

            var_array = df_.loc[df_["dielectron_mass"] > cut, var]

            if iswgt:

                wgt = df_.loc[(df_["dielectron_mass"] > cut), "pu_wgt"]
                wgt[wgt < 0] = 0
            if iscut:
                genmass = df_.loc[(df_["dielectron_mass"] > cut), "dielectron_mass"]
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

            vals_list.append(vals)
            errs_list.append(errs)    
            
                  
    return [vals_list, errs_list, bins]


def plotAcc(dfs, bins, name):

    cut = 120
    Yield = {}
    df_deno = {"0j":[],"1j":[],"2j":[]}
    df_no = {"0j":[],"1j":[],"2j":[]}
    df_mig = {}
    df_mig_acc = {}
    for df in dfs:
        df = df.compute()
        df.drop_duplicates(subset=["dielectron_mass"], inplace=True)
        #df_deno["0j"].append(df[df['nJets']==0])
        #df_deno["1j"].append(df[df['nJets']==1])
        #df_deno["2j"].append(df[df['nJets']>=2])

        df_deno["0j"].append(df)
        df_deno["1j"].append(df)
        df_deno["2j"].append(df)
        df_no["0j"].append(df[df['nJets']==0])
        df_no["1j"].append(df[df['nJets']==1])
        df_no["2j"].append(df[df['nJets']>=2])
        #df_no["0j"].append(df[(df['nJets_accepted']==0)&(df["accepted"])])
        #df_no["1j"].append(df[(df['nJets_accepted']==1)&(df["accepted"])])
        #df_no["2j"].append(df[(df['nJets_accepted']>=2)&(df["accepted"])])
        try:
            Yield["0j"] += np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]==0), "pu_wgt"].values)
            Yield["1j"] += np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]==1), "pu_wgt"].values)
            Yield["2j"] += np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]>=2), "pu_wgt"].values)
        except Exception:
            Yield["0j"] = np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]==0), "pu_wgt"].values)
            Yield["1j"] = np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]==1), "pu_wgt"].values)
            Yield["2j"] = np.sum(df.loc[(df["dielectron_mass"] > cut)&(df["nJets"]>=2), "pu_wgt"].values)
        df1 = df[df["accepted"]]
        df1 = df1[df1["reco"]]
        df1 = df1[df1["ID_pass"]]
        df1 = df1[df1["hlt"]>0]

        for i in range(3):
            for j in range(3):
                try:
                    if j <2 and i<2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']==j)&(df1["nJets"]==i)]) 
                    elif i==2 and j<2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']==j)&(df1["nJets"]>=i)])
                    elif i<2 and j==2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']>=j)&(df1["nJets"]==i)])
                    else:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']>=j)&(df1["nJets"]>=i)])
                except Exception:
                    df_mig[str(i)+"g_a"+str(j)] = []
                    if j <2 and i<2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']==j)&(df1["nJets"]==i)])
                    elif i==2 and j<2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']==j)&(df1["nJets"]>=i)])
                    elif i<2 and j==2:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']>=j)&(df1["nJets"]==i)])
                    else:
                        df_mig[str(i)+"g_a"+str(j)].append(df1[(df1['nJets_btag']>=j)&(df1["nJets"]>=i)])


    no = {}
    deno = {}
    mean = {}
    uncer = {}
    for key in df_deno.keys():
        vals1 = np.zeros(len(bins)-1)
        vals1_err = np.zeros(len(bins)-1)
        vals2 = np.zeros(len(bins)-1)
        vals2_err = np.zeros(len(bins)-1)
        for i in range(2):
            wgt1 = df_deno[key][i].loc[(df_deno[key][i]["dielectron_mass"] > cut), "pu_wgt"].values
            array1 = df_deno[key][i].loc[(df_deno[key][i]["dielectron_mass"] > cut), "dielectron_mass"].values
            vals1 += np.histogram(array1, bins=bins, weights=wgt1)[0]
            vals1_err += np.histogram(array1, bins=bins, weights=wgt1**2)[0]
            wgt2 = df_no[key][i].loc[(df_no[key][i]["dielectron_mass"] > cut), "pu_wgt"].values
            array2 = df_no[key][i].loc[(df_no[key][i]["dielectron_mass"] > cut), "dielectron_mass"].values
            vals2 += np.histogram(array2, bins=bins, weights=wgt2)[0]
            vals2_err += np.histogram(array2, bins=bins, weights=wgt2**2)[0]

        deno[key] = vals1
        no[key] = vals2
        r = sum(no[key])/sum(deno[key]) 
        err = r*np.sqrt(sum(vals1_err)/sum(vals1)**2 + sum(vals2_err)/sum(vals2)**2)
        mean[key] = r
        uncer[key] = err
        mig = {}
        migM = []
        for i in range(3):
            migA = []
            for j in range(3):
                val = 0
                for k in range(2):
                   
                    val += np.sum(df_mig[str(i)+"g_a"+str(j)][k].loc[(df_mig[str(i)+"g_a"+str(j)][k]["dielectron_mass"] > cut), "pu_wgt"].values)
                mig[str(i)+"g_a"+str(j)]=val
                migA.append(val/Yield[str(i)+"j"])
            migM.append(migA)
        migM = np.array(migM)
    axes = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        #"acceptance",
        "fraction",
        ratio=False,
        signal=False,
        logx=True,
        logy=False,
        xRange=[400, 3000],
        yRange=[0, 1.5],
        flavor="el",
        year="2018",
        )
    
    hep.histplot(
        [no["0j"]/deno["0j"], no["1j"]/deno["1j"], no["2j"]/deno["2j"]],
        bins,
        ax=axes[0],
        color=["red", "green", "blue"],
        label=["0 b-jet fraction (%s $\pm$ %s)"%(format(mean["0j"], ".3f"), format(uncer["0j"], ".4f")), "1 b-jet fraction (%s $\pm$ %s)"%(format(mean["1j"], ".3f"), format(uncer["1j"], ".4f")) , "$\geq$ 2 b-jet fraction (%s $\pm$ %s)"%(format(mean["2j"], ".3f"), format(uncer["2j"], ".4f"))],
        #label=["0 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["0j"], ".3f"), format(uncer["0j"], ".4f")), "1 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["1j"], ".3f"), format(uncer["1j"], ".4f")) , "2 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["2j"], ".3f"), format(uncer["2j"], ".4f"))],
    )
        
    axes[0].legend(loc=(0.3, 0.6), fontsize="x-small")
    axes[1].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/jet_fraction_el_{name}.pdf"
    )
    axes[1].clf()
    h0j = np.array([mig["0g_a0"], mig["0g_a1"], mig["0g_a2"]])
    h1j = np.array([mig["1g_a0"], mig["1g_a1"], mig["1g_a2"]])
    h2j = np.array([mig["2g_a0"], mig["2g_a1"], mig["2g_a2"]])
    if name.find("4TeV")!=-1:
        y_hlim = 2000
    else:
        y_hlim = 300
    axes1 = setFrame(
        "",
        "Yield of events",
        ratio=False,
        signal=False,
        logx=False,
        logy=False,
        xRange=[0, 3],
        yRange=[0, y_hlim],
        flavor="el",
        year="2018",
        )
    
    hep.histplot(
        [h0j, h1j, h2j],
        [0, 1, 2, 3],
        ax=axes1[0],
        stack = True,
        histtype="fill",
        #binticks=True,
        color=["red", "green", "blue"],
        label=["0 gen b-jet", "1 gen b-jet" , "$\geq$ 2 gen b-jet"],
    )
    axes1[0].set_xticks([0.5,1.5,2.5])
    axes1[0].set_xticklabels(["0 reco b-jet", "1 reco b-jet"," $\geq$ 2 reco b-jet"])
    axes1[0].legend(loc=(0.45, 0.6), fontsize="x-small")
    axes1[1].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/jet_acc_eff_migration_el_{name}.pdf"
    )
    axes1[1].clf()
    
    
    fig, ax = plt.subplots()
    hep.hist2dplot(
    migM,
    xbins=[0, 1, 2, 3],
    ybins=[0, 1, 2, 3],
    #labels=None,
    #cbar=True,
    color = "r",
    cbarsize="7%",
    cbarpad=0.2,
    cbarpos="right",
    cbarextend=False,
    #cmin=None,
    #cmax=None,
    ax=ax,
    #**kwargs,
    )
    ax.set_xlabel("N gen b-jets")
    ax.set_ylabel("N reco b-jets")
    ax.set_xticks([0.5,1.5,2.5])
    ax.set_xticklabels(["0", "1"," $\geq$ 2"])
    ax.set_yticks([0.5,1.5,2.5])
    ax.set_yticklabels(["0", "1"," $\geq$ 2"])
    for i in range(3):
        for j in range(3):

            ax.text(i+0.5, j+0.5, format(migM[i,j], ".3f"), ha="center", va="center", color="black")

    fig.savefig(f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/jet_acc_eff_matrix_el_{name}.pdf")
    fig.clf()



def plotdf2hist2D(var, dfs, binx, biny,  name, ID, ID_deno="none",iswgt=True, scale=True):
    cut = 120
   
    df1 = dfs[0].compute()
    if ID_deno!="none":
        df1 = df1[df1[ID_deno]]  
    df1_pass = df1[df1[ID]]
    df2 = dfs[1].compute()
    if ID_deno!="none":
        df2 = df2[df2[ID_deno]]
    df2_pass = df2[df2[ID]]


    if iswgt:

        wgt1 = df1.loc[(df1["dielectron_mass"] > cut), "pu_wgt"].values
        wgt2 = df2.loc[(df2["dielectron_mass"] > cut), "pu_wgt"].values
        wgt1_ID = df1_pass.loc[(df1_pass["dielectron_mass"] > cut), "pu_wgt"].values
        wgt2_ID = df2_pass.loc[(df2_pass["dielectron_mass"] > cut), "pu_wgt"].values
    else:
        wgt1 = 1.
        wgt2 = 1.
     
    varx1 = df1.loc[(df1["dielectron_mass"] > cut), var[0]].values 
    vary1 = df1.loc[(df1["dielectron_mass"] > cut), var[1]].values
    varx1_ID = df1_pass.loc[(df1_pass["dielectron_mass"] > cut), var[0]].values
    vary1_ID = df1_pass.loc[(df1_pass["dielectron_mass"] > cut), var[1]].values

    varx2 = df2.loc[(df2["dielectron_mass"] > cut), var[0]].values   
    vary2 = df2.loc[(df2["dielectron_mass"] > cut), var[1]].values  
    varx2_ID = df2_pass.loc[(df2_pass["dielectron_mass"] > cut), var[0]].values
    vary2_ID = df2_pass.loc[(df2_pass["dielectron_mass"] > cut), var[1]].values
    
    vals1 = np.histogram2d(varx1, vary1, bins=(binx, biny), weights=wgt1)[0]
    vals1_ID = np.histogram2d(varx1_ID, vary1_ID, bins=(binx, biny), weights=wgt1_ID)[0]
    vals2 = np.histogram2d(varx2, vary2, bins=(binx, biny), weights=wgt2)[0]
    vals2_ID = np.histogram2d(varx2_ID, vary2_ID, bins=(binx, biny), weights=wgt2_ID)[0]
    vals = vals1 + vals2
    vals_ID = vals1_ID + vals2_ID
    ratio = vals_ID/vals
    fig, ax = plt.subplots()
    hep.hist2dplot(
    ratio,
    xbins=binx,
    ybins=biny,
    #labels=None,
    #cbar=True,
    color = "r",
    cbarsize="7%",
    cbarpad=0.2,
    cbarpos="right",
    cbarextend=False,
    #cmin=None,
    #cmax=None,
    ax=ax,
    #**kwargs,
    ) 
    ax.set_xlabel("pt [GeV]")
    ax.set_ylabel("$\eta$")
    ax.set_xscale("log")
    fig.savefig(f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}_{ID}_frac_el_eff.pdf")
    fig.clf()
def plots(axes, MCs, labels, colors, name, flavor="el"):

    bins = MCs[2]
    MCs_vals = []
    if flavor == "el": 
        idx = 5
    elif flavor == "Jet":
        idx = 4
    elif flavor == "acc2":
        idx = 4
    else:
        idx = 3
    if flavor.find("acc") == -1: 
        for i in range(1, idx):
            MC_deno = np.array(MCs[0][i-1])    
            MC_no = np.array(MCs[0][i])
            ratio = MC_no/MC_deno
            MCs_vals.append(ratio)

        MC_deno = np.array(MCs[0][0])
        MC_no = np.array(MCs[0][idx-1])
        ratio = MC_no/MC_deno
        MCs_vals.append(ratio)
    
    else:
        for i in range(1, idx):
            MC_deno = np.array(MCs[0][0])
            MC_no = np.array(MCs[0][i])
            ratio = MC_no/MC_deno
            MCs_vals.append(ratio)

    
    hep.histplot(
        MCs_vals,
        bins,
        ax=axes[0],
        color=colors,
        label=labels,
    )


    axes[0].legend(loc=(0.4, 0.6), fontsize="x-small")
    #axes[1].legend(loc=(0.55, 0.55), fontsize="x-small")
    axes[1].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}.pdf"
    )
    axes[1].clf()

def plotVar(axes, MCs, name):

    bins = MCs[2]
    MCs_vals = np.array(MCs[0][0])


    hep.histplot(
        MCs_vals,
        bins,
        ax=axes[0],
        color="red",
    )
    #axes[0].legend(loc=(0.4, 0.6), fontsize="x-small")
    #axes[1].legend(loc=(0.55, 0.55), fontsize="x-small")
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
        [j for j in range(400, 600, 100)]
        + [j for j in range(600, 1200, 150)]
        + [j for j in range(1200, 1800, 300)]
        + [j for j in range(1800, 3001, 600)]
        #+ [j for j in range(1890, 3001, 200)]
        )
  
    path = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/"

    path_tt_inclusive = path + "ttbar_test/ttbar_lep/*.parquet"
    tt_inclusive_files = glob.glob(path_tt_inclusive)
    path_tt = path + "ttbar_test/ttbar_lep_*/*.parquet"
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
    df_list = {}
    for sig_name in CI_list:
        files = glob.glob(path + "CI_eff_el/" + sig_name + "/*")
        path_CI[sig_name] = files

    client = Client(LocalCluster(**client_args))
    mass_inclu = {}
    mass_0 = {}
    mass_1 = {}
    mass_2 = {}
    mass_Jet = {} 
    pt_Jet = {}
    eta_Jet = {}
    mass_acc2 = {}
    mass_acc1 = {}
    Jet_pt_bin = [10 ,15, 20, 25, 30, 35, 40, 45, 50, 60, 70 , 90, 150, 250, 500, 1000]
    Jet_eta_bin = np.linspace(-4., 4., 21)
    for key in path_CI.keys():
        
        masscut = 0
        iscut = False
        file_bag = chunk(path_CI[key], client_args["n_workers"])
        if len(file_bag) == 1:
            df = load2df_mc(file_bag)
        else:
            results = client.map(load2df, file_bag)

            dfs = client.gather(results)
            df = dd.concat(dfs)

        df_list[key] = df
        iswgt = True
       
        mass_inclu[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="el",
            iscut=iscut,
            iswgt=iswgt,
        )

        mass_0[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="el",
            iscut=iscut,
            iswgt=iswgt,
            nj = 0,
        )
        mass_1[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="el",
            iscut=iscut,
            iswgt=iswgt,
            nj = 1,
        )
        mass_2[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="el",
            iscut=iscut,
            iswgt=iswgt,
            nj = 2,
        )
        mass_acc2[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="acc",
            iscut=iscut,
            iswgt=iswgt,
            nj = 2,
        )
        mass_acc1[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="acc",
            iscut=iscut,
            iswgt=iswgt,
            nj = 1,
        )


        mass_Jet[key] = df2hist(
            "dielectron_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="Jet",
            iscut=iscut,
            iswgt=iswgt,
        )

        pt_Jet[key] = df2hist(
            "Jet_pt",
            df,
            Jet_pt_bin,
            masscut=masscut,
            flavor="Jet",
            iscut=iscut,
            iswgt=iswgt,
        )
        
        eta_Jet[key] = df2hist(
            "Jet_eta",
            df,
            Jet_eta_bin,
            masscut=masscut,
            flavor="Jet",
            iscut=iscut,
            iswgt=iswgt,
        )



    client.close()
    for sample in CI_list:
        if "1000" in sample: 
            continue

        for i in range(5):
            mass_inclu[sample][0][i] += mass_inclu[sample.replace("400", "1000")][0][i]
            mass_0[sample][0][i] += mass_0[sample.replace("400", "1000")][0][i]
            mass_1[sample][0][i] += mass_1[sample.replace("400", "1000")][0][i]
            mass_2[sample][0][i] += mass_2[sample.replace("400", "1000")][0][i]
        for i in range(3):
            mass_acc1[sample][0][i] += mass_acc1[sample.replace("400", "1000")][0][i]
        for i in range(4):
            mass_acc2[sample][0][i] += mass_acc2[sample.replace("400", "1000")][0][i]
        for i in range(4):
            mass_Jet[sample][0][i] += mass_Jet[sample.replace("400", "1000")][0][i]
            pt_Jet[sample][0][i] += pt_Jet[sample.replace("400", "1000")][0][i]   
            eta_Jet[sample][0][i] += eta_Jet[sample.replace("400", "1000")][0][i]

        binx = [ 30,   35,   40,   45,   50,  60, 70 ,  90, 150, 250,500,1000]
        biny = [-2.4, -2.1, -1.6, -1.2, -1.04, -0.9, -0.3, -0.2,  0.2, 0.3, 0.9, 1.04, 1.2, 1.6, 2.1, 2.4]
        IDs = ["none", "Jet_ID", "btag"] 
        for i in range(2): 

            plotdf2hist2D(["Jet_pt", "Jet_eta"], [df_list[sample], df_list[sample.replace("400", "1000")]], binx, biny,  sample, ID=IDs[i+1], ID_deno=IDs[i], iswgt=True, scale=True)
        plotAcc([df_list[sample], df_list[sample.replace("400", "1000")]], bins_mass, sample.replace("400", ""))     


        colors1 = ["red", "green", "brown", "blue", "skyblue"]
        labels1 = ["dielectron acceptance", "electron reconstruction", "high-pT electron ID", "trigger", "dielectron acceptance $\\times$ efficiency"]
        
        colors2 = ["red", "green", "blue", "skyblue"]
        labels2 = ["geometrical acceptance", "tight ID", "b-tagging", "total acceptance $\\times$ efficiency"]
        
        labels_acc = ["0 b-jet within acceptance", "1 b-jet within acceptance", "$\geq$ 2 b-jet within acceptance"]
        colors_acc = ["red", "green", "blue"]
        axes1 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.4],
            flavor="el",
            year="2018",
            )
        plots(axes1, mass_inclu[sample], labels1, colors1, sample.replace("400", "")+"_acceff_el_inclu")

        axes2 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.4],
            flavor="el",
            year="2018",
            )
        plots(axes2, mass_0[sample], labels1, colors1, sample.replace("400", "")+"_acceff_el_0j")

        axes3 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.4],
            flavor="el",
            year="2018",
            )
        plots(axes3, mass_1[sample], labels1, colors1, sample.replace("400", "")+"_acceff_el_1j")

        axes4 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.4],
            flavor="el",
            year="2018",
            )
        plots(axes4, mass_2[sample], labels1, colors1, sample.replace("400", "")+"_acceff_el_2j")
         
        axes5 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "acceptance",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.0],
            flavor="el",
            year="2018",
            )
        plots(axes5, mass_acc2[sample], labels_acc, colors_acc, sample.replace("400", "")+"_acc_el_2j", "acc2")

        axes6 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "acceptance",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.0],
            flavor="el",
            year="2018",
            )
        plots(axes6, mass_acc1[sample], labels_acc[:-1], colors_acc[:-1], sample.replace("400", "")+"_acc_el_1j", "acc1")
        
        axes7 = setFrame(
            "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.6],
            flavor="el",
            year="2018",
            )
        
        plots(axes7, mass_Jet[sample], labels2, colors2, sample.replace("400", "")+"_acceff_bjet_el_inclu", "Jet")

        axes8 = setFrame(
            "$p_{\mathrm{T}}$ [GeV]",
            "Yield",
            ratio=False,
            signal=False,
            logx=True,
            logy=True,
            xRange=[10, 1000],
            yRange=[1e-3,1e3],
            flavor="el",
            year="2018",
            )
        axes8[0].set_xticks([10., 20., 30., 50., 100., 200., 500., 1000.])
        plotVar(axes8, pt_Jet[sample], sample.replace("400", "")+"_bjet_el_pt")

        if "4TeV" in sample: ylim = 500
        else: ylim = 15
        axes9 = setFrame(
            "$\eta$",
            "Yield",
            ratio=False,
            signal=False,
            logx=False,
            logy=False,
            xRange=[-4., 4.],
            yRange=[0,ylim],
            flavor="el",
            year="2018",
            )
        axes9[0].set_xticks([-3.6, -2.4, -1.2, 0., 1.2, 2.4, 3.6])
        plotVar(axes9, eta_Jet[sample], sample.replace("400", "")+"_bjet_el_eta")
        




