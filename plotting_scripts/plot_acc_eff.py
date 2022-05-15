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
        "dimuon_mass",
        "pt_pass",
        "nJets",
        "nJets_accepted",
        "nJets_ID",
        "nJets_btag",
        "wgt_nominal",
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

def df2hist(var, df, bins, masscut, iscut=False, flavor="mu", iswgt=True, scale=True):
    cut = 120
    if var=="met":
        cut = 400
    df = df.compute()
    if flavor == "mu":
        df_acc = df[df["accepted"]]
        df_reco = df_acc[df_acc["reco"]]
        df_ID = df_reco[df_reco["ID_pass"]]
        df_trig = df_ID[df_ID["hlt"]>0]
        dfs = [df, df_acc, df_reco, df_ID, df_trig, ]
        vals_list = [] 
        errs_list = []

        for df_ in dfs:
            if var == "dimuon_mass":
                df_ = df_[df_["pt_pass"]]
                df_.drop_duplicates(subset=["dimuon_mass"], inplace=True)
             
            var_array = df_.loc[df_["dimuon_mass"] > cut, var]
            
            if iswgt:

                wgt = df_.loc[(df_["dimuon_mass"] > cut), "wgt_nominal"]
                wgt[wgt < 0] = 0
            if iscut:
                genmass = df_.loc[(df_["dimuon_mass"] > cut), "dimuon_mass"]
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
    else:

        df.drop_duplicates(subset=["dimuon_mass"], inplace=True)
        var_array = df.loc[df["dimuon_mass"] > cut, var]

        if iswgt:

            wgt = df.loc[(df["dimuon_mass"] > cut), "wgt_nominal"]
            nJets = df.loc[(df["dimuon_mass"] > cut), "nJets"]
            nJets_acc = df.loc[(df["dimuon_mass"] > cut), "nJets_accepted"]
            nJets_ID = df.loc[(df["dimuon_mass"] > cut), "nJets_ID"]
            nJets_btag = df.loc[(df["dimuon_mass"] > cut), "nJets_btag"]
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
               
            
                  
    return [vals_list, errs_list, bins]


def plotAcc(dfs, bins, name):

    cut = 120

    df_deno = {"0j":[],"1j":[],"2j":[]}
    df_no = {"0j":[],"1j":[],"2j":[]}

    for df in dfs:
        df = df.compute()
        df.drop_duplicates(subset=["dimuon_mass"], inplace=True)
        #df_deno["0j"].append(df[df['nJets']==0])
        #df_deno["1j"].append(df[df['nJets']==1])
        #df_deno["2j"].append(df[df['nJets']>=2])

        df_deno["0j"].append(df)
        df_deno["1j"].append(df)
        df_deno["2j"].append(df)
        df_no["0j"].append(df[(df['nJets']==0)&(df["accepted"])])
        df_no["1j"].append(df[(df['nJets']==1)&(df["accepted"])])
        df_no["2j"].append(df[(df['nJets']>=2)&(df["accepted"])])
        
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
            wgt1 = df_deno[key][i].loc[(df_deno[key][i]["dimuon_mass"] > cut), "wgt_nominal"].values
            array1 = df_deno[key][i].loc[(df_deno[key][i]["dimuon_mass"] > cut), "dimuon_mass"].values
            vals1 += np.histogram(array1, bins=bins, weights=wgt1)[0]
            vals1_err += np.histogram(array1, bins=bins, weights=wgt1**2)[0]
            wgt2 = df_no[key][i].loc[(df_no[key][i]["dimuon_mass"] > cut), "wgt_nominal"].values
            array2 = df_no[key][i].loc[(df_no[key][i]["dimuon_mass"] > cut), "dimuon_mass"].values
            vals2 += np.histogram(array2, bins=bins, weights=wgt2)[0]
            vals2_err += np.histogram(array2, bins=bins, weights=wgt2**2)[0]

        deno[key] = vals1
        no[key] = vals2
        r = sum(no[key])/sum(deno[key]) 
        err = r*np.sqrt(sum(vals1_err)/sum(vals1)**2 + sum(vals2_err)/sum(vals2)**2)
        mean[key] = r
        uncer[key] = err
    axes = setFrame(
        "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
        "acceptance",
        ratio=False,
        signal=False,
        logx=True,
        logy=True,
        xRange=[400, 3000],
        yRange=[1e-3,10.],
        flavor="mu",
        year="2018",
        )
    
    hep.histplot(
        [no["0j"]/deno["0j"], no["1j"]/deno["1j"], no["2j"]/deno["2j"]],
        bins,
        ax=axes[0],
        color=["red", "green", "blue"],
        label=["0 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["0j"], ".3f"), format(uncer["0j"], ".2f")), "1 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["1j"], ".3f"), format(uncer["1j"], ".2f")) , "2 b-jet acceptance (acc = %s $\pm$ %s)"%(format(mean["2j"], ".3f"), format(uncer["2j"], ".2f"))],
    )
    axes[0].legend(loc=(0.3, 0.6), fontsize="x-small")
    axes[1].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/acc_{name}.pdf"
    )
    axes[1].clf()




def plotdf2hist2D(var, dfs, binx, biny,  name, ID, iswgt=True, scale=True):
    cut = 120
   
    df1 = dfs[0].compute()  
    df1_pass = df1[df1[ID]]
    df2 = dfs[1].compute()
    df2_pass = df2[df2[ID]]


    if iswgt:
        wgt1 = df1.loc[(df1["dimuon_mass"] > cut), "wgt_nominal"].values
        wgt2 = df2.loc[(df2["dimuon_mass"] > cut), "wgt_nominal"].values
        wgt1_ID = df1_pass.loc[(df1_pass["dimuon_mass"] > cut), "wgt_nominal"].values
        wgt2_ID = df2_pass.loc[(df2_pass["dimuon_mass"] > cut), "wgt_nominal"].values
    else:
        wgt1 = 1.
        wgt2 = 1.
     
    varx1 = df1.loc[(df1["dimuon_mass"] > cut), var[0]].values 
    vary1 = df1.loc[(df1["dimuon_mass"] > cut), var[1]].values
    varx1_ID = df1_pass.loc[(df1_pass["dimuon_mass"] > cut), var[0]].values
    vary1_ID = df1_pass.loc[(df1_pass["dimuon_mass"] > cut), var[1]].values

    varx2 = df2.loc[(df2["dimuon_mass"] > cut), var[0]].values   
    vary2 = df2.loc[(df2["dimuon_mass"] > cut), var[1]].values  
    varx2_ID = df2_pass.loc[(df2_pass["dimuon_mass"] > cut), var[0]].values
    vary2_ID = df2_pass.loc[(df2_pass["dimuon_mass"] > cut), var[1]].values
    
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
    fig.savefig(f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}_{ID}_eff.pdf")
    fig.clf()
def plots(axes, MCs, labels, colors, name, flavor="mu"):

    bins = MCs[2]
    MCs_vals = []
    if flavor == "mu": 
        idx = 5
    else:
        idx = 4
     
    for i in range(1, idx):
        MC_deno = np.array(MCs[0][i-1])    
        MC_no = np.array(MCs[0][i])
        ratio = MC_no/MC_deno
        MCs_vals.append(ratio)

    
    MC_deno = np.array(MCs[0][0])
    MC_no = np.array(MCs[0][idx-1])
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
        files = glob.glob(path + "CI_effv2/" + sig_name + "/*")
        path_CI[sig_name] = files

    client = Client(LocalCluster(**client_args))
    mass_inclu = {}
    mass_Jet = {} 
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
            "dimuon_mass",
            df,
            bins_mass,
            masscut=masscut,
            flavor="mu",
            iscut=iscut,
            iswgt=iswgt,
        )

        #mass_0[key] = df2hist(
        #    "dimuon_mass",
        #    df,
        #    bins_mass,
        #    masscut=masscut,
        #    flavor="mu",
        #    iscut=iscut,
        #    iswgt=iswgt,
        #)


        mass_Jet[key] = df2hist(
            "dimuon_mass",
            df,
            bins_mass,
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
        for i in range(4):
            mass_Jet[sample][0][i] += mass_Jet[sample.replace("400", "1000")][0][i]


        binx = [ 30,   35,   40,   45,   50,  60, 70 ,  90, 150, 250,500,1000]
        biny = [-2.4, -2.1, -1.6, -1.2, -1.04, -0.9, -0.3, -0.2,  0.2, 0.3, 0.9, 1.04, 1.2, 1.6, 2.1, 2.4] 
        #for ID in ["Jet_ID", "btag"]: 

        #    plotdf2hist2D(["Jet_pt", "Jet_eta"], [df_list[sample], df_list[sample.replace("400", "1000")]], binx, biny,  sample, ID=ID, iswgt=True, scale=True)
        plotAcc([df_list[sample], df_list[sample.replace("400", "1000")]], bins_mass, sample.replace("400", ""))     

        axes1 = setFrame(
            "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0.85,1.2],
            flavor="mu",
            year="2018",
            )

  
        colors1 = ["red", "green", "brown", "blue", "skyblue"]
        labels1 = ["geometrical acceptance", "muon reconstruction", "high-pT muon ID", "trigger", "total acceptance $\\times$ efficiency"]
        colors2 = ["red", "green", "blue", "skyblue"]
        labels2 = ["geometrical acceptance", "tight ID", "b-tagging", "total acceptance $\\times$ efficiency"]
        
       
      
        plots(axes1, mass_inclu[sample], labels1, colors1, sample.replace("400", "")+"_acceff_inclu")
        axes2 = setFrame(
            "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
            "efficiency",
            ratio=False,
            signal=False,
            logx=True,
            logy=False,
            xRange=[400, 3000],
            yRange=[0,1.6],
            flavor="mu",
            year="2018",
            )
        
        plots(axes2, mass_Jet[sample], labels2, colors2, sample.replace("400", "")+"_acceff_bjet_inclu", "Jet")

        





