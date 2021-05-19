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
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator
from __future__ import annotations
import matplotlib as mpl


class frame:

        def __init__(self, xtitle, xRange, ytitle, yRange, bins, category, color, year, islogx, islogy, isMC, isFill):
                #self.title = title
                self.xtitle = xtitle
                self.ytitle = ytitle
                self.xRange = xRange
                self.yRange = yRange
                self.bins = bins
                self.islogx = islogx
                self.islogy = islogy
                self.category = category
                self.isMC = isMC
                self.color = color
                self.isFill = isFill 
                self.year = year
#cfg
path_DY="/depot/cms/users/minxi/NanoAOD_study/Zprime-mumu/output/DY/*/*.parquet"
path_data="/depot/cms/users/minxi/NanoAOD_study/Zprime-mumu/output/data/*/*.parquet"
path_save="/depot/cms/users/minxi/NanoAOD_study/Zprime-mumu/plots/"
variables_plot = 'dimuon_mass'
frames={}
bins = [j for j in range(50, 120, 5)] + [j for j in range(120, 150, 5)] + [j for j in range(150, 200, 10)] + [j for j in range(200, 600, 20)] + [j for j in range(600, 900, 30) ] + [j for j in range(900, 1250, 50)] + [j for j in range(1250, 1610, 60) ] + [j for j in range(1610, 1890, 70) ] + [j for j in range(1890, 3970, 80) ] + [j for j in range(3970, 6070, 100) ] + [6070]
frames['dimuon_mass'] = frame('m($\mu^{+}\mu^{-}$) [GeV]', [120, 4000], 'Events/GeV', [1e-5, 1e7], np.array(bins), "Combine", "skyblue", "2018", True, True, True, True)
#frames['dimuon_cos_theta_cs'] = frame('cos $\\theta$', [-1, 1], 'Events', [0, 50000], np.linspace(-1,1,25), "Combine", "cyan", "2018", False, False, True, True)
lumis={'2016':36.3, '2017':42.1 ,'2018':61.6}
np.linspace(-1, 1, 25)
# Zprime->ll mass plot style
Zprime2ll_mass = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "mathtext.fontset": "custom",
    "mathtext.rm": "TeX Gyre Heros",
    "mathtext.bf": "TeX Gyre Heros:bold",
    "mathtext.sf": "TeX Gyre Heros",
    "mathtext.it": "TeX Gyre Heros:italic",
    "mathtext.tt": "TeX Gyre Heros",
    "mathtext.cal": "TeX Gyre Heros",
    "mathtext.default": "regular",
    "figure.figsize": (10.0, 10.0),
    "font.size": 26,
    "axes.labelsize": "medium",
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{siunitx},\sisetup{detect-all}, \
                              \usepackage{helvet},\usepackage{sansmath}, \
                              \sansmath",

}

# Filter extra (labellocation) items if needed
Zprime2ll_mass = {k: v for k, v in CMS.items() if k in mpl.rcParams}












#hep.set_style(hep.style.CMS)
#plt.style.use([hep.style.CMS])
plt.style.use([Zprime2ll_mass])
# def function

def plot(MC, data, variable, path_save):
        fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': [5, 1]})
        frame = frames[variable]
        axs[1].set_xlabel(frame.xtitle)
        axs[0].set_ylabel(frame.ytitle)
        #axs[0].yaxis.grid(True, which='minor')
        axs[1].set_ylabel("(Data-Bkg)/Bkg", fontsize=16)
        axs[1].set_xlim(frame.xRange)
        axs[0].set_ylim(frame.yRange)
        axs[1].set_ylim([-1,1]) 
        if frame.islogx:
                axs[0].set_xscale('log')
        if frame.islogy:
                axs[0].set_yscale('log')
        print(variable)
        htype="step"
        MC_mass = MC[variable].compute()
        data_mass = data[variable].compute()
        data_run = data['run'].compute()
        m=data_mass[data_mass>2000].values
        run=data_run[data_mass>2000].values
        s = np.argsort(m)
        s = s[::-1]
        for i in s:
            print("run %i with mass %f"%(run[i], m[i]))
        MC_entries = MC_mass.values
        data_entries = data_mass.values
        weight=(MC['wgt_nominal'].compute()).values
        MC_yvals, bins  = np.histogram(MC_entries, bins=frame.bins, weights=weight)
        data_yvals, bins  = np.histogram(data_entries, bins=frame.bins)
        if variable == "dimuon_mass":
                binSize = np.diff(bins)
                MC_yvals = MC_yvals/binSize
                data_yvals = data_yvals/binSize
                axs[1].set_xticks([200, 300, 1000, 2000])
                axs[0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
                axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)                   
                axs[0].get_yaxis().set_major_locator(locmaj)
                locmin = LogLocator(base=10.0, subs=np.arange(2,10)*0.1, numticks=100)
                axs[0].yaxis.set_minor_locator(locmin)
                axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                axs[1].hlines((-0.5, 0, 0.5), frame.xRange[0], frame.xRange[1], color="black", linestyles="dotted")
                #axs[1].hlines(y=0.)
                #axs[1].hlines(y=0.5)
                #axs[0].tick_params(which='both', width=2)
                #axs[0].tick_params(which='major', length=7)
                #axs[0].tick_params(which='minor', length=4)

                #axs[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                #axs[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

        if frame.isFill:
                htype="fill"

        #hep.histplot([data_yvals, MC_yvals], bins, ax=axs[0], color=['black', frame.color], histtype=['errorbar' ,htype],label=["Data" ,"$\gamma$ /Z$\\rightarrow \mu^{+}\mu^{-}$"])
        hep.histplot( MC_yvals, bins, ax=axs[0], color=frame.color, histtype=htype, label="$\gamma$ /Z$\\rightarrow \mu^{+}\mu^{-}$")
        hep.histplot(data_yvals, bins, ax=axs[0], color='black', histtype='errorbar',label="Data")
        rvals = (data_yvals-MC_yvals)/MC_yvals
        axs[0].legend(loc=(0.2,0.8))
        hep.histplot(rvals, bins, ax=axs[1], histtype='errorbar', color='black')
        lumi=lumis[frame.year]
        _lumi = r"{lumi} (13 TeV)".format(lumi=str(lumi) + r" $\mathrm{fb^{-1}}$")
        print(_lumi)
        axs[0].text(1.0, 1.08, _lumi, verticalalignment='top', horizontalalignment='right', transform=axs[0].transAxes, fontsize=28)
        axs[0].text(0.88, 0.95, "CMS", verticalalignment='top', horizontalalignment='right', transform=axs[0].transAxes, fontsize=37, weight='bold')
        #hep.CMS.label(lumi=lumi, exp="")
        
        #hep.cms.text()
        fig.savefig(path_save+variable+"_nocms.pdf")

#load data 
DY_paths = glob.glob(path_DY)
with ProcessPoolExecutor(max_workers=48) as executor:
        DY_dfs = list(executor.map(dd.read_parquet, DY_paths))
DY_df=dd.concat(DY_dfs)

data_paths = glob.glob(path_data)
with ProcessPoolExecutor(max_workers=48) as executor:
        data_dfs = list(executor.map(dd.read_parquet, data_paths))
data_df=dd.concat(data_dfs)
#print(data.head())
#plot
#with ProcessPoolExecutor(max_workers=48) as executor:
#	executor.map(plot, itertools.repeat(DY_df),  ,variables_plot, itertools.repeat(path_save))
plot(DY_df, data_df, variables_plot, path_save)






