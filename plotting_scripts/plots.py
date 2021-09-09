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
path_read="/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/2018_test_march_2021_9_8_0_28_8/*/*.parquet"
path_save="/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/"
#variables_plot = ['dimuon_mass', 'dimuon_cos_theta_cs']
variables_plot = ['jet1_pt', 'jet1_eta', 'jet1_phi','jet1_btagDeepB','jet2_pt', 'jet2_eta', 'jet2_phi','jet2_btagDeepB']
frames={}
#bins = [j for j in range(50, 120, 5)] + [j for j in range(120, 150, 5)] + [j for j in range(150, 200, 10)] + [j for j in range(200, 600, 20)] + [j for j in range(600, 900, 30) ] + [j for j in range(900, 1250, 50)] + [j for j in range(1250, 1610, 60) ] + [j for j in range(1610, 1890, 70) ] + [j for j in range(1890, 3970, 80) ] + [j for j in range(3970, 6070, 100) ] + [6070]
#frames['dimuon_mass'] = frame('m($\mu^{+}\mu^{-}$) [GeV]', [120, 4000], 'Events/GeV', [1e-5, 1e7], np.array(bins), "Combine", "skyblue", "2018", True, True, True, True)
#frames['dimuon_cos_theta_cs'] = frame('cos $\\theta$', [-1, 1], 'Events', [0, 50000], np.linspace(-1,1,25), "Combine", "cyan", "2018", False, False, True, True)
frames['pt'] = frame('pt', [0, 1000], 'Events', [1e-3, 15000], np.linspace(0,1000,1000), "Combine", "cyan", "2018", False, True, True, True)
frames['eta'] = frame('eta', [-3, 3], 'Events', [0, 10000], np.linspace(-3,3,100), "Combine", "cyan", "2018", False, False, True, True)
frames['phi'] = frame('phi', [-3.2, 3.2], 'Events', [0, 6000], np.linspace(-3.2,3.2,100), "Combine", "cyan", "2018", False, False, True, True)
frames['btagDeepB'] = frame('btagDeepB', [0, 1], 'Events', [1e-3, 50000], np.linspace(0,1,100), "Combine", "cyan", "2018", False, True, True, True)
lumis={'2016':36.3, '2017':42.1 ,'2018':61.6}
np.linspace(-1, 1, 25)


hep.set_style(hep.style.CMS)
plt.style.use([hep.style.CMS])
# def function

def plot(data,variable,path_save):
        fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios': [5, 1]})
        var=variable.split('_')[1]
        frame = frames[var]
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
        df=data[variable].compute()
        entries = df.values
        weight=(data['wgt_nominal'].compute()).values
        print(weight)
        yvals, bins  = np.histogram(entries, bins=frame.bins, weights=weight)
        print(yvals)
        if variable == "dimuon_mass":
                binSize = np.diff(bins)
                yvals = yvals/binSize
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

        hep.histplot(yvals, bins, ax=axs[0], color=frame.color, histtype=htype,label="$\gamma$ /Z$\\rightarrow \mu^{+}\mu^{-}$")
        rvals = np.zeros(len(yvals))
        axs[0].legend(loc=(0.2,0.8))
        hep.histplot(rvals, bins, ax=axs[1], histtype='errorbar', color='black')
        lumi=lumis[frame.year]
        _lumi = r"{lumi} (13 TeV)".format(lumi=str(lumi) + r" $\mathrm{fb^{-1}}$")
        print(_lumi)
        axs[0].text(1.0, 1.08, _lumi, verticalalignment='top', horizontalalignment='right', transform=axs[0].transAxes, fontsize=28)
        axs[0].text(0.88, 0.95, "CMS", verticalalignment='top', horizontalalignment='right', transform=axs[0].transAxes, fontsize=37, weight='bold')
        #hep.CMS.label(lumi=lumi, exp="")
        
        #hep.cms.text()
        fig.savefig(path_save+variable+"_noCMS.pdf")

#load data 
file_paths = glob.glob(path_read)
#print(file_paths)
#dd.read_parquet(file_paths[0],header=None,index_col=[0,1],columns=['jet1_pt', 'jet1_eta', 'jet1_phi','jet1_btagDeepB','jet2_pt', 'jet2_eta', 'jet2_phi','jet2_btagDeepB'])
executor =  ProcessPoolExecutor(max_workers=12) 
dfs = list(executor.map(dd.read_parquet, file_paths))
data=dd.concat(dfs)
#print(data.head())
#plot
with ProcessPoolExecutor(max_workers=12) as executor:
	executor.map(plot, itertools.repeat(data), variables_plot, itertools.repeat(path_save))







