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
import matplotlib as mpl
import matplotlib.style
import matplotlib.font_manager
import uproot
import coffea.hist


class frame:
    def __init__(
        self,
        xtitle,
        xRange,
        ytitle,
        yRange,
        bins,
        category,
        color,
        year,
        islogx,
        islogy,
        isMC,
        isFill,
    ):
        # self.title = title
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


# cfg
path_DY = (
    "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/DY_muv2/*/*.parquet"
)
other_path = (
    "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/other_v2/*/*/*.parquet"
)
path_data = (
    "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/pre-UL2018/*/*.parquet"
)
path_save = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/"
path_signal = "/depot/cms/users/minxi/NanoAOD_study/dileptonmassplots/inputs/paperPlotInputs_DimuonMass_Run2.root"
variables_plot = "dimuon_mass"
frames = {}

bins = [
    120.0,
    129.95474058,
    140.73528833,
    152.41014904,
    165.0535115,
    178.74571891,
    193.57377942,
    209.63191906,
    227.02218049,
    245.85507143,
    266.2502669,
    288.3373697,
    312.25673399,
    338.16035716,
    366.21284574,
    396.59246138,
    429.49225362,
    465.12128666,
    503.70596789,
    545.49148654,
    590.74337185,
    639.74918031,
    692.82032303,
    750.29404456,
    812.53556599,
    879.94040575,
    952.93689296,
    1031.98888927,
    1117.59873655,
    1210.310449,
    1310.71317017,
    1419.4449167,
    1537.19663264,
    1664.71658012,
    1802.81509423,
    1952.36973236,
    2114.3308507,
    2289.72764334,
    2479.6746824,
    2685.37900061,
    2908.14776151,
    3149.39656595,
    3410.65844758,
    3693.59361467,
    4000.0,
    4500,
    5200,
    6000,
    7000,
    8000,
]


frames["dimuon_mass"] = frame(
    "$\mathrm{m}(\mu^{+}\mu^{-})$ [GeV]",
    [120, 6000],
    "Events/GeV",
    [1e-5, 1e7],
    np.array(bins),
    "Combine",
    "skyblue",
    "2018",
    True,
    True,
    True,
    True,
)
lumis = {"2016": 36.3, "2017": 42.1, "2018": 61.6}

# Zprime->ll mass plot style
style = hep.style.CMS
style["mathtext.fontset"] = "cm"
style["mathtext.default"] = "rm"
plt.style.use(style)
hep.set_style(style)

# def function


def plot(DY, other, ttbar, data, Zprime, G_RS, variable, path_save):
    njets = 2
    fig, axs = plt.subplots(
        2,
        sharex=True,
        sharey=False,
        gridspec_kw={
            "height_ratios": [4, 1],
        },
    )
    plt.subplots_adjust(hspace=0.07)
    frame = frames[variable]
    axs[1].set_xlabel(frame.xtitle)
    axs[0].set_ylabel(frame.ytitle)
    axs[1].set_ylabel("(Data-Bkg)/Bkg", fontsize=16)
    axs[1].set_xlim(frame.xRange)
    axs[0].set_ylim(frame.yRange)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[1].set_ylim([-1, 1])
    axs[1].set_xticks([200, 300, 1000, 2000])
    axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
    locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    axs[1].hlines(
        (-0.5, 0, 0.5),
        frame.xRange[0],
        frame.xRange[1],
        color="black",
        linestyles="dotted",
    )
    ax_signal = axs[0].twiny()
    ax_signal.set_xscale("log")
    ax_signal.set_yscale("log")
    ax_signal.set_ylim(frame.yRange)
    ax_signal.set_xlim(frame.xRange)
    ax_signal.set_yticks(
        [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
    )
    ax_signal.set_xticks([])
    locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
    ax_signal.get_yaxis().set_major_locator(locmaj)
    ax_signal.yaxis.set_minor_locator(locmin)
    ax_signal.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    # axs[0].set_xticks([])
    # locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
    axs[0].get_yaxis().set_major_locator(locmaj)
    axs[0].yaxis.set_minor_locator(locmin)
    axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    DY_mass = DY.loc[DY["njets"] == njets, variable].compute()
    # DY_n = DY["njets"].compute()
    # DY_mass=DY_mass[DY_n==njets]
    other_mass = other.loc[other["njets"] == njets, variable].compute()
    ttbar_mass = ttbar.loc[ttbar["njets"] == njets, variable].compute()
    # ttbar_n = ttbar["njets"].compute()
    # ttbar_mass=ttbar_mass[ttbar_n==njets]
    data_mass = data.loc[data["njets"] == njets, variable].compute()
    # data_n = data["njets"].compute()
    # data_mass=data_mass[data_n==njets]
    data_run = data["run"].compute()
    m = data_mass[data_mass > 2000].values
    # run=data_run[data_mass>2000].values
    # s = np.argsort(m)
    # s = s[::-1]
    # for i in s:
    #    print("run %i with mass %f"%(run[i], m[i]))

    DY_tree = DY_mass.values
    other_tree = other_mass.values
    ttbar_tree = ttbar_mass.values
    # ttbar_tree = ttbar_tree[ttbar_tree<500]
    data_tree = data_mass.values
    DY_weight = (DY.loc[DY["njets"] == njets, "wgt_nominal"].compute()).values
    DY_weight[DY_weight < 0] = 0
    DY_vals, bins = np.histogram(DY_tree, bins=frame.bins, weights=DY_weight)
    DY_vals2, bins = np.histogram(DY_tree, bins=frame.bins, weights=DY_weight ** 2)
    DY_errs = np.sqrt(DY_vals2)

    other_weight = (other.loc[other["njets"] == njets, "wgt_nominal"].compute()).values
    other_weight[other_weight < 0] = 0
    other_vals, bins = np.histogram(other_tree, bins=frame.bins, weights=other_weight)
    other_vals2, bins = np.histogram(
        other_tree, bins=frame.bins, weights=other_weight ** 2
    )
    other_errs = np.sqrt(other_vals2)

    ttbar_weight = (ttbar.loc[ttbar["njets"] == njets, "wgt_nominal"].compute()).values
    ttbar_weight = ttbar_weight[ttbar_tree < 500]
    ttbar_tree = ttbar_tree[ttbar_tree < 500]
    ttbar_vals, bins = np.histogram(ttbar_tree, bins=frame.bins, weights=ttbar_weight)
    ttbar_vals2, bins = np.histogram(
        ttbar_tree, bins=frame.bins, weights=ttbar_weight ** 2
    )
    ttbar_errs = np.sqrt(ttbar_vals2)
    other_vals = other_vals + ttbar_vals
    other_errs = np.sqrt(other_errs ** 2 + ttbar_errs ** 2)

    data_vals, bins = np.histogram(data_tree, bins=frame.bins)
    data_errs = np.sqrt(data_vals)
    Zfac = 1.0062
    # data_vals = data_vals/Zfac
    # print(DY_vals)
    # print(other_vals)
    # print(data_vals)
    binSize = np.diff(bins)
    DY_vals = DY_vals / binSize
    other_vals = other_vals / binSize
    data_vals = data_vals / (binSize * 0.85)
    DY_errs = DY_errs / binSize
    other_errs = other_errs / binSize
    data_errs = data_errs / binSize
    MC_errs = np.sqrt(other_errs ** 2 + DY_errs ** 2)
    MC_vals = DY_vals + other_vals
    r_vals = data_vals / MC_vals
    r_errs = r_vals * np.sqrt((data_errs / data_vals) ** 2 + (MC_errs / MC_vals) ** 2)
    r_MCerrs = MC_errs / MC_vals
    # print(DY_vals)
    # print(other_vals)
    # print(data_vals)
    # MC_errR = coffea.hist.poisson_interval(MC_vals, MC_vals2)
    # print(MC_errR)
    # lower = np.array(MC_errR[0])
    # upper = np.array(MC_errR[1])
    # MC_errs = (lower+upper)/2.

    lumi = lumis[frame.year]
    Zprime = Zprime * lumi / 140.0
    G_RS = G_RS * lumi / 140.0

    # stat_err_opts = {'step': 'post', 'label': 'Stat. unc.',
    #             'hatch': '//////', 'facecolor': 'none',
    #             'edgecolor': (0, 0, 0, .5), 'linewidth': 0}

    hep.histplot(
        data_vals,
        bins,
        ax=axs[0],
        color="black",
        histtype="errorbar",
        label="$Data$",
        yerr=data_errs,
    )
    hep.histplot(
        [other_vals, DY_vals],
        bins,
        ax=axs[0],
        color=["orangered", frame.color],
        histtype="fill",
        label=["$t\\bar{t}$", "$\gamma/\mathrm{Z}\\rightarrow \mu^{+}\mu^{-}$"],
        edgecolor=(0, 0, 0),
        stack=True,
    )
    # hep.histplot( other_vals, bins, ax=axs[0], color='orangered', histtype='fill', label="$t\\bar{t}$", edgecolor=(0,0,0))
    bins_mid = (bins[1:] + bins[:-1]) / 2
    ax_signal.fill_between(
        x=bins[:-1],
        y1=MC_vals - MC_errs,
        y2=MC_vals + MC_errs,
        interpolate=False,
        color="skyblue",
        alpha=0.3,
        step="post",
    )
    hep.histplot(
        G_RS,
        bins,
        ax=ax_signal,
        color="green",
        histtype="step",
        label="$G_{KK}, k/\\bar{M}_{Pl}$ = 0.05, M = 3.5 TeV",
    )
    hep.histplot(
        Zprime,
        bins,
        ax=ax_signal,
        color="magenta",
        histtype="step",
        label="$Z'_{SSM}$, M = 5 TeV",
        linestyle="dashed",
    )
    axs[0].legend(loc=(0.35, 0.7))
    ax_signal.legend(loc=(0.45, 0.55), fontsize="xx-small")

    hep.histplot(r_vals - 1, bins, ax=axs[1], histtype="errorbar", color="black")
    axs[1].fill_between(
        x=bins_mid,
        y1=-r_MCerrs,
        y2=r_MCerrs,
        interpolate=True,
        color="skyblue",
        alpha=0.3,
    )

    _lumi = r"{lumi} (13 TeV)".format(lumi=str(lumi) + r" fb$^{-1}$")
    axs[0].text(
        1.0,
        1.08,
        _lumi,
        verticalalignment="top",
        horizontalalignment="right",
        transform=axs[0].transAxes,
        fontsize=28,
    )
    axs[0].text(
        0.1,
        0.92,
        "SR",
        verticalalignment="top",
        horizontalalignment="left",
        transform=axs[0].transAxes,
        fontsize=28,
    )
    axs[0].text(
        0.88,
        0.95,
        "CMS",
        verticalalignment="top",
        horizontalalignment="right",
        transform=axs[0].transAxes,
        fontsize=37,
        weight="bold",
    )

    fig.savefig(path_save + variable + "2jet.pdf")


if __name__ == "__main__":

    # load signal

    Zprime = uproot.open(path_signal)["signalHist_0"].values()
    Zprime = np.array(Zprime[-len(bins) + 1 :])

    G_RS = uproot.open(path_signal)["signalHist_1"].values()
    G_RS = np.array(G_RS[-len(bins) + 1 :])

    # load data
    DY_paths = glob.glob(path_DY)
    # DY_paths = [p for p in DY_paths if '3' in p]
    with ProcessPoolExecutor(max_workers=48) as executor:
        DY_dfs = list(executor.map(dd.read_parquet, DY_paths))
    DY_df = dd.concat(DY_dfs)
    # print(DY_df.shape)

    other_paths = glob.glob(other_path)
    # other_paths = [p for p in other_paths if '3' in p]
    with ProcessPoolExecutor(max_workers=48) as executor:
        other_dfs = list(executor.map(dd.read_parquet, other_paths))
    other_df = dd.concat(other_dfs)

    ttbar_paths = glob.glob(
        "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/ttbar_inclusive_v2/*/*.parquet"
    )
    with ProcessPoolExecutor(max_workers=48) as executor:
        ttbar_dfs = list(executor.map(dd.read_parquet, ttbar_paths))
    ttbar_df = dd.concat(ttbar_dfs)
    data_paths = glob.glob(path_data)
    # data_paths = [p for p in data_paths if '3' in p]
    with ProcessPoolExecutor(max_workers=48) as executor:
        data_dfs = list(executor.map(dd.read_parquet, data_paths))
    data_df = dd.concat(data_dfs)

    plot(DY_df, other_df, ttbar_df, data_df, Zprime, G_RS, variables_plot, path_save)
