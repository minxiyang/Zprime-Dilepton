import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator
import matplotlib as mpl
import matplotlib.style
import matplotlib.font_manager
import mplhep as hep
import numpy as np

lumi_mu = {"2016": 36, "2017": 42, "2018": 62}
lumi_el = {"2016": 36, "2017": 42, "2018": 60}


def setFrame(xtitle, ytitle, ratio, signal, logx, logy, xRange, yRange, flavor, year):

    style = hep.style.CMS
    style["mathtext.fontset"] = "cm"
    style["mathtext.default"] = "rm"
    plt.style.use(style)
    hep.style.use(style)
    if ratio:

        fig, axs = plt.subplots(
            2,
            sharex=True,
            sharey=False,
            gridspec_kw={
                "height_ratios": [4, 1],
            },
        )
        plt.subplots_adjust(hspace=0.07)
        axs[0].xaxis.set_tick_params(labeltop=False)
        axs[1].xaxis.set_tick_params(labeltop=False)
        axs[1].set_xlabel(xtitle)
        axs[0].set_ylabel(ytitle)
        axs[1].set_ylabel("(Data-Bkg)/Bkg", fontsize=16)
        axs[1].set_xlim(xRange)
        axs[0].set_ylim(yRange)
        if logx:
            axs[0].set_xscale("log")
        if logy:
            axs[0].set_yscale("log")

        axs[1].set_ylim([-1, 1])
        if logx:

            tick = []
            for x in [200, 300, 500, 1000, 2000, 4000]:
                if x > xRange[0] and x < xRange[-1]:
                    tick.append(x)
            axs[1].set_xticks(tick)
            axs[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            axs[1].get_xaxis().set_minor_formatter(plt.NullFormatter())
        if logy:
            locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
            locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        axs[1].hlines(
            (-0.5, 0, 0.5), xRange[0], xRange[1], color="black", linestyles="dotted"
        )
        if signal:
            ax_signal = axs[0].twiny()
            ax_signal.xaxis.set_tick_params(labeltop=False)
            if logx:
                ax_signal.set_xscale("log")
            if logy:
                ax_signal.set_yscale("log")
            ax_signal.set_ylim(yRange)
            ax_signal.set_xlim(xRange)
            if logx:
                ax_signal.set_xticks(tick)
                ax_signal.get_xaxis().set_major_formatter(
                    matplotlib.ticker.ScalarFormatter()
                )
                ax_signal.get_xaxis().set_minor_formatter(plt.NullFormatter())
            if logy:
                locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
                ax_signal.get_yaxis().set_major_locator(locmaj)
                ax_signal.yaxis.set_minor_locator(locmin)
                ax_signal.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                axs[0].get_yaxis().set_major_locator(locmaj)
                axs[0].yaxis.set_minor_locator(locmin)
                axs[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if flavor == "el":
            lumi = lumi_el[year]
        else:
            lumi = lumi_mu[year]
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
            0.84,
            0.97,
            "CMS",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[0].transAxes,
            fontsize=37,
            weight="bold",
        )
        axs[0].text(
            0.9,
            0.9,
            "Preliminary",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[0].transAxes,
            fontsize=30,
            weight="bold",
        )
        if signal:
            return (axs[0], axs[1], ax_signal, fig)
        else:
            return (axs[0], axs[1], fig)

    else:

        ax = plt.gca()
        fig = plt.gcf()
        ax.xaxis.set_tick_params(labeltop=False)
        ax.set_ylabel(ytitle)
        ax.set_ylim(yRange)
        ax.set_xlabel(xtitle)
        ax.set_xlim(xRange)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")

        if logx:

            tick = []
            for x in [200, 300, 500, 1000, 2000, 4000]:
                if x > xRange[0] and x < xRange[-1]:
                    tick.append(x)
            ax.set_xticks(tick)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(plt.NullFormatter())
        if logy:
            locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
            locmin = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        if signal:
            ax_signal = ax.twiny()
            ax_signal.xaxis.set_tick_params(labeltop=False)
            if logx:
                ax_signal.set_xscale("log")
            if logy:
                ax_signal.set_yscale("log")
            ax_signal.set_ylim(yRange)
            ax_signal.set_xlim(xRange)
            if logx:
                ax_signal.set_xticks(tick)
                ax_signal.get_xaxis().set_major_formatter(
                    matplotlib.ticker.ScalarFormatter()
                )
                ax_signal.get_xaxis().set_minor_formatter(plt.NullFormatter())
            if logy:
                locmaj = LogLocator(base=10.0, subs=(1.0,), numticks=100)
                ax_signal.get_yaxis().set_major_locator(locmaj)
                ax_signal.yaxis.set_minor_locator(locmin)
                ax_signal.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.get_yaxis().set_major_locator(locmaj)
                ax.yaxis.set_minor_locator(locmin)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if flavor == "el":
            lumi = lumi_el[year]
        else:
            lumi = lumi_mu[year]
        _lumi = r"{lumi} (13 TeV)".format(lumi=str(lumi) + r" fb$^{-1}$")
        ax.text(
            1.0,
            1.08,
            _lumi,
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=28,
        )
        ax.text(
            0.84,
            0.97,
            "CMS",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=37,
            weight="bold",
        )
        ax.text(
            0.9,
            0.9,
            "Preliminary",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            fontsize=30,
            weight="bold",
        )
        if signal:
            return (ax, ax_signal, fig)
        else:
            return (ax, fig)
