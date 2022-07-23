import hist
import pickle
import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt
from coffea.lookup_tools.dense_lookup import dense_lookup


path_eff = "data/b-tagging/UL2018_ttbar_eff.pickle"
with open(path_eff, "rb") as handle:
    eff = pickle.load(handle)


efflookup = dense_lookup(eff.values(), [ax.edges for ax in eff.axes])

edgex = np.array(eff.axes[0])
binx = [edgex[0, 0]]
binCx = []
for bin_ in edgex:
    binx.append(bin_[1])
    binCx.append((bin_[0] + bin_[1]) / 2.0)

edgey = np.array(eff.axes[1])
biny = [edgey[0, 0]]
binCy = []
for bin_ in edgey:
    biny.append(bin_[1])
    binCy.append((bin_[0] + bin_[1]) / 2.0)
flavor = {"light": 0, "c": 1, "b": 2}

for key in flavor.keys():
    M = eff[{"flavor": flavor[key]}].values()

    fig, ax = plt.subplots()
    hep.hist2dplot(
        M,
        xbins=binx,
        ybins=biny,
        color="r",
        cbar=False,
        cbarsize="7%",
        cbarpad=0.2,
        cbarpos="right",
        cbarextend=False,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("jet pt [GeV]")
    ax.set_ylabel("|$\eta$|")

    for x in binCx:
        for y in binCy:

            ax.text(
                x,
                y,
                format(efflookup(x, y, flavor[key]), ".3f"),
                ha="center",
                va="center",
                color="white",
                fontsize=4,
            )

    fig.savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/btagging_eff_{key}.pdf"
    )
    fig.clf()
