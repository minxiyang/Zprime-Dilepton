import glob
import numpy as np
import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
import mplhep as hep
import uproot
from setFrame import setFrame
import itertools
from concurrent.futures import ProcessPoolExecutor


def files2hist(files, var, bins, masscut, njets=-1, iscut=False, iswgt=True):

    with ProcessPoolExecutor(max_workers=48) as executor:
        dfs = list(executor.map(dd.read_parquet, files))
    if njets == -1:
        df = dd.concat(dfs)
        mass = df.loc[(df["r"] != "ee") & (df["dielectron_mass"] > 120), var].compute()

        # print(mass)
        if iswgt:
            wgt = df.loc[
                (df["r"] != "ee") & (df["dielectron_mass"] > 120), "pu_wgt"
            ].compute()
            wgt[wgt < 0] = 0
            if iscut:
                genmass = df.loc[
                    (df["r"] != "ee") & (df["dielectron_mass"] > 120),
                    "dielectron_mass_gen",
                ].compute()
                wgt[genmass > masscut] = 0

            vals, bins = np.histogram(mass, bins=bins, weights=wgt)
            vals2, bins = np.histogram(mass, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:

            vals, bins = np.histogram(mass, bins=bins)
            errs = np.sqrt(vals)
            if var == "dielectron_mass":
                M = df.loc[
                    (df["r"] == "be") & (df["dielectron_mass"] > 2000), "run"
                ].compute()
                # M = mass.to_numpy()
                # M = np.sort(M)
                print("high mass events are:")
                print(M)
    else:
        df = dd.concat(dfs)
        mass = df.loc[(df["njets"] == njets) & (df["r"] != "ee"), var].compute()
        if iswgt:
            wgt = df.loc[(df["njets"] == njets) & (df["r"] != "ee"), "pu_wgt"].compute()
            wgt[wgt < 0] = 0
            if iscut:
                genmass = df.loc[
                    (df["njets"] == njets) & (df["r"] != "ee"), "dielectron_mass_gen"
                ].compute()
                wgt[genmass > masscut] = 0

            vals, bins = np.histogram(mass, bins=bins, weights=wgt)
            vals2, bins = np.histogram(mass, bins=bins, weights=wgt ** 2)
            errs = np.sqrt(vals2)
        else:
            vals, bins = np.histogram(mass, bins=bins)
            errs = np.sqrt(vals)

    binSize = np.diff(bins)

    # vals = vals/binSize
    # errs = errs/binSize
    return [vals, errs, bins]


def plots(axes, data, MCs, labels, colors, name):

    bins = data[2]
    MCs_vals = [MC[0] for MC in MCs]
    hep.histplot(
        data[0],
        bins,
        ax=axes[0],
        color="black",
        histtype="errorbar",
        label="$Data$",
        yerr=data[1],
    )
    hep.histplot(
        MCs_vals,
        bins,
        ax=axes[0],
        color=colors,
        histtype="fill",
        label=labels,
        edgecolor=(0, 0, 0),
        stack=True,
    )
    bins_mid = (bins[1:] + bins[:-1]) / 2
    MC_vals = np.zeros(len(MCs[0][0]))
    MC_errs = np.zeros(len(MCs[0][0]))
    for MC in MCs:
        MC_vals += MC[0]
        MC_errs += MC[1] ** 2
    MC_errs = np.sqrt(MC_errs)
    r_vals = data[0] / MC_vals
    r_errs = r_vals * np.sqrt((data[1] / data[0]) ** 2 + (MC_errs / MC_vals) ** 2)
    r_MCerrs = MC_errs / MC_vals
    axes[0].fill_between(
        x=bins[:-1],
        y1=MC_vals - MC_errs,
        y2=MC_vals + MC_errs,
        interpolate=False,
        color="skyblue",
        alpha=0.3,
        step="post",
    )
    axes[0].legend(loc=(0.7, 0.35), fontsize="xx-small")
    # axes[2].legend(loc=(0.45,0.55),fontsize='xx-small')
    hep.histplot(r_vals - 1, bins, ax=axes[1], histtype="errorbar", color="black")
    axes[1].fill_between(
        x=bins_mid,
        y1=-r_MCerrs,
        y2=r_MCerrs,
        interpolate=True,
        color="skyblue",
        alpha=0.3,
    )
    axes[3].savefig(
        f"/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/plots/{name}.pdf"
    )


path = "/depot/cms/users/minxi/NanoAOD_study/Zprime-Dilepton/output/"
path_dy = path + "DY_eev3/*/*.parquet"
dy_files = glob.glob(path_dy)
path_data = path + "UL_ee/*/*.parquet"
data_files = glob.glob(path_data)
path_tt_inclusive = path + "other_mc_eev2/ttbar_lep/*.parquet"
tt_inclusive_files = glob.glob(path_tt_inclusive)
path_tt = path + "other_mc_eev2/ttbar_lep_*/*.parquet"
tt_files = glob.glob(path_tt)
tt_files = [file_ for file_ in tt_files if "ext" not in file_]
path_wz = path + "other_mc_eev2/WZ*/*.parquet"
wz_files = glob.glob(path_wz)
path_zz = path + "other_mc_eev2/ZZ*/*.parquet"
zz_files = glob.glob(path_zz)
zz_files = [file_ for file_ in zz_files if "ext" not in file_]
path_tau = path + "other_mc_eev2/dyInclusive50/*.parquet"
tau_files = glob.glob(path_tau)
path_ww = path + "other_mc_eev2/WW*0/*.parquet"
ww_files = glob.glob(path_ww)
path_ww_inclusive = path + "other_mc_eev2/WWinclusive/*.parquet"
ww_inclusive_files = glob.glob(path_ww_inclusive)


# plot mass

var = "dielectron_mass"
bins_mass = (
    [j for j in range(120, 150, 5)]
    + [j for j in range(150, 200, 10)]
    + [j for j in range(200, 600, 20)]
    + [j for j in range(600, 900, 30)]
    + [j for j in range(900, 1250, 50)]
    + [j for j in range(1250, 1610, 60)]
    + [j for j in range(1610, 1890, 70)]
    + [j for j in range(1890, 3970, 80)]
    + [j for j in range(3970, 6070, 100)]
    + [6070]
)

for njets in [0, 1, 2]:
    dy_hist = files2hist(dy_files, var, bins_mass, 0, njets=njets)
    tt_hist = files2hist(tt_files, var, bins_mass, 0, njets=njets)
    zz_hist = files2hist(zz_files, var, bins_mass, 0, njets=njets)
    wz_hist = files2hist(wz_files, var, bins_mass, 0, njets=njets)
    ww_hist = files2hist(ww_files, var, bins_mass, 0, njets=njets)
    tau_hist = files2hist(tau_files, var, bins_mass, 0, njets=njets)
    tt_inclusive_hist = files2hist(
        tt_inclusive_files, var, bins_mass, 500.0, iscut=True, njets=njets
    )
    ww_inclusive_hist = files2hist(
        ww_inclusive_files, var, bins_mass, 200.0, iscut=True, njets=njets
    )
    data_hist = files2hist(data_files, var, bins_mass, 0, iswgt=False, njets=njets)
    tt_hist[0] = tt_hist[0] + tt_inclusive_hist[0]
    tt_hist[1] = np.sqrt(tt_hist[1] ** 2 + tt_inclusive_hist[1] ** 2)
    ww_hist[0] = ww_hist[0] + ww_inclusive_hist[0]
    ww_hist[1] = np.sqrt(ww_hist[1] ** 2 + ww_inclusive_hist[1] ** 2)

    MCs = [tau_hist, zz_hist, wz_hist, ww_hist, tt_hist, dy_hist]
    labels = [
        "$tautau$",
        "$ZZ$",
        "$WZ$",
        "$WW$",
        "$t\\bar{t}$",
        "$\gamma/\mathrm{Z}\\rightarrow \mu^{+}\mu^{-}$",
    ]
    colors = ["yellow", "red", "darkred", "brown", "blue", "skyblue"]
    ylims = [1e-5, 1e7]
    if njets == 1:
        ylims = [1e-6, 1e6]
    if njets == 2:
        ylims = [1e-7, 1e5]
    if njets == -1:
        name = var + "_inclusive"
    else:
        name = var + "_" + str(njets) + "nbjets"
    axes_mass = setFrame(
        "$\mathrm{m}(e^{+}e^{-})$ [GeV]",
        "Events/GeV",
        True,
        True,
        [120, 6000],
        ylims,
        "el",
        "2018",
    )
    plots(axes_mass, data_hist, MCs, labels, colors, name)

print("finsh the mass plot")

# plot Collins-Soper angle
"""
var = "dielecron_cos_theta_cs"
bins_cs = np.linspace(-1., 1., 26)

for njets in [-1, 0, 1, 2]:
    dy_hist = files2hist(dy_files, var, bins_cs, 0, njets=njets)
    tt_hist = files2hist(tt_files, var, bins_cs, 0, njets=njets)
    zz_hist = files2hist(zz_files, var, bins_cs, 0, njets=njets)
    wz_hist = files2hist(wz_files, var, bins_cs, 0, njets=njets)
    ww_hist = files2hist(ww_files, var, bins_cs, 0, njets=njets)
    tau_hist = files2hist(tau_files, var, bins_cs, 0, njets=njets)
    tt_inclusive_hist = files2hist(tt_inclusive_files, var, bins_cs, 500., iscut=True, njets=njets)
    ww_inclusive_hist = files2hist(ww_inclusive_files, var, bins_cs, 200., iscut=True, njets=njets)
    data_hist = files2hist(data_files, var, bins_cs, 0, iswgt=False, njets=njets)
    tt_hist[0] = tt_hist[0] + tt_inclusive_hist[0]
    tt_hist[1] = np.sqrt(tt_hist[1]**2 + tt_inclusive_hist[1]**2)
    ww_hist[0] = ww_hist[0] + ww_inclusive_hist[0]
    ww_hist[1] = np.sqrt(ww_hist[1]**2 + ww_inclusive_hist[1]**2)

    MCs = [tau_hist, zz_hist, wz_hist, ww_hist, tt_hist, dy_hist]
    labels = ["$tautau$","$ZZ$","$WZ$","$WW$","$t\\bar{t}$","$\gamma/\mathrm{Z}\\rightarrow \mu^{+}\mu^{-}$"]
    colors = ["yellow","red","darkred","brown","blue", "skyblue"]
    if njets == -1: name=var+"_inclusive"
    else: name=var+"_"+str(njets)+"nbjets"
    ylim = 50000
    if njets == 1: ylim = 10000
    elif njets == 2: ylim = 2000
    axes_cs = setFrame("$\mathrm{cos}\\theta$", "Events", False, False, [-1., 1.], [0, ylim], "el", "2018")
    plots(axes_cs, data_hist, MCs, labels, colors, name)

print("finsh the angle plot")
"""
# plot event as the number of the b-jets

var = "njets"
bins = np.linspace(0, 7, 8)
print(bins)
dy_hist = files2hist(dy_files, var, bins, 0)
tt_hist = files2hist(tt_files, var, bins, 0)
zz_hist = files2hist(zz_files, var, bins, 0)
wz_hist = files2hist(wz_files, var, bins, 0)
ww_hist = files2hist(ww_files, var, bins, 0)
tau_hist = files2hist(tau_files, var, bins, 0)
tt_inclusive_hist = files2hist(tt_inclusive_files, var, bins, 500.0, iscut=True)
ww_inclusive_hist = files2hist(ww_inclusive_files, var, bins, 200.0, iscut=True)
data_hist = files2hist(data_files, var, bins, 0, iswgt=False)
tt_hist[0] = tt_hist[0] + tt_inclusive_hist[0]
tt_hist[1] = np.sqrt(tt_hist[1] ** 2 + tt_inclusive_hist[1] ** 2)
ww_hist[0] = ww_hist[0] + ww_inclusive_hist[0]
ww_hist[1] = np.sqrt(ww_hist[1] ** 2 + ww_inclusive_hist[1] ** 2)

MCs = [tau_hist, zz_hist, wz_hist, ww_hist, tt_hist, dy_hist]
labels = [
    "$tautau$",
    "$ZZ$",
    "$WZ$",
    "$WW$",
    "$t\\bar{t}$",
    "$\gamma/\mathrm{Z}\\rightarrow \mu^{+}\mu^{-}$",
]
colors = ["yellow", "red", "darkred", "brown", "blue", "skyblue"]
name = var
axes = setFrame("b-jets", "Events", False, True, [0, 7], [1e-6, 1e10], "el", "2018")
plots(axes, data_hist, MCs, labels, colors, name)

print("finsh the njet plot")
