from hist import Hist
import glob
import pandas as pd
import numpy as np
import pickle

files = glob.glob("output/ttbar_eff/*/*")
df = pd.read_parquet(files)
df =  df[(df["accepted"]) & (df["reco"]) & (df["ID_pass"]) & (df["hlt"]>0)]

binx = [ 20., 25., 30.,  35., 40., 45., 50., 60., 70., 90., 150., 250., 500., 1000.]
biny = np.array([0., 0.2, 0.4, 0.6, 0.9, 1.2, 1.5, 1.9, 2.4])
print(df["hadronFlavour"].to_numpy())
efficiencyinfo = (
    Hist.new
    .Var(binx, name="pt")
    .Var(biny, name="abseta")
    .IntCat([0, 4, 5], name="flavor")
    .Bool(name="passWP")
    .Double()
    .fill(
        pt = df.Jet_pt.to_numpy(),
        abseta = np.abs(df.Jet_eta.to_numpy()),
        flavor = abs(df["hadronFlavour"].to_numpy()),
        passWP = df.btag.to_numpy(), 
        weight = df.wgt_nominal.to_numpy(),
    )
)

eff = efficiencyinfo[{"passWP": True}] / efficiencyinfo[{"passWP": sum}]
print(eff)
print(eff.values())

path = "data/b-tagging/UL2018_ttbar_eff.pickle"
with open(path, "wb+") as handle:
    pickle.dump(eff, handle, protocol=pickle.HIGHEST_PROTOCOL)



