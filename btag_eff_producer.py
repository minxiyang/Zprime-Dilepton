from hist import Hist
import glob
import pandas as pd
import numpy as np
import pickle

files = glob.glob("output/ttbar_eff/*/*")
df = pd.read_parquet(files[0:100])
df =  df[(df["accepted"]) & (df["reco"]) & (df["ID_pass"]) & (df["hlt"]>0)&(df["gpv"])&(df["Jet_ID"])]
binx = [ 10., 20., 30., 40., 50., 100., 200., 400., 1000.]
biny = [0., 0.4, 1.4, 2.4]
print(df["Jet_pt"].to_numpy()[0:100])
print(df["Jet_pt_reco"].to_numpy()[0:100])
efficiencyinfo = (
    Hist.new
    .Var(binx, name="pt")
    .Var(biny, name="abseta")
    .IntCat([0, 4, 5], name="flavor")
    .Bool(name="passWP")
    .Double()
    .fill(
        pt = df.Jet_pt_reco.to_numpy(),
        abseta = np.abs(df.Jet_eta_reco.to_numpy()),
        flavor = abs(df["flavor_reco"].to_numpy()),
        passWP = df.btag.to_numpy(), 
        weight = df.wgt_nominal.to_numpy(),
    )
)

eff = efficiencyinfo[{"passWP": True}] / efficiencyinfo[{"passWP": sum}]
print(eff)
print(eff.values())

path = "data/b-tagging/UL2018_ttbar_eff.pickle"
#with open(path, "wb+") as handle:
#    pickle.dump(eff, handle, protocol=pickle.HIGHEST_PROTOCOL)



