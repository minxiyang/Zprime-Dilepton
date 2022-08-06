import torch
import numpy as np
import torch.nn as nn
import glob
import dask.dataframe as dd
import dask
import pandas as pd
from matplotlib import pyplot as plt




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_features = [
    "mu1_eta",
    "mu1_phi",
    "mu1_pt_log",
    "mu2_eta",
    "mu2_phi",
    "mu2_pt_log",
    "jet1_eta",
    "jet1_phi",
    "jet1_btagDeepFlavB",
    "jet1_pt_log",
    "jet1_mass_log",
    "jet2_eta",
    "jet2_phi",
    "jet2_btagDeepFlavB",
    "jet2_pt_log",
    "jet2_mass_log",
    "njets",
    "met_log",
    "wgt_nominal",
    "dataset",
]

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ELU())
        
        for i in range(len(hidden_sizes)-1):

            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ELU())
            #layers.append(nn.Dropout(p=0.1))

        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        layers.append(nn.Sigmoid())
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            out = self.layers[i](out)
      
        return out



input_size = len(load_features) - 2
hidden_sizes = [128, 64, 32, 16, 16, 16, 16, 16, 16]
num_classes = 1

model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)
model = model.double()
model.load_state_dict(torch.load("model.ckpt"))
model.eval()

sig_path = "output/trainData_v1/2018/sig/bbll_4TeV_*_posLL/*parquet"
bkg_path = "output/trainData_v1/2018/bkg/*/*parquet"
data_path = "output/trainData_v1/2018/data/*/*parquet"
sig_files = glob.glob(sig_path)
bkg_files = glob.glob(bkg_path)
data_files = glob.glob(data_path)
df_sig = dd.read_parquet(sig_files)
df_sig = df_sig.compute()
df_bkg = dd.read_parquet(bkg_files)
df_bkg = df_bkg.compute()
df_data = dd.read_parquet(data_files)
df_data = df_data.compute()
df_sig = df_sig[load_features]
df_bkg = df_bkg[load_features]
df_data = df_data[load_features]

bkg_yield = sum(df_bkg.wgt_nominal)
sig_yield = sum(df_sig.wgt_nominal)
df_sig["wgt_nominal"] = df_sig["wgt_nominal"]*(bkg_yield/sig_yield)

sig = df_sig.drop(columns = ["wgt_nominal", "dataset"]).values
sig_wgt = df_sig["wgt_nominal"].values
bkg = df_bkg.drop(columns = ["wgt_nominal", "dataset"]).values
bkg_wgt = df_bkg["wgt_nominal"].values
data = df_data.drop(columns = ["wgt_nominal", "dataset"]).values
data_wgt = df_data["wgt_nominal"].values
 
sig = torch.from_numpy(sig).to(device)
sig_scores = model(sig.double()) 
sig_scores = sig_scores.cpu().detach().numpy()
sig_scores = sig_scores.ravel()

bkg = torch.from_numpy(bkg).to(device)
bkg_scores = model(bkg.double())
bkg_scores = bkg_scores.cpu().detach().numpy()
bkg_scores = bkg_scores.ravel()

data = torch.from_numpy(data).to(device)
data_scores = model(data.double())
data_scores = data_scores.cpu().detach().numpy()
data_scores = data_scores.ravel()

bins = np.linspace(0, 1, 100)
plt.hist(sig_scores, bins, weights=sig_wgt, alpha=0.3, label='sig')
plt.hist(bkg_scores, bins, weights=bkg_wgt, alpha=0.3, label='bkg')
plt.hist(data_scores, bins, weights=data_wgt, alpha=0.3, label='data')
plt.legend(loc='upper right')
plt.savefig("pic/bbll_4TeV_POSLL_scores.png")
plt.clf()

for key in df_sig.drop(columns = ["wgt_nominal", "dataset"]).columns:

    fea_sig = df_sig[key].values
    fea_bkg = df_bkg[key].values
    fea_data = df_data[key].values
    max_ = np.max(fea_data)
    min_ = np.min(fea_data)
    bins = np.linspace(min_, max_, 100)

    plt.hist(fea_sig, bins, weights=sig_wgt, alpha=0.3, label='sig')
    plt.hist(fea_bkg, bins, weights=bkg_wgt, alpha=0.3, label='bkg')
    plt.hist(fea_data, bins, weights=data_wgt, alpha=0.3, label='data')
    plt.legend(loc='upper right')
    plt.savefig(f"pic/bbll_4TeV_POSLL_{key}.png")
    plt.clf()

    plt.hist(fea_sig[sig_scores<0.25], bins, weights=sig_wgt[sig_scores<0.25], alpha=0.3, label='sig')
    plt.hist(fea_bkg[bkg_scores<0.25], bins, weights=bkg_wgt[bkg_scores<0.25], alpha=0.3, label='bkg')
    plt.hist(fea_data[data_scores<0.25], bins, weights=data_wgt[data_scores<0.25], alpha=0.3, label='data')
    plt.legend(loc='upper right')
    plt.savefig(f"pic/bbll_4TeV_POSLL_{key}_cut.png")
    plt.clf()



