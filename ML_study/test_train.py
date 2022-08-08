import torch
import numpy as np
import torch.nn as nn
import glob
import dask.dataframe as dd
import dask
import pandas as pd





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
    "dimuon_mass",
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



input_size = len(load_features) - 3
hidden_sizes = [128, 64, 32, 16, 16, 16, 16, 16, 16]
num_classes = 1
num_epochs = 20
batch_size = 128
learning_rate = 0.0001

model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)
model = model.double()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
decayRate = 0.8
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
sig_path = "output/trainData_v1/2018/sig/bbll_4TeV_*_posLL/*parquet"
bkg_path = "output/trainData_v1/2018/bkg/*/*parquet"
sig_files = glob.glob(sig_path)
bkg_files = glob.glob(bkg_path)
df_sig = dd.read_parquet(sig_files)
df_sig = df_sig.compute()
df_sig = df_sig.loc[(abs(df_sig["jet1_eta"])<2.4) & (abs(df_sig["jet2_eta"])<2.4), :]
df_bkg = dd.read_parquet(bkg_files)
df_bkg = df_bkg.compute()
df_bkg = df_bkg.loc[(abs(df_bkg["jet1_eta"])<2.4) & (abs(df_bkg["jet2_eta"])<2.4), :]
df_sig = df_sig[load_features]
df_bkg = df_bkg[load_features]
df_sig["label"] = 1.0
df_bkg["label"] = 0

bkg_yield = sum(df_bkg.wgt_nominal)
sig_yield = sum(df_sig.wgt_nominal)
df_sig["wgt_nominal"] = df_sig["wgt_nominal"]*(bkg_yield/sig_yield)

dataset = pd.concat([df_sig, df_bkg], ignore_index=True)
dataset = dataset.dropna()
dataset = dataset.sample(frac=1.)
train_size = int(0.8*len(dataset))
train_data = dataset.iloc[:train_size, :]
train_data.to_parquet("output/trainData_v1/2018/train_data.parquet")
#train_data = train_data.loc[train_data["dimuon_mass"] > 200., :].copy()
train = train_data.drop(columns = ["wgt_nominal", "dataset", "label", "dimuon_mass"]).values
train_labels = train_data["label"].values
train_wgt = train_data["wgt_nominal"].values
val_data = dataset.iloc[train_size:, :]
val_data.to_parquet("output/trainData_v1/2018/val_data.parquet")
#val_data = val_data.loc[val_data["dimuon_mass"] > 200., :].copy()
val = val_data.drop(columns = ["wgt_nominal", "dataset", "label", "dimuon_mass"]).values
val_labels = val_data["label"].values
val_wgt = val_data["wgt_nominal"].values

#train = torch.from_numpy(train)
#train_label = torch.from_numpy(train_label)
#train_wgt = torch.from_numpy(train_wgt)

total_step = train_size
for epoch in range(num_epochs):
    mean_loss = 0
    tot_wgt = 0
    val_mean_loss = 0
    val_tot_wgt = 0
    for i in range(int(train_size/batch_size)):  
        # Move tensors to the configured device
        data = torch.from_numpy(train[i*batch_size: (i+1)*batch_size]).to(device)
        label = torch.from_numpy(train_labels[i*batch_size: (i+1)*batch_size].reshape((batch_size,1))).to(device)
        #w = torch.from_numpy(train_wgt[i]).to(device)
        w = torch.from_numpy(train_wgt[i*batch_size: (i+1)*batch_size]).to(device)
        # Forward pass
        outputs = model(data.double()) 
        loss = criterion(outputs, label.double())
        weight_loss = loss*w
        # Backward and optimize
        optimizer.zero_grad()
        weight_loss.mean().backward()
        optimizer.step()
        mean_loss += weight_loss.mean().item()*batch_size
        tot_wgt += sum(w.cpu().detach().numpy())
        if i%4 == 0:
            j = int(i/4)
            val_data = torch.from_numpy(val[j*batch_size: (j+1)*batch_size]).to(device)
            val_label = torch.from_numpy(val_labels[j*batch_size: (j+1)*batch_size].reshape(val_labels[j*batch_size: (j+1)*batch_size].shape[0],1)).to(device)
            val_w = torch.from_numpy(val_wgt[j*batch_size: (j+1)*batch_size]).to(device)
            val_outputs = model(val_data.double())
            #if abs(val_outputs)>1:
            #    print(val_data)
            #    print(val_output)
            val_loss = criterion(val_outputs, val_label.double())
            val_mean_loss += (val_loss*val_w).mean().item()*batch_size
            val_tot_wgt += sum(val_w.cpu().detach().numpy())
            
        if (i+1) % 1000 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, int(total_step/batch_size), mean_loss/tot_wgt, val_mean_loss/val_tot_wgt))
            mean_loss=0
            tot_wgt=0
            val_mean_loss=0
            val_tot_wgt=0
    my_lr_scheduler.step()

torch.save(model.state_dict(), 'model.ckpt')




