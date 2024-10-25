# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Description: this script creates the class and runs VanillaRNN training, running the samples (windows of the series) with HAND-MADE lines of code.
This allows us to turn on/off the shuffle and mimic the QRNN algorithm.
"""

#### LOAD MODULES #######################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import sys
import time as tm

try:
    job_id = int(sys.argv[1])
except:
    job_id = int(tm.time())%100000000


torch.manual_seed(job_id)

#### DATA PREPARATION  ##################################################################
fname = 'data_santafe10.dat'
data = np.loadtxt('../'+fname)

x_data = np.array([[data[i,1]] for i in range(len(data[:,1]))]) # 2D-array!
#x_data = np.array([[data[i,1],data[i,1]] for i in range(len(data[:,1]))]) # 2D-array!
y_data = np.array([item for item in data[:len(x_data),2]])    # 1D-array!

nT = 20 # size of predicting window
Nwout = 5  # size of window to predict
nsamples = int(len(data[:,1])/nT)

sequences = x_data.reshape(nsamples,nT,1)
#sequences = x_data.reshape(nsamples,nT,2)
targets =   np.array([item[-Nwout:] for item in y_data.reshape(nsamples,nT)])


# Divide data into training and validation set, and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in the training + validation set

ddist_seed = 4 # data distribution seed
random.seed(ddist_seed)
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training

# TRAINING DATA
train_sequences = torch.tensor(np.array([sequences[i] for i in tr_samples])).float()
train_targets   = torch.tensor(np.array([targets[i] for i in tr_samples])).float()

# VALIDATION DATA
val_sequences   = torch.tensor(np.array([sequences[i] for i in val_samples])).float()
val_targets     = torch.tensor(np.array([targets[i] for i in val_samples])).float()
#####################################################################


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers) #, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x must have shape (nT,Nin), so that it must be a sample of series with nT time steps and Nin input variables
        out, _ = self.rnn(x)
        out = self.fc(out[-Nwout:, :])
        return out
    
    def __name__(self):
        return 'VanillaRNN'


Sinput = 1; Shidden = 4; Sout = 1; Nlayers = 1

model = VanillaRNN(input_size=Sinput, hidden_size=Shidden, output_size=Sout, num_layers=Nlayers)
torch.save(model, 'state0_%i.pt' %job_id)

Nparams = sum(p.numel() for p in model.parameters())
print('total number of weights: ', Nparams)


##### TRAINING ##################################################################################
lossfile = 'loss_'+str(job_id)+'.dat'
logfile  = 'log_'+str(job_id)+'.csv'


loss_fn = nn.MSELoss()
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
Nepochs = 2000 # maximum number of iterations
shuffle = True # shuffle

losses_tra = []
losses_val = []

best_Eval = 50. # high value

opt_name = str(optimizer).split()[0]
logdf = pd.DataFrame([{'jobID':job_id,  'cores':None, 'memory':None, 'CPU time':None, 'time':None,
                      'datfile':fname, 'datdist seed':ddist_seed, 'Npoints':len(x_data), 'Nwout':Nwout,
                      'model': model.__name__(), 'input_size':Sinput, 'hidden_size':Shidden, 'output_size':Sout, 'num_layers':Nlayers, 'Nparams': Nparams, 'nT':nT, 'Nwout':Nwout,
                      'optimizer':opt_name, 'Nepochs':Nepochs, 'shuffle':shuffle, 'LR':LR}])

with open(logfile, 'w') as f:
    logdf.to_csv(f, index=False)


np.random.seed(job_id)
Ntr = len(train_sequences)
for epoch in range(Nepochs):
    if shuffle:
        indices = np.random.permutation(Ntr)
    else:
        indices = list(range(Ntr))
    
    X_shuffled = train_sequences[indices]
    y_shuffled = train_targets[indices]

    #https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    for xsample, ysample in zip(X_shuffled, y_shuffled):
        optimizer.zero_grad()
        predictions = model(xsample)
        loss = loss_fn(predictions, ysample.unsqueeze(-1))
        loss.backward()
        optimizer.step()
    

    #model.eval()
    with torch.no_grad():
        Etr = 0.
        for i in range(len(train_sequences)):
            xsamp_tra = model(train_sequences[i])
            Etr  += loss_fn(xsamp_tra, train_targets[i].unsqueeze(-1)).detach().numpy().item()
        Etr = (1./(i+1))*Etr
        Eval = 0.
        for i in range(len(val_sequences)):
            xsamp_val = model(val_sequences[i])
            Eval += loss_fn(xsamp_val, val_targets[i].unsqueeze(-1)).detach().numpy().item()
        Eval = (1./(i+1))*Eval
    
    with open(lossfile, 'a') as f:
        f.write('==> Loss (RMSE)    %4i  %10.6f %10.6f' %(epoch, np.sqrt(Etr), np.sqrt(Eval)))
        f.write('\n')

    if Eval < best_Eval:
        torch.save(model, 'state_best_%i.pt' %job_id)
        best_Etr  = Etr
        best_Eval = Eval
        best_epoch = epoch
    
    if epoch%10==0: print('epoch ', epoch, np.sqrt(Etr), np.sqrt(Eval))


torch.save(model, 'state_final_%i.pt' %job_id)


logdf['final epochs'] = [epoch+1]
logdf['epoch min VAL'] = [best_epoch+1]
logdf['RMSE tra'] = [np.sqrt(best_Etr)]
logdf['RMSE val'] = [np.sqrt(best_Eval)]
logdf['RMSE tes'] = [None]
logdf['RMSE ftes'] = [None]

with open(logfile, 'w') as f:
    logdf.to_csv(f, index=False)