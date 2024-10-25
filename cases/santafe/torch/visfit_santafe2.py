# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Description: this script is specifically made for reading the model from opt_santafe_*****_adam2.py and plot the resulting series and save the RMSEs in the different regions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pandas as pd
import subprocess

import matplotlib.pyplot as plt
import sys


# Versión Vanilla
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        super(VanillaRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers) #, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-N:, :])
        return out
    
    def __name__(self):
        return 'VanillaRNN'

# Versión LSTM
# Escribe el código de la versión LSTM del modelo
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x must have shape (nT,Nin), so that it must be a sample of series with nT time steps and Nin input variables
        out, _ = self.lstm(x)
        out = self.fc(out[-N:, :])
        return out
    
    def __name__(self):
        return 'LSTMModel'

# Versión GRU
# Escribe el código de la versión GRU del modelo
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        #https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers) #, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x must have shape (nT,Nin), so that it must be a sample of series with nT time steps and Nin input variables
        out, _ = self.gru(x)
        out = self.fc(out[-N:, :])
        return out
    
    def __name__(self):
        return 'GRUModel'


def total_loss(xdata,ydata):
    # compute the loss with the whole train/val dataset
    Loss = 0.
    for i in range(len(xdata)):
        xpred = model(train_sequences[i])
        Loss  += loss_fn(xpred, ydata[i].unsqueeze(-1)).detach().numpy().item()
    return Loss/(i+1)


#job_id = input('Insert JOBID ')
job_id = sys.argv[1]

logfile = 'log_'+str(job_id)+'.csv'
log_data = pd.read_csv(logfile).iloc[0] # load LOGS

data = np.loadtxt('../'+log_data['datfile'])

n = int(log_data['nT']); N = int(log_data['Nwout'])
ddist_seed = int(log_data['datdist seed'])
LR = float(log_data['LR'])
shuffle = bool(log_data['shuffle'])

# Preprocesado de los datos
nsamples = int(len(data[:,1])/n)

times = data[:,0].reshape(nsamples,n)
sequences = data[:,1].reshape(nsamples,n,1)
tarseq = data[:,2].reshape(nsamples,n)
targets =   np.array([item[-N:] for item in data[:,2].reshape(nsamples,n)])
tartimes =  np.array([item[-N:] for item in times])


# Dividir los datos en una secuencia de entrenamiento y validación y otra de test
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% para entrenamiento y validación
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% para validación de entre el conjunto de entreno + validación

random.seed(ddist_seed)
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training

# TRAINING DATA
train_times     = np.array([times[i] for i in tr_samples])
train_tartimes  = np.array([tartimes[i] for i in tr_samples])
train_tarseq    = np.array([tarseq[i] for i in tr_samples])
train_sequences = torch.tensor(np.array([sequences[i] for i in tr_samples])).float()
train_targets   = torch.tensor(np.array([targets[i] for i in tr_samples])).float()

# VALIDATION DATA
val_times       = np.array([times[i] for i in val_samples])
val_tartimes    = np.array([tartimes[i] for i in val_samples])
val_tarseq      = np.array([tarseq[i] for i in val_samples])
val_sequences   = torch.tensor(np.array([sequences[i] for i in val_samples])).float()
val_targets     = torch.tensor(np.array([targets[i] for i in val_samples])).float()

# TEST DATA
test_times      = times[trval_nsamples:]
test_tartimes   = tartimes[trval_nsamples:]
test_tarseq     = tarseq[trval_nsamples:]
test_sequences  = torch.tensor(sequences[trval_nsamples:]).float()
test_targets    = torch.tensor(targets[trval_nsamples:]).float()

# TEST DATA FILLING ALL POINTS
trval_npoints = trval_nsamples * n
npoints = n * nsamples
ts_npoints = npoints - trval_npoints
filltest_times  = np.array([data[:,0][trval_npoints+N*i:trval_npoints+N*(i+1)] for i in range(ts_npoints//N)])
filltest_sequences = torch.tensor(np.array([data[:,1][trval_npoints-n+N+N*i:trval_npoints-n+N+N*i+n] for i in range(ts_npoints//N)])).float().unsqueeze(-1)
filltest_targets = torch.tensor(np.array([data[:,2][trval_npoints+N*i:trval_npoints+N*i+N] for i in range(ts_npoints//N)])).float()


modelfile = 'state_best_'+job_id+'.pt'
model = torch.load(modelfile)
loss_fn = nn.MSELoss()

model.eval()


train_outputs = np.array([model(train_seq).detach().numpy() for train_seq in train_sequences])
val_outputs   = np.array([model(val_seq).detach().numpy() for val_seq in val_sequences])
test_outputs  = np.array([model(test_seq).detach().numpy() for test_seq in test_sequences])
filltest_outputs = np.array([model(filltest_seq).detach().numpy() for filltest_seq in filltest_sequences])


train_loss  = torch.sqrt(loss_fn(torch.tensor(train_outputs),  train_targets.unsqueeze(-1)))
val_loss  = torch.sqrt(loss_fn(torch.tensor(val_outputs),  val_targets.unsqueeze(-1)))
test_loss = torch.sqrt(loss_fn(torch.tensor(test_outputs), test_targets.unsqueeze(-1)))
filltest_loss = torch.sqrt(loss_fn(torch.tensor(filltest_outputs), filltest_targets.unsqueeze(-1)))


train_corrcoef = np.corrcoef(np.array([train_outputs.flatten(),train_targets.flatten()]), rowvar=True)[0,1]
val_corrcoef = np.corrcoef(np.array([val_outputs.flatten(),val_targets.flatten()]), rowvar=True)[0,1]
test_corrcoef = np.corrcoef(np.array([test_outputs.flatten(),test_targets.flatten()]), rowvar=True)[0,1]
filltest_corrcoef = np.corrcoef(np.array([filltest_outputs.flatten(),filltest_targets.flatten()]), rowvar=True)[0,1]
print('Training   RMSE = %8.5f     R = %8.5f' %(train_loss,train_corrcoef))
print('Validation RMSE = %8.5f     R = %8.5f' %(val_loss,val_corrcoef))
print('Test       RMSE = %8.5f     R = %8.5f' %(test_loss,test_corrcoef))
print('Com. test  RMSE = %8.5f     R = %8.5f' %(filltest_loss,filltest_corrcoef))


try:
    command = "seff %s" %job_id[-5:]
    result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    resultstr = result.stdout
    info = {}
    for line in resultstr.splitlines():
        key, value = line.split(": ")
        info[key] = value
except:
    print('Could not run seff command')



try:
    logdf = pd.read_csv(logfile)
    try:
        logdf['cores'] = [int(info['Nodes'])*int(info['Cores per node'])]
    except:
        print('Could not write cores on log file')
    try:
        logdf['memory'] = [info['Memory Utilized']]
    except:
        print('Could not write memory on log file')
    try:
        logdf['CPU time'] = [info['CPU Utilized']]
    except:
        print('Could not write CPU time on log file')
    try:
        logdf['time'] = [info['Job Wall-clock time']]
    except:
        print('Could not write time on log file')
    try:
        log_data['RMSE tra']
        try:
            assert np.abs(train_loss-log_data['RMSE tra']) < 1.e-5, 'Calculated TR RMSE (1) does not match the one in the main program (2) '
        except AssertionError as msg:
            print(msg, train_loss.detach().numpy().item(), log_data['RMSE tra'])
    except:
        print('WARNING. Coincidence of train_loss not found in original log. Write the one computed her instead.')
        logdf['RMSE tra'] = [train_loss.detach().numpy().item()]
    try:
        log_data['RMSE val']
        try:
            assert np.abs(val_loss-log_data['RMSE val']) < 1.e-5, 'Calculated VAL RMSE (1) does not match the one in the main program (2) '
        except AssertionError as msg:
            print(msg, val_loss.detach().numpy().item(), log_data['RMSE val'])
    except:
        print('WARNING. Coincidence of val_loss not found in original log. Write the one computed her instead.')
        logdf['RMSE val'] = [val_loss.detach().numpy().item()]
    try:
        logdf['RMSE tes'] = [test_loss.detach().numpy().item()]
    except:
        print('Could not write RMSE tes on log file')
    try:
        logdf['RMSE ftes'] = [filltest_loss.detach().numpy().item()]
    except:
        print('Could not write RMSE filltest on log file')
    try:
        logdf['R tra'] = [train_corrcoef]
    except:
        print('Could not write R tra on log file')
    try:
        logdf['R val'] = [val_corrcoef]
    except:
        print('Could not write R val on log file')
    try:
        logdf['R tes'] = [test_corrcoef]
    except:
        print('Could not write R tes on log file')
    try:
        logdf['R ftes'] = [filltest_corrcoef]
    except:
        print('Could not write R ftes on log file')

    with open(logfile, 'w') as f:
        logdf.to_csv(f, index=False)
    
except Exception as e:
    print('Could not open log file: ', e)


try:
    historicdf = pd.read_csv('historic_torch_santafe.csv')
    historicdf.loc[len(historicdf)] = logdf.iloc[0]
    historicdf.to_csv('historic_torch_santafe.csv', index=False)
except:
    try:
        logdf.to_csv('historic_torch_santafe.csv', index=False)
    except:
        print('Could not save historic_torch_santafe')



##### PLOT #####
plt.figure(1,figsize=(100,5))
plt.title(model.__name__())
for x,y in zip(train_times,train_tarseq):
    plt.plot(x,y,'-', color='tab:orange')
for x,y in zip(val_times,val_tarseq):
    plt.plot(x,y,'-', color='tab:blue')
for x,y in zip(test_times,test_tarseq):
    plt.plot(x,y,'-', color='tab:green')

for x,y in zip(train_tartimes,train_outputs):
    plt.plot(x, y, '.', color='red')
for x,y in zip(val_tartimes,val_outputs):
    plt.plot(x, y, '.', color='blue')
for x,y in zip(test_tartimes,test_outputs):
    plt.plot(x, y, '.', color='green')
for x,y in zip(filltest_times,filltest_outputs):
    plt.plot(x, y, '*', color='pink')

plt.savefig(modelfile[:-3]+'.png')
