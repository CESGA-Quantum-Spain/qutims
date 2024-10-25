# -*- coding: utf-8 -*-
"""
Created on

@author: jdviqueira
"""

import sys
sys.path.append('../../..')

import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import csv
import pandas as pd

from qurecnets.emc import EMCZ2
import sys

def rmse(a,b):
    return np.sqrt((1./len(a))*sum([(ai-bi)**2 for ai,bi in zip(a,b)]))

#job_id = input('Insert JOBID ')
job_id = sys.argv[1]

logfile = 'log_'+str(job_id)+'.csv'
log_data = pd.read_csv(logfile).iloc[0] # load LOGS

fname_x = '../'+log_data['datfile'] # INPUT
fname_p = 'param_best_'+job_id+'.dat' #PARAMETERS

data = np.loadtxt(fname_x)


Npoints = int(log_data['Npoints']); nT = int(log_data['nT']); Nwout = int(log_data['Nwout'])
nE = int(log_data['nE']); nM = int(log_data['nM']); nL = int(log_data['nL']); nx = int(log_data['nx'])
ddist_seed = int(log_data['datdist seed'])
pinorm = float(log_data['normalization'])
LR = float(log_data['LR'])
shuffle = bool(log_data['shuffle'])


if nE == 1 and data.shape[1]==3:
    x_data = pinorm*np.array([[data[i,1]] for i in range(len(data[:,1]))])
if nE > 1 and data.shape[1]==3:
    x_data = pinorm*np.array([[data[i,1] for j in range(nE)] for i in range(len(data[:,1]))])
if nE > 1 and data.shape[1]==(nE+2):
    x_data = pinorm*data[:,1:-2]


## PREPARING DATA #############################################################
# Data preprocessing
Npoints = len(x_data)
nT = 20 # Prediction window size
Nwout = 5  # Size of window to predict
nsamples = int(len(data[:,1])/nT)

times = data[:,0].reshape(nsamples,nT)
sequences = x_data.reshape(nsamples,nT,x_data.shape[1])
#if nE == 1:
#    sequences = x_data.reshape(nsamples,nT,1)
#if nE == 2:
#    sequences = x_data.reshape(nsamples,nT,2)

#tarseq = data[:,3].reshape(nsamples,nT)
tarseq = data[:,-1].reshape(nsamples,nT)
#targets =   np.array([item[-Nwout:] for item in data[:,3].reshape(nsamples,nT)])
targets =   np.array([item[-Nwout:] for item in data[:,-1].reshape(nsamples,nT)])
tartimes =  np.array([item[-Nwout:] for item in times])


# Divide data into training + validation set and test set
TRVAL = 80
trval_nsamples = nsamples * TRVAL // 100 # TRVAL% for training and validation
VAL = 20
val_nsamples = trval_nsamples * VAL // 100 #VAL% for validation in training + validation set

random.seed(ddist_seed) # !!! must be the same as in training
val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training

# TRAINING DATA
train_times     = np.array([times[i] for i in tr_samples])
train_tartimes  = np.array([tartimes[i] for i in tr_samples])
train_tarseq    = np.array([tarseq[i] for i in tr_samples])
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_times       = np.array([times[i] for i in val_samples])
val_tartimes    = np.array([tartimes[i] for i in val_samples])
val_tarseq      = np.array([tarseq[i] for i in val_samples])
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])

# TEST DATA
test_times      = times[trval_nsamples:]
test_tartimes   = tartimes[trval_nsamples:]
test_tarseq     = tarseq[trval_nsamples:]
test_sequences  = sequences[trval_nsamples:]
test_targets    = targets[trval_nsamples:]

# TEST DATA FILLING ALL POINTS
trval_npoints = trval_nsamples * nT
npoints = nT * nsamples
ts_npoints = npoints - trval_npoints
filltest_times  = np.array([data[:,0][trval_npoints+Nwout*i:trval_npoints+Nwout*(i+1)] for i in range(ts_npoints//Nwout)])
filltest_sequences = np.array([x_data[trval_npoints-nT+Nwout+Nwout*i:trval_npoints-nT+Nwout+Nwout*i+nT] for i in range(ts_npoints//Nwout)])
#filltest_targets = np.array([data[:,3][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])
filltest_targets = np.array([data[:,-1][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])
###############################################################################


params = np.loadtxt(fname_p)
Nparam = len(params)

qrnn = EMCZ2(nT,nE,nM,nL,nx)

# TRAINING PREDICTIONS
train_outputs = []
for train_sample in train_sequences:
    evalu = qrnn.evaluate(params[1:], train_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    train_outputs += [ypredi[-Nwout:]]
train_outputs = np.array(train_outputs)
train_rmse = rmse(train_outputs.flatten(),train_targets.flatten())
train_corrcoef = np.corrcoef(np.array([train_outputs.flatten(),train_targets.flatten()]), rowvar=True)[0,1]
print('Training   RMSE = %8.5f     R = %8.5f' %(train_rmse,train_corrcoef))


# VALIDATION PREDICTIONS
val_outputs = []
for val_sample in val_sequences:
    evalu = qrnn.evaluate(params[1:], val_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    val_outputs += [ypredi[-Nwout:]]
val_outputs = np.array(val_outputs)
val_rmse = rmse(val_outputs.flatten(),val_targets.flatten())
val_corrcoef = np.corrcoef(np.array([val_outputs.flatten(),val_targets.flatten()]), rowvar=True)[0,1]
print('Validation RMSE = %8.5f     R = %8.5f' %(val_rmse,val_corrcoef))


# TEST PREDICTIONS
test_outputs = []
for test_sample in test_sequences:
    evalu = qrnn.evaluate(params[1:], test_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    test_outputs += [ypredi[-Nwout:]]
test_outputs = np.array(test_outputs)
test_rmse = rmse(test_outputs.flatten(),test_targets.flatten())
test_corrcoef = np.corrcoef(np.array([test_outputs.flatten(),test_targets.flatten()]), rowvar=True)[0,1]
print('Test       RMSE = %8.5f     R = %8.5f' %(test_rmse,test_corrcoef))



# FILL TEST PREDICTIONS
filltest_outputs = []
for filltest_sample in filltest_sequences:
    evalu = qrnn.evaluate(params[1:], filltest_sample)
    ypredi = np.array([params[0]]*len(evalu)) + evalu
    filltest_outputs += [ypredi[-Nwout:]]
filltest_outputs = np.array(filltest_outputs)
filltest_rmse = rmse(filltest_outputs.flatten(),filltest_targets.flatten())
filltest_corrcoef = np.corrcoef(np.array([filltest_outputs.flatten(),filltest_targets.flatten()]), rowvar=True)[0,1]
print('Com. test  RMSE = %8.5f     R = %8.5f' %(filltest_rmse,filltest_corrcoef))


command = "seff %s" %job_id[-5:]
result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
resultstr = result.stdout
info = {}
for line in resultstr.splitlines():
    key, value = line.split(": ")
    info[key] = value


try:
    logdf = pd.read_csv('log_'+str(job_id)+'.csv')
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
        assert np.abs(train_rmse-log_data['RMSE tra']) < 1.e-4, 'Calculated TR RMSE does not match the one in the main program'
    except AssertionError as msg:
        print(msg, train_rmse, log_data['RMSE tra'])
    try:
        assert np.abs(val_rmse-log_data['RMSE val']) < 1.e-4, 'Calculated VAL RMSE does not match the one in the main program'
    except AssertionError as msg:
        print(msg, val_rmse, log_data['RMSE val'])
    try:
        logdf['RMSE tes'] = [test_rmse]
    except:
        print('Could not write RMSE tes on log file')
    try:
        logdf['RMSE ftes'] = [filltest_rmse]
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
    
except:
    print('Could not open log file')


try:
    historicdf = pd.read_csv('historic_emcz2_santafe.csv')
    historicdf.loc[len(historicdf)] = logdf.iloc[0]
    historicdf.to_csv('historic_emcz2_santafe.csv', index=False)
except:
    try:
        logdf.to_csv('historic_emcz2_santafe.csv', index=False)
    except:
        print('Could not save historic_emcz2_santafe')


plt.close('all')
plt.figure(1,figsize=(100,5))
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

plt.savefig(fname_p.split('/')[-1][:-4]+'.png')
