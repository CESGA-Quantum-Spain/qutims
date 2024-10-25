# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira
"""

#### LOAD MODULES #######################################################################
import numpy as np
import random

import sys
#sys.path.append('') # append sys path to main directory

from qurecnets.models import CZladder2p1, encodeP2
from qurecnets.loss_fs import mse
from qurecnets.readout import expectZ
from qurecnets.emc import emulator


#import pennylane as qml
from pennylane import numpy as npp
from pennylane.optimize import AdamOptimizer #, GradientDescentOptimizer, AdagradOptimizer

from time import time, sleep
import csv
import pandas as pd


job_id_init = sys.argv[1]
job_id = sys.argv[2]

#"""
####### DASK ###############################
from dask import delayed
from distributed import Client, wait
from dask_cluster import create_dask_client

print('modules imported')

info = "./scheduler_info_%s.json" %job_id
client = create_dask_client(info)

print(client)
############################################
#"""



#### DATA PREPARATION  ##################################################################
fname = 'data_santafe10.dat' # INPUT
data = np.loadtxt('../'+fname)

# QRNN size and layers
nE = 1; nM = 2; nL = 5; nx = 3

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
train_sequences = np.array([sequences[i] for i in tr_samples])
train_targets   = np.array([targets[i] for i in tr_samples])

# VALIDATION DATA
val_sequences   = np.array([sequences[i] for i in val_samples])
val_targets     = np.array([targets[i] for i in val_samples])
#####################################################################

fname_init = '../bash0/param0_'+str(job_id_init)+'.dat'
param0 = np.loadtxt(fname_init)#[:,0]

##rnd_seed = int(job_id) # !!!
##np.random.seed(rnd_seed)
##param0 = np.concatenate(([0.],2.*np.pi*np.random.random(2*(nE*nx + nL*(nE+nM))+1*nE)))
#np.random.seed(None)

##np.savetxt('param0_'+str(job_id)+'.dat', param0)
np.savetxt('param0_'+str(job_id_init)+'_'+str(job_id)+'.dat', param0)

##print('seed for init ', rnd_seed)
print('len param0 ', len(param0))
print('nT=%i; nE=%i; nM=%i; nL=%i; nx=%i' %(nT,nE,nM,nL,nx))


##### DEFINE CLASS and COST FUNCTIONS ########
class train_qrnn(emulator, encodeP2, CZladder2p1, expectZ, mse):
    """Class for QRNN training. Inherits classes for emulation. Same constructor.
    Adds scores for training and validation datasets: MSE of all involved points (last Nwout points of all train/val windows).
    """

    def loss_train(self, params,x,y,shots=0):
        tr_losses = []
        for i in range(Ntr):
            evalu = self.evaluate(params[1:],x[i], shots=shots)
            yb = np.array([params[0]]*len(evalu)) + evalu
            tr_losses += [(1./Nwout)*sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:], y[i][-Nwout:])])]
        Etr = np.mean(tr_losses)
        return Etr

    def validate(self, params,x,y, shots=0):
        val_losses = []
        for i in range(Nval):
            evalu = self.evaluate(params[1:],x[i], shots=shots)
            yb = np.array([params[0]]*len(evalu)) + evalu
            val_losses += [(1./Nwout)*sum([(ybi-yi)**2 for (ybi,yi) in zip(yb[-Nwout:], y[i][-Nwout:])])]
        Eval = np.mean(val_losses)
        return Eval

    def __model__(self):
        return 'EMCZ2'
####################################



#### OPTIMIZATION  ############################################################
logfile = 'log_'+job_id+'.csv'
lossfile = 'loss_'+job_id+'.dat'


nshots = 0
LR = 0.001 # Learning Rate
epochs = 2000
shuffle = True
opt = AdamOptimizer(stepsize=LR)


try: client_df=str(client)
except: client_df=None
try: optimizer_df=str(opt)
except: optimizer_df=None


param0 = npp.array(param0, requires_grad=True)
params = param0

best_Eval = 50. # high value
Ntr = len(train_sequences); Nval = len(val_sequences)
print('Begin optimization')

checkpoints = [499,999,1499,1999]

if shuffle:
    # for reproducibility, the shuffle must be the same as in the origin optimisation
    shuffle_repeat = np.loadtxt('../bash0/.shuffle_indices_'+str(job_id_init)+'.csv', dtype=np.int32, delimiter=',')


### CREATE TRAINING QRNN OBJECT
trqrnn = train_qrnn(nT,nE,nM,nL,nx, shots=nshots, rseed=None)
pinorm = 1. # normalization of input range

logdf = pd.DataFrame([{'jobID_init': job_id_init, 'jobID':job_id,  'cores':None, 'memory':None, 'CPU time':None, 'time':None, 'client': client_df,
                      'datfile':fname, 'datdist seed':ddist_seed, 'Npoints':len(x_data), 'Nwout':Nwout,
                      'model':trqrnn.__model__(), 'nT':nT, 'nE':nE, 'nM':nM, 'nL':nL, 'nx':nx, 'Nparam':len(param0),
                      'rnd seed':None, 'optimizer':optimizer_df, 'Nepochs':epochs, 'grad':True, 'shuffle':shuffle, 'LR':LR, 'Nshots':nshots}])

with open(logfile, 'w') as f:
    logdf.to_csv(f, index=False)

for it in range(epochs):
    
    if shuffle:
        #indices = np.random.permutation(Ntr)
        indices = shuffle_repeat[it]

    else:
        indices = list(range(Ntr))

    X_shuffled = train_sequences[indices]
    y_shuffled = train_targets[indices]

    t0 = time()
    for xsample, ysample in zip(X_shuffled, y_shuffled):
        xsample = pinorm*npp.array(xsample, requires_grad=False)  # CHANGE: input data renormalization: interval [-pinorm,+pinorm]
        ysample = npp.array(ysample, requires_grad=False)

        gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout=Nwout, client=client)
        params = npp.array(opt.apply_grad(gradient, params))
    t1 = time()
    #print(t1-t0)


    Etr = trqrnn.loss_train(params, pinorm*train_sequences, train_targets)
    Eval = trqrnn.validate(params, pinorm*val_sequences, val_targets)

    with open(lossfile, 'a') as f:
        f.write('==> Loss (RMSE)    %4i  %10.6f %10.6f' %(it, np.sqrt(Etr),np.sqrt(Eval)))
        f.write('\n')
    #print('==> Loss (RMSE)    %4i  %10.6f %10.6f' %(it, np.sqrt(Etr),np.sqrt(Eval)))

    if Eval < best_Eval:
        np.savetxt('param_best_'+job_id+'.dat', params)
        best_Etr  = Etr
        best_Eval = Eval
        best_it = it
        best_gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout, client=client)
        np.savetxt('.grad_best_'+job_id+'.dat', best_gradient)
    
    if it in checkpoints:
        np.savetxt('.param_'+str(it)+'_'+job_id+'.dat', params)
        gradient = trqrnn.grad_BL_psr(params, xsample, ysample, Nwout, client=client)
        np.savetxt('.grad_'+str(it)+'_'+job_id+'.dat', best_gradient)

client.shutdown()


logdf['final epochs'] = [it+1]
logdf['epoch min VAL'] = [best_it+1]
logdf['Ncev/w'] = [len(param0)+1]
logdf['Ncev'] = [(len(param0)+1)*len(train_sequences)]
logdf['RMSE tra'] = [np.sqrt(best_Etr)]
logdf['RMSE val'] = [np.sqrt(best_Eval)]
logdf['RMSE tes'] = [None]
logdf['RMSE ftes'] = [None]
logdf['normalization'] = [pinorm]

with open(logfile, 'w') as f:
    logdf.to_csv(f, index=False)
