# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Classes for loss curves results plots.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.optim as optim

import os


N = 5 # !!!trampa


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
    




class data_collection:
    """
    """
    def __init__(self,path,fname_x):
        self.pwd = os.getcwd()
        os.chdir(path)
        self.data = np.loadtxt(fname_x)
        
        self.nV = self.data.shape[1] - 2  # number of input variables
        self.Npoints = self.data.shape[0]
        
    

    def load_dist_data(self,nT,nIN,Nwout, TRVAL='80pc', VAL='20pc', ddist_seed=None):

        self.nT=nT; self.nIN=nIN
        self.Nwout = Nwout

        if type(TRVAL)==str:
            TRVAL = int(TRVAL[:-2])
        if type(VAL)==str:
            VAL = int(VAL[:-2])


        if self.nIN==self.nV:
            self.x_data = self.data[:,1:-1].reshape(self.Npoints,self.nV)
        elif self.nV==1 and self.nIN>self.nV:
            self.x_data = np.column_stack((self.data[:,1],)*self.nIN)
        else:
            print('No known data-shape was found')
        
        nsamples = int(len(self.data[:,1])/self.nT)
        times = self.data[:,0].reshape(nsamples,self.nT)
        sequences = self.x_data.reshape(nsamples,self.nT,self.nV)
        tarseq = self.data[:,-1].reshape(nsamples,self.nT)
        targets =   np.array([item[-Nwout:] for item in self.data[:,-1].reshape(nsamples,self.nT)])
        tartimes =  np.array([item[-Nwout:] for item in times])

        trval_nsamples = nsamples * int(TRVAL) // 100 # TRVAL% for training and validation
        val_nsamples = trval_nsamples * int(VAL) // 100 #VAL% for validation in training + validation set

        random.seed(ddist_seed)
        val_samples = np.sort(random.sample(range(trval_nsamples),val_nsamples)) # samples for validation
        tr_samples  = [i for i in range(trval_nsamples) if i not in val_samples] # samples for training


        # TRAINING DATA
        train_times     = times[tr_samples]         #*1
        train_tartimes  = tartimes[tr_samples]      #*2
        train_tarseq    = tarseq[tr_samples]        #*1
        train_sequences = torch.tensor(sequences[tr_samples]).float()     #*1
        train_targets   = torch.tensor(targets[tr_samples]).float()       #*2
        #1: shape(tr_nsamples,nT,nV);  2: shape(tr_nsamples,Nwout)

        # VALIDATION DATA
        val_times       = times[val_samples]        #*1
        val_tartimes    = tartimes[val_samples]     #*2
        val_tarseq      = tarseq[val_samples]       #*1
        val_sequences   = torch.tensor(sequences[val_samples]).float()    #*1
        val_targets     = torch.tensor(targets[val_samples]).float()      #*2
        #1: shape(val_nsamples,nT,nV);  2: shape(val_nsamples,Nwout)

        # TEST DATA
        test_times      = times[trval_nsamples:]        #*1
        test_tartimes   = tartimes[trval_nsamples:]     #*2
        test_tarseq     = tarseq[trval_nsamples:]       #*1
        test_sequences  = torch.tensor(sequences[trval_nsamples:]).float()  #*1
        test_targets    = torch.tensor(targets[trval_nsamples:]).float()    #*2
        #1: shape(test_nsamples,nT,nV);  2: shape(test_nsamples,Nwout)

        # TEST DATA FILLING ALL POINTS
        trval_npoints = trval_nsamples * self.nT
        ts_npoints = self.Npoints - trval_npoints
        filltest_times  = np.array([self.data[:,0][trval_npoints+Nwout*i:trval_npoints+Nwout*(i+1)] for i in range(ts_npoints//Nwout)])                                 #*4
        filltest_sequences = torch.tensor(np.array([self.x_data[trval_npoints-self.nT+Nwout+Nwout*i:trval_npoints-self.nT+Nwout+Nwout*i+self.nT] for i in range(ts_npoints//Nwout)])).float() #*3
        filltest_targets = torch.tensor(np.array([self.data[:,-1][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])).float()                             #*4
        #3: shape(ts_nsamples,nT,nV);  4: shape(ts_nsamples,Nwout)

        for var in ['nsamples', 'trval_nsamples', 'val_nsamples', 'ts_npoints']:
            setattr(self, var, locals()[var])
        for var in ['train_times', 'train_tartimes', 'train_tarseq', 'train_sequences', 'train_targets']:
            setattr(self, var, locals()[var])
        for var in ['val_times', 'val_tartimes', 'val_tarseq', 'val_sequences', 'val_targets']:
            setattr(self, var, locals()[var])
        for var in ['test_times', 'test_tartimes', 'test_tarseq', 'test_sequences', 'test_targets']:
            setattr(self, var, locals()[var])
        for var in ['filltest_times', 'filltest_sequences', 'filltest_targets']:
            setattr(self, var, locals()[var])
        
        return



    def make_predictions(self,job_id,path=''):
        N = self.Nwout
        modelfile = path+'state_best_%s.pt' %str(job_id)
        model = torch.load(modelfile)

        train_outputs = np.array([model(train_seq).detach().numpy() for train_seq in self.train_sequences])
        val_outputs   = np.array([model(val_seq).detach().numpy() for val_seq in self.val_sequences])
        test_outputs  = np.array([model(test_seq).detach().numpy() for test_seq in self.test_sequences])
        filltest_outputs = np.array([model(filltest_seq).detach().numpy() for filltest_seq in self.filltest_sequences])

        return train_outputs, val_outputs, test_outputs, filltest_outputs



    def make_series_plots(self,axis,train_outputs, val_outputs, test_outputs, filltest_outputs, title=None):
            for x,y in zip(self.train_times,self.train_sequences):
                axis.plot(x,y, color='tab:orange', linestyle='dashed', linewidth=0.5, alpha=0.75,  label=r'$x_{(t)}$ (tra)')
            for x,y in zip(self.val_times,self.val_sequences):
                axis.plot(x,y, color='tab:blue', linestyle='dashed',  linewidth=0.5, alpha=0.75,  label=r'$x_{(t)}$ (val)')
            for x,y in zip(self.test_times,self.test_sequences):
                axis.plot(x,y, color='tab:green', linestyle='dashed', linewidth=0.5, alpha=0.75,   label=r'$x_{(t)}$ (tes)')


            for x,y in zip(self.train_times,self.train_tarseq):
                axis.plot(x,y,'-', color='tab:orange', label=r'$y_{(t)}$ (tra)')
            for x,y in zip(self.val_times,self.val_tarseq):
                axis.plot(x,y,'-', color='tab:blue', label=r'$y_{(t)}$ (val)')
            for x,y in zip(self.test_times,self.test_tarseq):
                axis.plot(x,y,'-', color='tab:green', label=r'$y_{(t)}$ (tes)')

            for x,y in zip(self.train_tartimes,train_outputs):
                axis.plot(x, y, '.', color='red', markersize=2, label=r'$\overline{y}_{(t)}$ (tra)')
            for x,y in zip(self.val_tartimes,val_outputs):
                axis.plot(x, y, '.', color='blue', markersize=2, label=r'$\overline{y}_{(t)}$ (val)')
            for x,y in zip(self.filltest_times,filltest_outputs):
                axis.plot(x, y, '.', color='goldenrod', markersize=1, label=r'$\overline{y}_{(t)}$ (fte)')
            for x,y in zip(self.test_tartimes,test_outputs):
                axis.plot(x, y, '.', color='darkgoldenrod', markersize=2, label=r'$\overline{y}_{(t)}$ (tes)')
    
            axis.set_title(title)

            return axis
    


    def __del__(self):
        os.chdir(self.pwd)