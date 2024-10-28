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
from qurecnets.emc import EMCZ2
import random
import pandas as pd

import os




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
        if self.nV==1 and self.nIN>self.nV:
            self.x_data = np.column_stack((self.data[:,1],)*self.nIN)
        #else:
        #    print('No known data-shape was found')
        
        nsamples = int(len(self.data[:,1])/self.nT)
        times = self.data[:,0].reshape(nsamples,self.nT)
        sequences = self.x_data.reshape(nsamples,self.nT,self.nIN)
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
        train_sequences = sequences[tr_samples]     #*1
        train_targets   = targets[tr_samples]       #*2
        #1: shape(tr_nsamples,nT,nV);  2: shape(tr_nsamples,Nwout)

        # VALIDATION DATA
        val_times       = times[val_samples]        #*1
        val_tartimes    = tartimes[val_samples]     #*2
        val_tarseq      = tarseq[val_samples]       #*1
        val_sequences   = sequences[val_samples]    #*1
        val_targets     = targets[val_samples]      #*2
        #1: shape(val_nsamples,nT,nV);  2: shape(val_nsamples,Nwout)

        # TEST DATA
        test_times      = times[trval_nsamples:]        #*1
        test_tartimes   = tartimes[trval_nsamples:]     #*2
        test_tarseq     = tarseq[trval_nsamples:]       #*1
        test_sequences  = sequences[trval_nsamples:]    #*1
        test_targets    = targets[trval_nsamples:]      #*2
        #1: shape(test_nsamples,nT,nV);  2: shape(test_nsamples,Nwout)

        # TEST DATA FILLING ALL POINTS
        trval_npoints = trval_nsamples * self.nT
        ts_npoints = self.Npoints - trval_npoints
        filltest_times  = np.array([self.data[:,0][trval_npoints+Nwout*i:trval_npoints+Nwout*(i+1)] for i in range(ts_npoints//Nwout)])                                 #*4
        filltest_sequences = np.array([self.x_data[trval_npoints-self.nT+Nwout+Nwout*i:trval_npoints-self.nT+Nwout+Nwout*i+self.nT] for i in range(ts_npoints//Nwout)]) #*3
        filltest_targets = np.array([self.data[:,-1][trval_npoints+Nwout*i:trval_npoints+Nwout*i+Nwout] for i in range(ts_npoints//Nwout)])                             #*4
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



    def recover_ddist_seed_from_log(self,job_id,path=''):
        logtable = pd.read_csv(path+'log_'+job_id+'.csv')
        return logtable['datdist seed'].iloc[0]
    


    def recover_emcz2_model_from_log(self,job_id,path=''):
        logtable = pd.read_csv(path+'log_'+job_id+'.csv')
        nT = logtable['nT']
        nE = logtable['nE']
        nM = logtable['nM']
        nL = logtable['nL']
        nx = logtable['nx']
        return EMCZ2(nT,nE,nM,nL,nx)



    def eval_linear(self,model,xin,params,shots=None):
        evalu = model.evaluate(params[1:],xin,shots=shots)
        yb = np.array([params[0]]*len(evalu)) + evalu
        return yb[-self.Nwout:]



    def make_predictions(self,model,job_id,path='',shots=None):
        fname_p = path+'param_best_%s.dat' %str(job_id)
        params = np.loadtxt(fname_p)
        train_outputs    = np.array([self.eval_linear(model,train_sample,params,shots=shots) for train_sample in self.train_sequences])
        val_outputs      = np.array([self.eval_linear(model,val_sample,params,shots=shots) for val_sample in self.val_sequences])
        test_outputs     = np.array([self.eval_linear(model,test_sample,params,shots=shots) for test_sample in self.test_sequences])
        filltest_outputs = np.array([self.eval_linear(model,filltest_sample,params,shots=shots) for filltest_sample in self.filltest_sequences])

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
            axis.plot(x, y, '.', color='red', markersize=2.5, label=r'$\overline{y}_{(t)}$ (tra)')
        for x,y in zip(self.val_tartimes,val_outputs):
            axis.plot(x, y, '.', color='blue', markersize=2.5, label=r'$\overline{y}_{(t)}$ (val)')
        for x,y in zip(self.filltest_times,filltest_outputs):
            axis.plot(x, y, '.', color='goldenrod', markersize=1.5, label=r'$\overline{y}_{(t)}$ (fte)')
        for x,y in zip(self.test_tartimes,test_outputs):
            axis.plot(x, y, '.', color='darkgoldenrod', markersize=2.5, label=r'$\overline{y}_{(t)}$ (tes)')

        axis.set_title(title)

        return axis
    


    def __del__(self):
        os.chdir(self.pwd)

    

"""
    def conv_curves(self,fname):
        iteration = []
        tra_loss  = []
        val_loss  = []
        with open(fname) as f:
            for line in f:
                if line[:3] == '==>':
                    sectors = line.split()
                    iteration.append(int(sectors[3]))
                    tra_loss.append(float(sectors[4]))
                    val_loss.append(float(sectors[5]))
        return iteration, tra_loss, val_loss



    def mov_avg(self,x,y,size):
        # moving average with x-axis re-dimension
        xn = np.linspace(x[0],x[-1],len(x)-size)
        yn = np.array([np.mean(y[i:i+size]) for i in range(len(y)-size)])
        return xn,yn


    
    def make_single_loss_plots(self,axis,job_id,mav_size,format_tr,format_val,label_tr,label_val,path='',preffix='loss_',suffix='.dat',title=None):
        x,y,z = self.conv_curves(path+preffix+str(job_id)+suffix)
        xn,yn = self.mov_avg(x,y,mav_size)
        _,zn  = self.mov_avg(x,z,mav_size)
        axis.plot(xn, yn, *format_tr, label=label_tr)
        axis.plot(xn, zn, *format_val, label=label_val)



    def make_avg_loss_plots(self,axis,job_ids,mav_size,format_tr,format_val,label_tr,label_val,path='',preffix='loss_',suffix='.dat',title=None)
        ys = [];  zs = []
        yns = []; zns = []
        for i,id in enumerate(ids):
            x,y,z = conv_curves('dectriang/bash1k/log_'+id+'.dat')
            ys.append(y)
            zs.append(z)
        ys = np.array(ys)
        zs = np.array(zs)

        ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
        zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

        xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
        _ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)

        axis.fill_between(xn, ymeann-ystdn, ymeann+ystdn, *format_tr[1], label=None)
        axis.plot(xn, ymeann, *format_tr[0], label=label_tr)

        axis.fill_between(xn, zmeann-zstdn, zmeann+zstdn, *format_val[1], label=None)
        axis.plot(xn, zmeann, *format_tr[0], label=label_val)

    axis.set_title(title)
    return axis

"""