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
import pandas as pd


def conv_curves(fname):
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



def mov_avg(x,y,size):
    # moving average with x-axis re-dimension
    xn = np.linspace(x[0],x[-1],len(x)-size)
    yn = np.array([np.mean(y[i:i+size]) for i in range(len(y)-size)])
    return xn,yn



def make_single_loss_plots(axis,job_id,mav_size,format_tr,format_val,label_tr=None,label_val=None,path='',preffix='loss_',suffix='.dat',title=None):
    x,y,z = conv_curves(path+preffix+str(job_id)+suffix)
    xn,yn = mov_avg(x,y,mav_size)
    _,zn  = mov_avg(x,z,mav_size)
    axis.plot(xn, yn, label=label_tr, **format_tr)
    axis.plot(xn, zn, label=label_val, **format_val)
    return axis



def make_avg_loss_plots(axis,job_ids,mav_size,formats_tr,formats_val,label_tr=None,label_val=None,path='',preffix='loss_',suffix='.dat',title=None):
    ys = [];  zs = []
    yns = []; zns = []
    for i,id in enumerate(job_ids):
        x,y,z = conv_curves(path+preffix+str(id)+suffix)
        ys.append(y)
        zs.append(z)
    ys = np.array(ys)
    zs = np.array(zs)

    ymean = np.mean(ys, axis=0); ystd = np.std(ys, axis=0)#/np.sqrt(len(ys))
    zmean = np.mean(zs, axis=0); zstd = np.std(zs, axis=0)#/np.sqrt(len(zs))

    xn,ymeann = mov_avg(x,ymean,mav_size); _,ystdn = mov_avg(x,ystd,mav_size)
    _ ,zmeann = mov_avg(x,zmean,mav_size); _,zstdn = mov_avg(x,zstd,mav_size)

    axis.fill_between(xn, ymeann-ystdn, ymeann+ystdn, label=None, **formats_tr[1])
    axis.plot(xn, ymeann, label=label_tr, **formats_tr[0])

    axis.fill_between(xn, zmeann-zstdn, zmeann+zstdn, label=None, **formats_val[1])
    axis.plot(xn, zmeann, label=label_val, **formats_val[0])

    axis.set_title(title)
    return axis


def mark_best_val(axis,job_id,format=None,label=None,column='epoch min VAL',path='',preffix1='log_',preffix2='loss_'):
    try:
        table = pd.read_csv(path+preffix1+job_id+'.csv')
    except:
        'Could not open table'
    x,_,val_loss = conv_curves(path+preffix2+str(job_id)+'.dat')
    epoch_min_val = np.argmin(val_loss) + 1
    min_val = np.min(val_loss)

    try:
        print(epoch_min_val == table[column].iloc[0])
    except:
        print('Could not open column')

    xm,zm = mov_avg(x,val_loss,50)
    min_val_xm_index = np.argmin(np.abs(xm-np.array([epoch_min_val]*len(xm))))
    min_val_avg = zm[min_val_xm_index]
    
    axis.plot([epoch_min_val],[min_val_avg], label=label, marker='^', linewidth=0., **format)
    return axis



def mark_avg_best_val(axis,job_ids,format=None,label=None,column='epoch min VAL',path='',preffix1='log_',preffix2='loss_',suffix1='.csv',suffix2='.dat'):
    epoch_min_vals = []
    min_vals = []
    x = []
    z = []

    print(format)

    for i,id in enumerate(job_ids):
        try:
            table = pd.read_csv(path+preffix1+id+'.csv')
        except:
            'Could not open table'
        xi,_,val_loss = conv_curves(path+preffix2+str(id)+'.dat')
        epoch_min_val = np.argmin(val_loss) + 1
        min_val = np.min(val_loss)
        x.append(xi)
        z.append(val_loss)

        try:
            print(id, epoch_min_val == table[column].iloc[0])
        except:
            print('Could not open column')
        
        epoch_min_vals.append(epoch_min_val)
        min_vals.append(min_val)

    epoch_min_vals = np.array(epoch_min_vals)
    min_vals = np.array(min_vals)
    x = np.array(x); z = np.array(z)
    
    # we plot only the position of the average minimum validation, not teh value itself, because of the moving average #
    epoch_min_val_mean = np.mean(epoch_min_vals)
    epoch_min_val_sem = np.std(epoch_min_vals)/np.sqrt((len(epoch_min_vals)-1))
    xm,zm = mov_avg(x[0],np.mean(z, axis=0),50)
    min_val_xm_index = np.argmin(np.abs(xm-np.array([epoch_min_val_mean]*len(xm))))
    min_val_avg = zm[min_val_xm_index]
    # #

    axis.errorbar(x=[epoch_min_val_mean],y=[min_val_avg], xerr=[epoch_min_val_sem],yerr=None, label=label, marker='^', linewidth=0., elinewidth=0.5, ecolor='grey',capsize=1, capthick=0.5, **format)
    return axis