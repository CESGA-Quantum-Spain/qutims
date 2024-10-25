# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Script for Santa Fe box plots and comparison with https://www.nature.com/articles/s41534-023-00682-z.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import pandas as pd


def add_corr2_csv_filter(fname,column,criterium):
    table = pd.read_csv(fname)
    table = table.drop(table[table[column] != criterium].index)
    table = table.drop_duplicates(subset='jobID', keep='first')
    return np.array(list(table['R ftes']))**2


def add_corr2_csv_2filter(fname,column1,criterium1,column2,criterium2):
    table = pd.read_csv(fname)
    table = table.drop(table[table[column1] != criterium1].index)
    table = table.drop(table[table[column2] != criterium2].index)
    table = table.drop_duplicates(subset='jobID', keep='first')
    return np.array(list(table['R ftes']))**2


ppath = '../'


plt.close('all')
fig,ax = plt.subplots(1,3, figsize=(0.95*9,3))

def myplot(axis,delay,qrc_data):
    # Santa Fe - del
    ALL_CORR2 = {}
    fname = ppath + 'santafe/bash0/historic_emcz2_santafe.csv'
    ALL_CORR2['QRNN'] = add_corr2_csv_filter(fname,'datfile','data_santafe%i.dat' %delay)
    fname = ppath + 'santafe/bash10k/historic_emcz2_santafe_sn10000.csv'
    ALL_CORR2['Q. $10^4$'] = add_corr2_csv_filter(fname,'datfile','data_santafe%i.dat' %delay)
    fname = ppath + 'santafe/bash1k/historic_emcz2_santafe_sn1000.csv'
    ALL_CORR2['Q. $10^3$'] = add_corr2_csv_filter(fname,'datfile','data_santafe%i.dat' %delay)
    fname = ppath + 'santafe/bash100/historic_emcz2_santafe_sn100.csv'
    ALL_CORR2['Q. $10^2$'] = add_corr2_csv_filter(fname,'datfile','data_santafe%i.dat' %delay)

    colors = ['tab:purple', 'gold', 'gold','gold']
    boxplot = axis.boxplot([item for item in ALL_CORR2.values()], patch_artist=True)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    
    axis.set_xticks(range(-1,5))
    axis.set_xlim(-1.5,4.5)
    axis.set_xticklabels(labels=['QRC 0.3','QRC 10']+list(ALL_CORR2.keys()),rotation=30)
    axis.set_title('$t_d = %i$' %delay, ) #**title_format

    #axis.plot([0.],qrc_data[0], marker='o', markerfacecolor='black', markeredgecolor='black', markeredgewidth=0.1)
    #axis.plot([0.],qrc_data[1], marker='o', markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.1)
    axis.plot([0.],qrc_data[0], marker='d', alpha=0.5, markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.1, markersize=15)
    axis.plot([-1.],qrc_data[1], marker='p', alpha=0.5, markerfacecolor='green', markeredgecolor='black', markeredgewidth=0.1, markersize=15)
    #axis.set_yticks(np.arange(0.50,1.00,0.05))

    return axis



ax[0] = myplot(ax[0],1, [0.9578,0.9320])
ax[0].set_yticks(np.arange(0.90,1.00,0.025))
ax[0].set_ylabel('capacity')
ax[1] = myplot(ax[1],5, [0.7222,0.6656])
ax[1].set_yticks(np.arange(0.60,0.90,0.05))
ax[2] = myplot(ax[2],10, [0.5370,0.4866])
ax[2].set_yticks(np.arange(0.45,0.75,0.05))

plt.subplots_adjust(wspace=0.3)


plt.savefig('box_plots_santafe.pdf', bbox_inches='tight')