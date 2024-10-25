# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Script for results' box plots.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

import pandas as pd


def add_rmse_csv(fname):
    table = pd.read_csv(fname)
    table = table.drop_duplicates(subset='jobID', keep='first')
    data = {'tra':list(table['RMSE tra']),
            'val':list(table['RMSE val']),
            'tes':list(table['RMSE tes']),
            'ftes':list(table['RMSE ftes'])}
    return data

def add_rmse_csv_filter(fname,column,criterium):
    table = pd.read_csv(fname)
    table = table.drop(table[table[column] != criterium].index)
    table = table.drop_duplicates(subset='jobID', keep='first')
    data = {'tra':list(table['RMSE tra']),
            'val':list(table['RMSE val']),
            'tes':list(table['RMSE tes']),
            'ftes':list(table['RMSE ftes'])}
    return data


def add_rmse_csv_2filter(fname,column1,criterium1,column2,criterium2):
    table = pd.read_csv(fname)
    table = table.drop(table[table[column1] != criterium1].index)
    table = table.drop(table[table[column2] != criterium2].index)
    table = table.drop_duplicates(subset='jobID', keep='first')
    data = {'tra':list(table['RMSE tra']),
            'val':list(table['RMSE val']),
            'tes':list(table['RMSE tes']),
            'ftes':list(table['RMSE ftes'])}
    return data


def add_text(ax,text):
    # Build a rectangle in axes coords
    left, width = .25, .5
    bottom, height = .25, .0
    right = left + width
    top = bottom + height
    ax.text(0.5 * (left + right), 0.5 * (bottom + top), text,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)
    ax.set_axis_off()
    return ax


def plot_one_dataset(dictionary,row,dataset,ylims,yticks):
    ax0 = fig.add_subplot(gs[row+1,0])
    ax1 = fig.add_subplot(gs[row+1,1])
    ax2 = fig.add_subplot(gs[row+1,2])
    ax3 = fig.add_subplot(gs[row+1,3])

    colors = ['tab:blue', 'tab:blue', 'tab:blue','tab:purple', 'gold', 'gold','gold']
    boxplot = ax0.boxplot([item['tra'] for item in ALL_RMSE.values()], patch_artist=True)
    #ax0.set_title('training', **title_format)
    ax0.set_xticklabels(['' for item in ALL_RMSE.keys()])
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    #ax0.set_ylim(ymin,ymax)
    ax0.set_ylabel('%s\nRMSE' %dataset)
    boxplot = ax1.boxplot([item['val'] for item in ALL_RMSE.values()], patch_artist=True)
    #ax1.set_title('validation', **title_format)
    ax1.set_xticklabels(['' for item in ALL_RMSE.keys()])
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    #ax1.set_yticks(yticks)
    #ax1.set_ylim(ymin,ymax)
    boxplot = ax2.boxplot([item['tes'] for item in ALL_RMSE.values()], patch_artist=True)
    #ax2.set_title('test', **title_format)
    ax2.set_xticklabels(['' for item in ALL_RMSE.keys()])
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    #ax2.set_yticks(yticks)
    #ax2.set_ylim(ymin,ymax)
    boxplot = ax3.boxplot([item['ftes'] for item in ALL_RMSE.values()], patch_artist=True)
    #ax3.set_title('full test', **title_format)
    ax3.set_xticklabels(['' for item in ALL_RMSE.keys()])
    #ax3.set_yticks(yticks)
    #ax3.set_ylim(ymin,ymax)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    for ax in [ax0,ax1,ax2,ax3]:
        ax.set_ylim(*ylims)
        ax.set_yticks(yticks)
        ax.set_yticklabels(['' for it in yticks])
    ax0.set_yticklabels([str(it) for it in yticks])

    return ax0,ax1,ax2,ax3


def best_qrnn_validation(fname):
    table = pd.read_csv(fname)
    min_row = table[table['RMSE val'] == table['RMSE val'].min()]
    return (min_row['RMSE tra'],min_row['RMSE val'], min_row['RMSE tes'], min_row['RMSE ftes'])


def best_qrnn_validation_filter(fname,column,criterium):
    table = pd.read_csv(fname)
    table = table.drop(table[table[column] != criterium].index)
    table = table.drop_duplicates(subset='jobID', keep='first')
    min_row = table[table['RMSE val'] == table['RMSE val'].min()]
    return (min_row['RMSE tra'],min_row['RMSE val'], min_row['RMSE tes'], min_row['RMSE ftes'])


def indicate_best_qrnn_validation(ax0,ax1,ax2,ax3,values,format=None):
    for value,ax in zip(values,[ax0,ax1,ax2,ax3]):
        ax.plot([4],[value],**format)
    return ax0,ax1,ax2,ax3


ppath = '../'


plt.close('all')
fig = plt.figure(1, figsize=(11,16))
gs = fig.add_gridspec(7,4, height_ratios=[0.2,1,1,1,1,1,1]) #

title_format = {'loc':'left', 'y':0.8, 'x':0.05, 'fontsize':'small'}

format_bval_indicator = {'marker':'_','color':'lime', 'markersize':10}

#yticks = np.linspace(0.,0.03,10)
#ymin,ymax = 0.0, 0.03

ax0 = fig.add_subplot(gs[0,0])
ax0 = add_text(ax0,'training')
ax1 = fig.add_subplot(gs[0,1])
ax1 = add_text(ax1,'validation')
ax2 = fig.add_subplot(gs[0,2])
ax2 = add_text(ax2,'test')
ax3 = fig.add_subplot(gs[0,3])
ax3 = add_text(ax3,'full test')


### Case (a) 'dectriang' ####################################################################
ALL_RMSE = {}
fname = ppath + 'dectriang/torch/historic_torch_dectriang.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_filter(fname,'model','VanillaRNN')
ALL_RMSE['LSTM'] = add_rmse_csv_filter(fname,'model','LSTMModel')
ALL_RMSE['GRU'] = add_rmse_csv_filter(fname,'model','GRUModel')
fname1 = ppath + 'dectriang/bash0/optimisations_info_dectriang.csv'
ALL_RMSE['QRNN'] = add_rmse_csv(fname1)
fname = ppath + 'dectriang/bash10k/optimisations_info_dectriang.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv(fname)
fname = ppath + 'dectriang/bash1k/optimisations_info_dectriang.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv(fname)
fname = ppath + 'dectriang/bash100/optimisations_info_dectriang.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv(fname)

ylims  = (0.,0.042)
yticks = [0.01,0.02,0.03,0.04]

ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,0,'(a) Decreasing triang.', ylims,yticks)
best_val_values = best_qrnn_validation(fname1)
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)

####################################################################################################




### Case (b) 'vdp1' ####################################################################
ALL_RMSE = {}
fname = ppath + 'vdp1/torch/historic_torch_vdp1.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_filter(fname,'model','VanillaRNN')
ALL_RMSE['LSTM'] = add_rmse_csv_filter(fname,'model','LSTMModel')
ALL_RMSE['GRU'] = add_rmse_csv_filter(fname,'model','GRUModel')
fname1 = ppath + 'vdp1/bash0/optimisations_info_vdp1.csv'
ALL_RMSE['QRNN'] = add_rmse_csv(fname1)
fname = ppath + 'vdp1/bash10k/optimisations_info_vdp1.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv(fname)
fname = ppath + 'vdp1/bash1k/optimisations_info_vdp1.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv(fname)
fname = ppath + 'vdp1/bash100/optimisations_info_vdp1.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv(fname)


ylims  = (0.068,0.142)
yticks = [0.08,0.10,0.12,0.14]
ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,1,'(b) Van der Pol 1v', ylims,yticks)
best_val_values = best_qrnn_validation(fname1)
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)
####################################################################################################




### Case (c) 'vdp2' ####################################################################
ALL_RMSE = {}
fname = ppath + 'vdp2/torch/historic_torch_vdp2.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_filter(fname,'model','VanillaRNN')
ALL_RMSE['LSTM'] = add_rmse_csv_filter(fname,'model','LSTMModel')
ALL_RMSE['GRU'] = add_rmse_csv_filter(fname,'model','GRUModel')
fname1 = ppath + 'vdp2/bash0/optimisations_info_vdp2.csv'
ALL_RMSE['QRNN'] = add_rmse_csv(fname1)
fname = ppath + 'vdp2/bash10k/optimisations_info_vdp2.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv(fname)
fname = ppath + 'vdp2/bash1k/optimisations_info_vdp2.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv(fname)
fname = ppath + 'vdp2/bash100/optimisations_info_vdp2.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv(fname)


ylims  = (0.01,0.12)
yticks = [0.02,0.04,0.06,0.08,0.10]
ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,2,'(c) Van der Pol 2v', ylims,yticks)
best_val_values = best_qrnn_validation(fname1)
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)
####################################################################################################




### Case (d) 'santafe - del 1' ####################################################################
ALL_RMSE = {}
fname = ppath + 'santafe/torch/historic_torch_santafe.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_2filter(fname,'model','VanillaRNN','datfile','data_santafe1.dat')
ALL_RMSE['LSTM'] = add_rmse_csv_2filter(fname,'model','LSTMModel','datfile','data_santafe1.dat')
ALL_RMSE['GRU'] = add_rmse_csv_2filter(fname,'model','GRUModel','datfile','data_santafe1.dat')
fname1 = ppath + 'santafe/bash0/historic_emcz2_santafe.csv'
ALL_RMSE['QRNN'] = add_rmse_csv_filter(fname1,'datfile','data_santafe1.dat')
fname = ppath + 'santafe/bash10k/historic_emcz2_santafe_sn10000.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv_filter(fname,'datfile','data_santafe1.dat')
fname = ppath + 'santafe/bash1k/historic_emcz2_santafe_sn1000.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv_filter(fname,'datfile','data_santafe1.dat')
fname = ppath + 'santafe/bash100/historic_emcz2_santafe_sn100.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv_filter(fname,'datfile','data_santafe1.dat')


ylims  = (0.01,0.08)
yticks = [0.02,0.04,0.06,0.08]
ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,3,'(d) Santa Fe - $t_d=1$', ylims,yticks)
best_val_values = best_qrnn_validation_filter(fname1,'datfile','data_santafe1.dat')
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)
####################################################################################################




### Case (d) 'santafe - del 5' ####################################################################
ALL_RMSE = {}
fname = ppath + 'santafe/torch/historic_torch_santafe.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_2filter(fname,'model','VanillaRNN','datfile','data_santafe5.dat')
ALL_RMSE['LSTM'] = add_rmse_csv_2filter(fname,'model','LSTMModel','datfile','data_santafe5.dat')
ALL_RMSE['GRU'] = add_rmse_csv_2filter(fname,'model','GRUModel','datfile','data_santafe5.dat')
fname1 = ppath + 'santafe/bash0/historic_emcz2_santafe.csv'
ALL_RMSE['QRNN'] = add_rmse_csv_filter(fname1,'datfile','data_santafe5.dat')
fname = ppath + 'santafe/bash10k/historic_emcz2_santafe_sn10000.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv_filter(fname,'datfile','data_santafe5.dat')
fname = ppath + 'santafe/bash1k/historic_emcz2_santafe_sn1000.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv_filter(fname,'datfile','data_santafe5.dat')
fname = ppath + 'santafe/bash100/historic_emcz2_santafe_sn100.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv_filter(fname,'datfile','data_santafe5.dat')


ylims  = (0.02,0.26)
yticks = [0.05,0.10,0.15,0.20,0.25]
ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,4,'(d) Santa Fe - $t_d=5$', ylims,yticks)
best_val_values = best_qrnn_validation_filter(fname1,'datfile','data_santafe5.dat')
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)
####################################################################################################



### Case (d) 'santafe - del 10' ####################################################################
ALL_RMSE = {}
fname = ppath + 'santafe/torch/historic_torch_santafe.csv'
ALL_RMSE['cRNN'] = add_rmse_csv_2filter(fname,'model','VanillaRNN','datfile','data_santafe10.dat')
ALL_RMSE['LSTM'] = add_rmse_csv_2filter(fname,'model','LSTMModel','datfile','data_santafe10.dat')
ALL_RMSE['GRU'] = add_rmse_csv_2filter(fname,'model','GRUModel','datfile','data_santafe10.dat')
fname1 = ppath + 'santafe/bash0/historic_emcz2_santafe.csv'
ALL_RMSE['QRNN'] = add_rmse_csv_filter(fname1,'datfile','data_santafe10.dat')
fname = ppath + 'santafe/bash10k/historic_emcz2_santafe_sn10000.csv'
ALL_RMSE['Q. $10^4$'] = add_rmse_csv_filter(fname,'datfile','data_santafe10.dat')
fname = ppath + 'santafe/bash1k/historic_emcz2_santafe_sn1000.csv'
ALL_RMSE['Q. $10^3$'] = add_rmse_csv_filter(fname,'datfile','data_santafe10.dat')
fname = ppath + 'santafe/bash100/historic_emcz2_santafe_sn100.csv'
ALL_RMSE['Q. $10^2$'] = add_rmse_csv_filter(fname,'datfile','data_santafe10.dat')


ylims  = (0.029,0.165)
yticks = [0.04,0.08,0.12,0.16]
ax0,ax1,ax2,ax3 = plot_one_dataset(ALL_RMSE,5,'(d) Santa Fe - $t_d=10$', ylims,yticks)
best_val_values = best_qrnn_validation_filter(fname1,'datfile','data_santafe10.dat')
ax0,ax1,ax2,ax3 = indicate_best_qrnn_validation(ax0,ax1,ax2,ax3, best_val_values, format_bval_indicator)

for ax in [ax0,ax1,ax2,ax3]:
    ax.set_xticklabels(labels=ALL_RMSE.keys(),rotation=30)

ax0.plot([1],[-1], marker='s', linewidth=0, color='tab:blue', alpha=0.3, label='classical')
ax0.plot([1],[-1], marker='s', linewidth=0, color='tab:purple', alpha=0.3, label='quantum (noiseless)')
ax0.plot([1],[-1], marker='s', linewidth=0, color='gold', alpha=0.3, label='quantum (noisy)')


plt.subplots_adjust(hspace=0.08,wspace=0.08)

strokes_labels = ax0.get_legend_handles_labels()

strokes = []
labels = []
for stroke, label in zip(*strokes_labels):
    if label not in labels:
        strokes.append(stroke)
        labels.append(label)
fig.legend(strokes, labels, loc='center right', bbox_to_anchor=(1.1, 0.2), ncol=1, fontsize=10)


plt.savefig('box_plots_compar.pdf', bbox_inches='tight')