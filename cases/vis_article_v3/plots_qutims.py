# -*- coding: utf-8 -*-

"""
Created on 
@author: jdviqueira

Script for QUTIMS' results series visualization.
"""

import sys
sys.path.append('..')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
plt.rcParams.update({'font.size': 11})

import random
from qurecnets.emc import EMCZ2


from utils_vis.collect_results_qutims import data_collection




plt.close('all')
#fig, ax = plt.subplots(31,20, figsize=(2*5.8,2*3))
fig = plt.figure(1, figsize=(2*5.8,2*3))
gs = fig.add_gridspec(3,2,width_ratios=[3,2],height_ratios=[1,1,1])

ppath = '../'


##### (a) DECTRIANG #####################################################
nT=20; nIN=1
path = ppath+'dectriang/'
collection = data_collection(path,'data_dectriang_ab_100_1000p.dat')
collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=2)

model = EMCZ2(20,1,2,3,3)
predictions = collection.make_predictions(model,'13192','bash0/')

ax0 = fig.add_subplot(gs[0,0])
ax0 = collection.make_series_plots(ax0,*predictions)

del collection

xticks = np.arange(0,100,10)
ax0.set_xticks(xticks)
ax0.set_xticklabels(['' for i in range(len(xticks))])
ax0.set_xlim(0,100)
ax0.set_yticks(np.arange(-0.75,0.75+0.1,0.25))
ax0.set_yticklabels(['','-0.5','','0.0','','0.5',''])
ax0.set_ylim(-0.75,0.75)
ax0.grid(color='gainsboro')
ax0.set_title('(a) Decreasing triang. ', loc='right', y=0.8, x=1.0, fontsize='small')



##### (b) VDP1 #########################################################
nT=20; nIN=2
path = ppath+'vdp1/'
collection = data_collection(path,'data_vdp_mu_2_del_15_ab_100_1000p.dat')
#collection.nV = 2
collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=1)

model = EMCZ2(20,2,2,4,1)
predictions = collection.make_predictions(model,'13227','bash0/')

ax1 = fig.add_subplot(gs[1,0])
ax1 = collection.make_series_plots(ax1,*predictions)

del collection

xticks = np.arange(0,100,10)
ax1.set_xticks(xticks)
ax1.set_xticklabels(['' for i in range(len(xticks))])
ax1.set_xlim(0,100)
ax1.set_yticks(np.arange(-0.75,0.75+0.1,0.25))
ax1.set_yticklabels(['','-0.5','','0.0','','0.5',''])
ax1.set_ylim(-0.75,0.75)
ax1.grid(color='gainsboro')
ax1.set_title('(b) Van der Pol 1v ', loc='right', y=0.8, x=1.0, fontsize='small')


##### (c) VDP2 #########################################################
nT=20; nIN=2
path = ppath+'vdp2/'
collection = data_collection(path,'data_vdp_mu_1_3_del_5_16_ab_100_1000p.dat')
#collection.nV = 2
collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=0)

model = EMCZ2(20,2,3,5,3)
predictions = collection.make_predictions(model,'13244','bash0/')

ax2 = fig.add_subplot(gs[2,0])
ax2 = collection.make_series_plots(ax2,*predictions)

del collection

xticks = np.arange(0,100,10)
ax2.set_xticks(xticks)
#ax2.set_xticklabels(['' for i in range(len(xticks))])
ax2.set_xlim(0,100)
ax2.set_xlabel('$t$', loc='right', y=0.5)
ax2.xaxis.set_label_coords(1.0, -0.1)
ax2.set_yticks(np.arange(-0.75,0.75+0.1,0.25))
ax2.set_yticklabels(['','-0.5','','0.0','','0.5',''])
ax2.set_ylim(-0.75,0.75)
ax2.grid(color='gainsboro')
ax2.set_title('(c) Van der Pol 2v ', loc='right', y=0.8, x=1.0, fontsize='small')


##### (d) SANTAFE - del 1 ##############################################
nT=20; nIN=1
path = ppath+'santafe/'
collection = data_collection(path,'data_santafe1.dat')
job_id = '34716'

ddist_seed = 4
model = EMCZ2(20,1,2,5,3)

collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=ddist_seed)

predictions = collection.make_predictions(model,job_id,'bash0/')

ax3 = fig.add_subplot(gs[0,1])
ax3 = collection.make_series_plots(ax3,*predictions)

del collection


xmin = 1880; xmax = 1980
xticks = np.arange(xmin,xmax,10)
ax3.set_xticks(xticks)
ax3.set_xticklabels(['' for i in range(len(xticks))])
ax3.set_xlim(xmin,xmax)
ax3.set_yticks(np.arange(-0.5,0.75+0.1,0.25))
ax3.set_yticklabels(['-0.5','','0.0','','0.5',''])
ax3.set_ylim(-0.5,0.75)
ax3.grid(color='gainsboro')
ax3.set_title('(d) Santa Fe - $t_d=1$  ', loc='right', y=0.8, x=1.0, fontsize='small')



##### (d) SANTAFE - del 5 ##############################################
nT=20; nIN=1
path = ppath+'santafe/'
collection = data_collection(path,'data_santafe5.dat')
job_id = '34362'

ddist_seed = 4
model = EMCZ2(20,1,2,5,3)

collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=ddist_seed)

predictions = collection.make_predictions(model,job_id,'bash0/')

ax4 = fig.add_subplot(gs[1,1])
ax4 = collection.make_series_plots(ax4,*predictions)

del collection


xmin = 1880; xmax = 1980
xticks = np.arange(xmin,xmax,10)
ax4.set_xticks(xticks)
ax4.set_xticklabels(['' for i in range(len(xticks))])
ax4.set_xlim(xmin,xmax)
ax4.set_yticks(np.arange(-0.5,0.75+0.1,0.25))
ax4.set_yticklabels(['-0.5','','0.0','','0.5',''])
ax4.set_ylim(-0.5,0.75)
ax4.grid(color='gainsboro')
ax4.set_title('(d) Santa Fe - $t_d=5$  ', loc='right', y=0.8, x=1.0, fontsize='small')


##### (d) SANTAFE - del 10 ##############################################
nT=20; nIN=1
path = ppath+'santafe/'
collection = data_collection(path,'data_santafe10.dat')
job_id = '33711'

ddist_seed = 4
model = EMCZ2(20,1,2,5,3)

collection.load_dist_data(nT,nIN, Nwout=5, ddist_seed=ddist_seed)

predictions = collection.make_predictions(model,job_id,'bash0/')

ax5 = fig.add_subplot(gs[2,1])
ax5 = collection.make_series_plots(ax5,*predictions)

del collection

xmin = 1880; xmax = 1980
xticks = np.arange(xmin,xmax,10)
ax5.set_xticks(xticks)
ax5.set_xticklabels(xticks,rotation=30)
ax5.set_xlim(xmin,xmax)
ax5.set_xlabel('$t$', loc='right', y=0.5)
ax5.xaxis.set_label_coords(1.0, -0.1)
ax5.set_yticks(np.arange(-0.5,0.75+0.1,0.25))
ax5.set_yticklabels(['-0.5','','0.0','','0.5',''])
ax5.set_ylim(-0.5,0.75)
ax5.grid(color='gainsboro')
ax5.set_title('(d) Santa Fe - $t_d=10$ ', loc='right', y=0.8, x=1.0, fontsize='small')

plt.subplots_adjust(wspace=0.1)


strokes_labels = ax5.get_legend_handles_labels()

strokes = []
labels = []
for stroke, label in zip(*strokes_labels):
    if label not in labels:
        strokes.append(stroke)
        labels.append(label)

#print(strokes,labels)

fig.legend(strokes, labels, loc='center right', bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize=10)

plt.subplots_adjust(hspace=0.08)

plt.savefig('article_plots_qutims.pdf', bbox_inches='tight') #, dpi=300