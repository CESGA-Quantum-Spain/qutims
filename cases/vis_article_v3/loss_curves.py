# -*- coding: utf-8 -*-
"""
Created on 
@author: jdviqueira

Script for loss curves plots.
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from qurecnets.emc import EMCZ2
import random

from utils_vis.collect_loss_curves import conv_curves, mov_avg, make_single_loss_plots, make_avg_loss_plots, mark_best_val, mark_avg_best_val

plt.rcParams.update({'font.size': 11})

mav_size = 50

##### LINESTYLES #######################################################################
# quantum
format_tr_bash0  = {'linestyle':'solid', 'color':'blue', 'linewidth':1.}
format_val_bash0 = {'linestyle':'dashed', 'color':'blue', 'linewidth':1.}
format_marker_bash0 = {'markerfacecolor':'blue','markeredgewidth':0.5,'markeredgecolor':'black'}

format_tr_bash0k  = {'linestyle':'solid', 'color':'limegreen', 'linewidth':0.5}
format_val_bash0k = {'linestyle':'dashed', 'color':'limegreen', 'linewidth':0.5}
format_marker_bash0k = {'markerfacecolor':'limegreen', 'markersize':2.5,'markeredgewidth':0.5,'markeredgecolor':'black'}

formats_tr_bash100 = ({'linestyle':'solid', 'color':'gold', 'linewidth':0.5},
                      {'facecolor':'gold', 'alpha':0.2})
formats_val_bash100 = ({'linestyle':(5, (10, 3)), 'color':'gold', 'linewidth':0.5},
                       {'facecolor':'gold', 'alpha':0.2})
format_marker_bash100 = {'markerfacecolor':'gold','markeredgewidth':0.5,'markeredgecolor':'black'}

formats_tr_bash1k = ({'linestyle':'solid', 'color':'darkorange', 'linewidth':0.5},
                     {'facecolor':'darkorange', 'alpha':0.2})
formats_val_bash1k = ({'linestyle':(5, (10, 3)), 'color':'darkorange', 'linewidth':0.5},
                      {'facecolor':'darkorange', 'alpha':0.2})
format_marker_bash1k = {'markerfacecolor':'darkorange','markeredgewidth':0.5,'markeredgecolor':'black'}

formats_tr_bash10k = ({'linestyle':'solid', 'color':'red', 'linewidth':0.5},
                     {'facecolor':'red', 'alpha':0.2})
formats_val_bash10k = ({'linestyle':(5, (10, 3)), 'color':'red', 'linewidth':0.5},
                      {'facecolor':'red', 'alpha':0.2})
format_marker_bash10k = {'markerfacecolor':'red','markeredgewidth':0.5,'markeredgecolor':'black'}

# classic
format_tr_vanillarnn  = {'linestyle':'solid', 'color':'black', 'linewidth':0.5}
format_val_vanillarnn = {'linestyle':'dashed', 'color':'black', 'linewidth':0.5}

format_tr_lstm  = {'linestyle':'solid', 'color':'slategrey', 'linewidth':0.5}
format_val_lstm = {'linestyle':'dashed', 'color':'slategrey', 'linewidth':0.5}

format_tr_gru  = {'linestyle':'solid', 'color':'palevioletred', 'linewidth':0.5}
format_val_gru = {'linestyle':'dashed', 'color':'palevioletred', 'linewidth':0.5}
#######################################################################################



plt.close('all')
fig, ax = plt.subplots(2,3, figsize=(2*5.65,2*3.0))
#fig = plt.figure(1, figsize=(10,5))

ppath = '../'


### Case (a) 'dectriang'
# bash0
path = ppath+'dectriang/bash0/'
job_id = '13192'
ax[0,0] = make_single_loss_plots(ax[0,0],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,0] = mark_best_val(ax[0,0],job_id,path=path, preffix2='log_', column='Nepochs', format=format_marker_bash0)

# bash0k
path = ppath+'dectriang/bash0k/'
job_id = '13626'
ax[0,0] = make_single_loss_plots(ax[0,0],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,0] = mark_best_val(ax[0,0],job_id,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash0k)

# bash100
path = ppath+'dectriang/bash100/'
job_ids = ['16815','18095','32281','32282','37843','37844','37845','37846']
ax[0,0] = make_avg_loss_plots(ax[0,0],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,0] = mark_avg_best_val(ax[0,0],job_ids,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash100)

# bash1k
path = ppath+'dectriang/bash1k/'
job_ids = ['13300','13304','13305','13306','13406','13407','13408','13409']
ax[0,0] = make_avg_loss_plots(ax[0,0],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,0] = mark_avg_best_val(ax[0,0],job_ids,path=path, preffix2='log_',  format=format_marker_bash1k)

# bash10k
path = ppath+'dectriang/bash10k/'
job_ids = ['13302','13307','13308','13309','13410','13411','13412','13413']
ax[0,0] = make_avg_loss_plots(ax[0,0],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,0] = mark_avg_best_val(ax[0,0],job_ids,path=path, preffix2='log_', format=format_marker_bash10k)



ax[0,0].set_yscale('log')
#ax[0,0].set_xlabel('epochs')
ax[0,0].grid(which='both')
ax[0,0].set_ylabel('Loss (RMSE)')
ax[0,0].set_title('(a) Dec. triang. ', loc='right', y=0.85, x=1.0, fontsize='small')
ax[0,0].set_xlim(0,2000)
ax[0,0].set_xticks([0,500,1000,1500,2000])
ax[0,0].set_xticklabels(['' for i in range(5)])
ax[0,0].set_yticks([0.01,0.03,0.06,0.1,0.2])
ax[0,0].set_yticklabels(['0.01','0.03','0.06','0.1','0.2'])



### Case (b) 'vdp1'
# bash0
path = ppath+'vdp1/bash0/'
job_id = '13227'
ax[0,1] = make_single_loss_plots(ax[0,1],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,1] = mark_best_val(ax[0,1],job_id,path=path, preffix2='log_', column='Nepochs', format=format_marker_bash0)

# bash0k
path = ppath+'vdp1/bash0k/'
job_id = '13627'
ax[0,1] = make_single_loss_plots(ax[0,1],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,1] = mark_best_val(ax[0,1],job_id,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash0k)

# bash100
path = ppath+'bash100/'
job_ids = ['16872','18092','32283','32284','32294','37847','37848','37859']
ax[0,1] = make_avg_loss_plots(ax[0,1],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,1] = mark_avg_best_val(ax[0,1],job_ids,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash100)

# bash1k
path = ppath+'vdp1/bash1k/'
job_ids = ['13312','13313','13314','13315','13414','13415','13416','13417']
ax[0,1] = make_avg_loss_plots(ax[0,1],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,1] = mark_avg_best_val(ax[0,1],job_ids,path=path, preffix2='log_',  format=format_marker_bash1k)

# bash10k
path = ppath+'vdp1/bash10k/'
job_ids = ['13316','13317','13318','13319','13418','13419','13420','13421']
ax[0,1] = make_avg_loss_plots(ax[0,1],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,1] = mark_avg_best_val(ax[0,1],job_ids,path=path, preffix2='log_', format=format_marker_bash10k)


#ax[0,1].ticklabel_format(style='plain', useOffset=True)
ax[0,1].set_yscale('log')
#ax[0,1].set_xlabel('epochs')
ax[0,1].grid(which='both')
#ax[0,1].set_ylabel('Loss')
ax[0,1].set_yticks([0.07,0.1,0.14,0.2,0.3])
ax[0,1].set_yticklabels(['0.07','0.1','0.14','0.2','0.3'])
ax[0,1].set_xticks([0,500,1000,1500,2000])
ax[0,1].set_xticklabels(['' for i in range(5)])
ax[0,1].set_xlim(0,2000)
ax[0,1].set_title('(b) Van der Pol 1v ', loc='right', y=0.85, x=1.0, fontsize='small')



### Case (c) 'vdp2'
# bash0
path = ppath+'vdp2/bash0/'
job_id = '13244'
ax[0,2] = make_single_loss_plots(ax[0,2],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,2] = mark_best_val(ax[0,2],job_id,path=path, preffix2='log_', column='Nepochs', format=format_marker_bash0)

# bash0k
path = ppath+'vdp2/bash0k/'
job_id = '13629'
ax[0,2] = make_single_loss_plots(ax[0,2],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,2] = mark_best_val(ax[0,2],job_id,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash0k)

# bash100
path = ppath+'vdp2/bash100/'
job_ids = ['32288','32289','32291','16877','18091','37856','37857','37858']
ax[0,2] = make_avg_loss_plots(ax[0,2],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,2] = mark_avg_best_val(ax[0,2],job_ids,path=path, preffix2='log_', column='Nepochs',  format=format_marker_bash100)

# bash1k
path = ppath+'vdp2/bash1k/'
job_ids = ['13320','13321','13322','13323','13422','13424','13425','13426']
ax[0,2] = make_avg_loss_plots(ax[0,2],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,2] = mark_avg_best_val(ax[0,2],job_ids,path=path, preffix2='log_',  format=format_marker_bash1k)

# bash10k
path = ppath+'vdp2/bash10k/'
job_ids = ['13324','13325','13326','13327','13427','13428','13430','13432']
ax[0,2] = make_avg_loss_plots(ax[0,2],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr=None,label_val=None,path=path,preffix='log_',suffix='.dat',title=None)
ax[0,2] = mark_avg_best_val(ax[0,2],job_ids,path=path, preffix2='log_', format=format_marker_bash10k)


#ax[0,2].ticklabel_format(style='plain', useOffset=True)
ax[0,2].set_yscale('log')
#ax[0,2].set_xlabel('epochs')
ax[0,2].grid(which='both')
ax[0,2].set_xlim(0,2000)
#ax[0,2].set_ylabel('Loss')
ax[0,2].set_yticks([0.04,0.06,0.1,0.2,0.3,0.4])
ax[0,2].set_yticklabels(['0.04','0.06','0.1','0.2','0.3','0.4'])
ax[0,2].set_xticks([0,500,1000,1500,2000])
ax[0,2].set_xticklabels(['' for i in range(5)])
ax[0,2].set_title('(c) Van der Pol 2v ', loc='right', y=0.85, x=1.0, fontsize='small')

#ax[0,2].legend(loc='lower right', bbox_to_anchor=(1.51, 0.15), ncol=1,fontsize=10)



### Case (d) 'santafe - del 1'
# bash0
path = ppath+'santafe/bash0/'
job_id = '34716'
ax[1,0] = make_single_loss_plots(ax[1,0],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,0] = mark_best_val(ax[1,0],job_id,path=path, format=format_marker_bash0)

# bash0k
path = ppath+'santafe/bash0k/'
job_id = '35346'
ax[1,0] = make_single_loss_plots(ax[1,0],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,0] = mark_best_val(ax[1,0],job_id,path=path, format=format_marker_bash0k)

# bash100
path = ppath+'santafe/bash100/'
job_ids = ['35399','35497','36067','36068','36838','36839','37779','37780']
ax[1,0] = make_avg_loss_plots(ax[1,0],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,0] = mark_avg_best_val(ax[1,0],job_ids,path=path, format=format_marker_bash100)

# bash1k
path = ppath+'santafe/bash1k/'
job_ids = ['35376','35500','36073','36074','36844','36845','37784','37785']
ax[1,0] = make_avg_loss_plots(ax[1,0],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,0] = mark_avg_best_val(ax[1,0],job_ids,path=path, format=format_marker_bash1k)

# bash10k
path = ppath+'santafe/bash10k/'
job_ids = ['35360','35503','36079','36080','36850','36851','37790','37791']
ax[1,0] = make_avg_loss_plots(ax[1,0],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,0] = mark_avg_best_val(ax[1,0],job_ids,path=path, format=format_marker_bash10k)


ax[1,0].set_yscale('log')
ax[1,0].set_xlabel('epochs')
ax[1,0].grid(which='both')
ax[1,0].set_ylabel('Loss (RMSE)')
ax[1,0].set_title('(d) Santa Fe $t_d=1 $', loc='right', y=0.85, x=1.0, fontsize='small')
ax[1,0].set_xlim(0,2000)
ax[1,0].set_yticks([0.03,0.04,0.06,0.08,0.1,0.13])
ax[1,0].set_yticklabels(['0.03','0.04','0.06','0.08','0.1','0.13'])


### Case (d) 'santafe - del 5'
# bash0
path = ppath+'santafe/bash0/'
job_id = '34362'
ax[1,1] = make_single_loss_plots(ax[1,1],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,1] = mark_best_val(ax[1,1],job_id,path=path, format=format_marker_bash0)

# bash0k
path = ppath+'santafe/bash0k/'
job_id = '35353'
ax[1,1] = make_single_loss_plots(ax[1,1],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,1] = mark_best_val(ax[1,1],job_id,path=path, format=format_marker_bash0k)

# bash100
path = ppath+'santafe/bash100/'
job_ids = ['35404','35498','36069','36070','36840','36841','37781','37782']
ax[1,1] = make_avg_loss_plots(ax[1,1],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,1] = mark_avg_best_val(ax[1,1],job_ids,path=path, format=format_marker_bash100)

# bash1k
path = ppath+'santafe/bash1k/'
job_ids = ['35403','35456','36075','36076','36846','36847','37786','37787']
ax[1,1] = make_avg_loss_plots(ax[1,1],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,1] = mark_avg_best_val(ax[1,1],job_ids,path=path, format=format_marker_bash1k)

# bash10k
path = ppath+'santafe/bash10k/'
job_ids = ['35400','35457','36081','36082','36852','36853','37792','37793']
ax[1,1] = make_avg_loss_plots(ax[1,1],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr=None,label_val=None,path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,1] = mark_avg_best_val(ax[1,1],job_ids,path=path, format=format_marker_bash10k)


ax[1,1].set_yscale('log')
ax[1,1].set_xlabel('epochs')
ax[1,1].grid(which='both')
#ax[1,1].set_ylabel('Loss')
ax[1,1].set_title('(d) Santa Fe $t_d=5 $', loc='right', y=0.85, x=1.0, fontsize='small')
ax[1,1].set_xlim(0,2000)
ax[1,1].set_yticks([0.05,0.06,0.08,0.1,0.15])
ax[1,1].set_yticklabels(['0.05','0.06','0.08','0.1','0.15'])



### Case (d) 'santafe - del 10'
# bash0
path = ppath+'santafe/bash0/'
job_id = '33711'
ax[1,2] = make_single_loss_plots(ax[1,2],job_id,mav_size,format_tr_bash0,format_val_bash0,label_tr='tra num.',label_val='val num.',path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,2] = mark_best_val(ax[1,2],job_id,path=path, label='best num.', format=format_marker_bash0)

# bash0k
path = ppath+'santafe/bash0k/'
job_id = '35354'
ax[1,2] = make_single_loss_plots(ax[1,2],job_id,mav_size,format_tr_bash0k,format_val_bash0k,label_tr='tra ana.',label_val='val ana.',path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,2] = mark_best_val(ax[1,2],job_id,path=path, label='best ana.', format=format_marker_bash0k)

# bash100
path = ppath+'santafe/bash100/'
job_ids = ['35455','35499','36071','36072','36842','36843','37783','37835']
ax[1,2] = make_avg_loss_plots(ax[1,2],job_ids,mav_size,formats_tr_bash100,formats_val_bash100,label_tr='tra $10^2$ sh',label_val='val $10^2$ sh',path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,2] = mark_avg_best_val(ax[1,2],job_ids,path=path, label='best $10^2$ sh', format=format_marker_bash100)

# bash1k
path = ppath+'santafe/bash1k/'
job_ids = ['35501','35502','36077','36078','36848','36849','37788','37789']
ax[1,2] = make_avg_loss_plots(ax[1,2],job_ids,mav_size,formats_tr_bash1k,formats_val_bash1k,label_tr='tra $10^3$ sh',label_val='val $10^3$ sh',path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,2] = mark_avg_best_val(ax[1,2],job_ids,path=path, label='best $10^3$ sh', format=format_marker_bash1k)

# bash10k
path = ppath+'santafe/bash10k/'
job_ids = ['35504','35505','36083','36084','36854','36855','37794','37795']
ax[1,2] = make_avg_loss_plots(ax[1,2],job_ids,mav_size,formats_tr_bash10k,formats_val_bash10k,label_tr='tra $10^4$ sh',label_val='val $10^4$ sh',path=path,preffix='loss_',suffix='.dat',title=None)
ax[1,2] = mark_avg_best_val(ax[1,2],job_ids,path=path, label='best $10^4$ sh', format=format_marker_bash10k)


ax[1,2].set_yscale('log')
ax[1,2].set_xlabel('epochs')
ax[1,2].grid(which='both')
#ax[1,2].set_ylabel('Loss')
ax[1,2].set_title('(d) Santa Fe $t_d=10 $', loc='right', y=0.85, x=1.0, fontsize='small')
ax[1,2].set_xlim(0,2000)
ax[1,2].set_yticks([0.08,0.09,0.1,0.12,0.14,0.16])
ax[1,2].set_yticklabels(['0.08','0.09','0.1','0.12','0.14','0.16'])



strokes_labels = ax[1,2].get_legend_handles_labels()

strokes = []
labels = []
for stroke, label in zip(*strokes_labels):
    if label not in labels:
        strokes.append(stroke)
        labels.append(label)
fig.legend(strokes, labels, loc='center right', bbox_to_anchor=(1.03, 0.52), ncol=1, fontsize=10)

plt.subplots_adjust(hspace=0.08)


plt.savefig('loss_curves_log_simp.pdf', bbox_inches='tight') #, dpi=600