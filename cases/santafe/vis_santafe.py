# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:34:09 2023

@author: jdviqueira

USE CASE - vis: Visualization of Santa Fe series
https://web.cecs.pdx.edu/mcnames/DataSets/index.html
"""

import numpy as np
import matplotlib.pyplot as plt

fname = 'data_santafe_2000.dat'
data = np.loadtxt(fname)

#"""
length = len(data)
print(len(data))

plt.figure(figsize=(length/100.,4))
plt.plot(data, linewidth=0.5)
plt.xticks(list(plt.xticks()[0]) + [1450,1460,1470,1480,1490,1500,1510,1520,1530,1540,1550])
plt.grid()

plt.savefig(fname[:-4]+'.png')
#"""

"""
t = data[:,0]#[:100]
x0 = data[:,1]#[:100]
y = data[:,2]#[:100]

#plt.plot(t, x0, '.--', alpha=0.8)
#plt.plot(t, y, '.-')

plt.figure(1, figsize=(80,5))
plt.plot(t, x0, '.-', linewidth=0.5, alpha=0.8, label='x0')
plt.plot(t, y, '.-', linewidth=0.5, label='y')
plt.legend()
plt.savefig(fname[:-4]+'.png')

plt.figure(2, figsize=(7,7))
plt.plot(x0, y, '-', linewidth=0.75)
plt.plot(x0[0],y[0], 'go', label='start')
plt.plot(x0[-1],y[-1], 'rx', label='end')
plt.legend()
plt.savefig(fname[:-4]+'_phase.png')
"""