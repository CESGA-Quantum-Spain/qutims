import numpy as np
import pandas as pd
import time as tm

delay = 10

# source: https://web.cecs.pdx.edu/mcnames/DataSets/index.html
originfile = 'data_santafe_2000.dat'
data = np.loadtxt(originfile)[(20-delay):]

data = np.array(data) - np.mean(data)
data = 0.75*np.array(data)/np.max(np.abs(data)) # normalization to [-1,1] range

data_t = np.array(range(0,len(data)))

data_t = data_t[:-delay]
data_x = data[:-delay]
data_y = data[delay:]
data_f = np.vstack((data_t,data_x,data_y)).T

print(data_f.shape)



filename = "data_santafe10.dat"
np.savetxt(filename, data_f)

log_data = pd.DataFrame({'delay':delay, 'filename':[filename]})

logfile = '.logdatagen_%i.dat' %int(tm.time())
log_data.to_csv(logfile, index=False)
