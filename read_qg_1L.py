import numpy as np
import matplotlib.pyplot as plt
import os

## Use this file after running qg_1L_io.py

# Read from files
## Note: data contains: Ly,Lj,Hm,hy,f0,bet,g0,F0,dkk,Ny,hy,nEV in that order 
data = np.fromfile('storage/InputData')
grow = np.fromfile('storage/grow')
freq = np.fromfile('storage/freq')
mode = np.fromfile('storage/mode',dtype=np.complex128)
kk = np.fromfile('storage/kk')
evalsArr = np.fromfile('storage/evals')
Lj = data[1]
Ny = data[9]
nEV = data[11]
nk = len(kk)

grow = grow.reshape([nEV,nk])
freq = freq.reshape([nEV,nk])
mode = mode.reshape([Ny+1,nEV,nk])

for i in range(nk):
    for j in range(int(evalsArr[i])):
        plt.plot(kk*Lj, grow[j]*3600*24, 'o')

plt.ylabel('1/day')
plt.xlabel('k')
plt.title('Growth Rate: 1-Layer QG')
plt.savefig('Grow1L_QG.eps', format='eps', dpi=1000)
plt.show()

