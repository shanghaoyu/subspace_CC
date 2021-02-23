import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
from scipy import interpolate
from scipy import optimize
import seaborn as sns
import pandas as pd
from scipy.special import loggamma,gamma
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,  WhiteKernel
from numpy.linalg import solve, cholesky

H1 = np.random.normal(-5,5,[50,50])
HL = np.tril(H1)
B1 = np.random.normal(-5,5,[15,15])
B2 = np.random.normal(-5,5,[20,20])
B3 = np.random.normal(-5,5,[15,15])
Ball = spla.block_diag(B1,B2,B3)

#B1 = np.random.normal(-5,5,[20,10])
#B2 = np.random.normal(-5,5,[30,10])
#B3 = np.random.normal(-5,5,[20,20])
#Ball = spla.block_diag(B1,B2,B3)


H  = Ball 


fig1 = plt.figure('fig1')
plt.figure(figsize=(6,10))
plt.subplots_adjust(wspace =0.3, hspace =0.4)

fontsize_x_label = 10
#plt.figure(figsize=(5,5))
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
#plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

ax1 = plt.subplot(131)
plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
ax1.matshow(H1)

matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax2 = plt.subplot(132)
plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['top'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.spines['right'].set_linewidth(2)

ax2.matshow(Ball)
#plt.colorbar(extend='both',fraction=0.041, pad=0.04 )
#plt.vlines(len(K_all)/2-0.5,0,len(K_all)-1,lw=1)
#plt.hlines(len(K_all)/2-0.5,0,len(K_all)-1,lw=1)

#plt.text(50, -2, 'E/N')
#plt.text(170, -2, 'E/A')
#plt.text(-20, 50, 'E/N')
#plt.text(-20, 170, 'E/A')

plt.xticks([])
plt.yticks([])

plot_path = 'random_matrix.pdf'
plt.savefig(plot_path)
plt.close('all')
