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
from validation import io_1
from validation import NM_emulator

my_LEC_label = ['cE','cD','c1','c2','c3','c4','Ct1S0pp','Ct1S0np','Ct1S0nn','Ct3S1','C1S0','C3P0','C1P1','C3P1','C3S1','CE1','C3P2']
LEC_batch    = io_1.read_LEC_batch("LEC_read1.txt")
LEC_batch    = LEC_batch[0:64]
LEC_batch    = np.array(LEC_batch)
print(len(LEC_batch))

print(my_LEC_label)

sns.set_style("white")
fig,axes = plt.subplots(4,5,figsize=(10,12),sharey='all')
plt.subplots_adjust(wspace =0.1, hspace =0.5)
#matplotlib.rcParams['xtick.direction'] = 'in'
#matplotlib.rcParams['ytick.direction'] = 'in'
fig_count = 0
for ket in range(4):
    for bar in range(5):
        ax = axes[ket][bar]
        if fig_count < 17:
            x_list = LEC_batch[:,fig_count]
            sns.distplot(x_list,kde=False,rug=True,rug_kws={"height": -.09,"clip_on":False},bins=10,ax=ax)
            ax.set_ylim(0,19)
            x_min = min(x_list)
            x_max = max(x_list)
            x_mid = (x_max-x_min)/2
            x_shift = x_mid *0.4
            x_min_shift = round((x_min + x_shift),2)
            x_max_shift = round((x_max - x_shift),2)
            round_ = round(math.log(x_mid * 2))
            #ax.set_xlim(x_min,x_max)
            ax.set_xticks([x_min_shift,x_max_shift])
            if round_ >=0:
                ax.set_xticklabels([round(x_min,2),round(x_max,2)])
            elif round_ <0:
                ax.set_xticklabels([round(x_min,3),round(x_max,3)])
            ax.set_xlabel("%s" %(my_LEC_label[fig_count]),fontsize = 20)
            ax.tick_params(labelsize=14)
            if bar == 0:
                ax.set_ylabel("count")
            fig_count = fig_count + 1
        else:
            fig.delaxes(axes[ket][bar])


plot_path = 'hist_LECs.pdf'
#plot_path = 'emulator_vs_ccd_without_small_batch_voting_new.pdf'
plt.savefig(plot_path,bbox_inches='tight')
