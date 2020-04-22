import numpy as np
import matplotlib
#matplotlib.use('PS')
import matplotlib.pyplot as plt

import re
from math import log 
from math import e
from scipy import interpolate


def input_file_2(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    width = np.size(raw_data,1)
    print ("width= "+str(width))
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                for loop3 in range(width):
                    raw_data[loop2][loop3] = float(temp_1[loop3])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 

def input_raw_data_count(file_path):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                loop2 = loop2 + 1
            loop1 = loop1 + 1
    return loop2



file_path = "/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_dens_5points/density_extrapolation.txt"
data_num = input_raw_data_count(file_path)
raw_data_1 = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data_1)

x_list_1 = raw_data_1[:,0]  # dens
y_list_1 = raw_data_1[:,1]  # ccd
y_list_2 = raw_data_1[:,2]  # extrapolation

print(x_list_1)
print(y_list_1)


file_path = "/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_dens_5points/subspace_sample_5point.txt"
data_num = input_raw_data_count(file_path)
raw_data_2 = np.zeros((data_num,2),dtype = np.float)
input_file_2(file_path,raw_data_2)

x_list_3 = raw_data_2[:,0] # subspace data
y_list_3 = raw_data_2[:,1] #

print(x_list_3)
print(y_list_3)

interpol_count = 100
#X = []
#Y = []

dens = x_list_1
spl_pnm = interpolate.UnivariateSpline(dens,y_list_2,k = 5)
spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)

interp_pnm = spl_pnm(spldens)

x_list_2_new = spldens
y_list_2_new = interp_pnm 

#print(x_list_2_new)
#print(y_list_2_new)

# start plotting

fig1 = plt.figure('fig1')
l1 = plt.scatter(x_list_1,y_list_1,color = 'r', marker = 'd',zorder=1,label="CCD calculation")
l2 = plt.plot(x_list_2_new,y_list_2_new,color = 'k',linestyle=':',linewidth=1.5,alpha=0.9,  label="SP-CC(5)",zorder=1)
l3 = plt.plot(x_list_3,y_list_3,color = 'k', marker = 'o',markersize = 10,markerfacecolor='none',linestyle='',zorder=3, label="subspace samples")

#plt.yticks(np.arange(8,24,2),fontsize = 13) 
#plt.xticks(np.arange(0.12,0.205,0.01),fontsize = 13) 
plt.legend(loc='lower right',fontsize = 13)

plt.ylabel('$E/A$ [MeV]',fontsize=18)
plt.xlabel(r"$\rho$ [fm$^{-3}$]",fontsize=18)

plot_path = 'density_extrapolation.pdf'
plt.savefig(plot_path,bbox_inches='tight')

#plt.show()



