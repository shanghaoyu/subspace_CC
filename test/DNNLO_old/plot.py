import numpy as np 
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
from scipy import interpolate

def input_file_1(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data = f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])  #density
                raw_data[loop2][1] = float(temp_1[1])  #ccd
                raw_data[loop2][2] = float(temp_1[2])  #ccd(t)
                raw_data[loop2][3] = (3 * math.pi**2 * raw_data[loop2][0])**(1./3)#pf
                loop2 = loop2 + 1
            loop1 = loop1 + 1

def input_file_count(file_path):
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
       # print ('data_num='+str(loop2))
        return loop2
cutoff = 450

file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNNLO%d_pnm/EOS_different_density.txt"  %(cutoff)
data_num  = input_file_count(file_path)
DNNLO450_data = np.zeros((data_num,5),dtype = np.float)
input_file_1(file_path,DNNLO450_data)

file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNLO%d_pnm/EOS_different_density.txt"  %(cutoff)
data_num  = input_file_count(file_path)
DNLO450_data = np.zeros((data_num,5),dtype = np.float)
input_file_1(file_path,DNLO450_data)

file_path = "/home/slime/subspace_CC/test/DNNLO_old/DLO%d_pnm/EOS_different_density.txt"  %(cutoff) 
data_num  = input_file_count(file_path)
DLO450_data = np.zeros((data_num,5),dtype = np.float)
input_file_1(file_path,DLO450_data)

lambda_b = 500
hbarc = 197.326960277
Q  =  DNNLO450_data[:,3] * hbarc / lambda_b
print(Q)
print(DLO450_data[:,3])
 
a0 = np.ones(len(DNNLO450_data)) 
a1 = np.zeros(len(DNNLO450_data)) 
#a2 = (DNLO450_data[:,2] / DLO450_data[:,2] - a0) / (Q**2)
#a3 = (DNNLO450_data[:,2] / DLO450_data[:,2] - a0 - a2 * Q**2) / (Q**3)
a2 = (DNLO450_data[:,2] / DLO450_data[:,2] - a0) / (Q**2)
a3 = (DNNLO450_data[:,2] / DLO450_data[:,2] - a0 - a2 * Q**2) / (Q**3)


#print(Q)
print(a0)
print(a2)
print(a3)
### D...[:,3] = pf
X0 = DLO450_data[:,2]
#print(X0)
for loop1 in range(len(DNNLO450_data)):
    DLO450_data[loop1,4]   = X0[loop1] * Q[loop1]**2 * max(abs(a0[loop1]),abs(a1[loop1]))
    #print(max(abs(a0[loop1]),abs(a1[loop1]),abs(a2[loop1])))
    DNLO450_data[loop1,4]  = X0[loop1] * Q[loop1]**3 * max(abs(a0[loop1]),abs(a1[loop1]),abs(a2[loop1]))
    DNNLO450_data[loop1,4] = X0[loop1] * Q[loop1]**4 * max(abs(a0[loop1]),abs(a1[loop1]),abs(a2[loop1]),abs(a3[loop1])) 

print(DLO450_data[:,4])
print(DNLO450_data[:,4])
print(DNNLO450_data[:,4])

rn = 1
#data interpolation
x        = DNNLO450_data[:,0]
x_new    = np.linspace(x.min(),x.max(),1000)

f_smooth = interpolate.splrep(DNNLO450_data[:,0],DNNLO450_data[:,2])
y1       = interpolate.splev(x_new,f_smooth)

f_smooth = interpolate.splrep(DNNLO450_data[:,0],DNNLO450_data[:,4])
y1_error = rn * interpolate.splev(x_new,f_smooth)


f_smooth = interpolate.splrep(DNLO450_data[:,0],DNLO450_data[:,2])
y2       = interpolate.splev(x_new,f_smooth)
f_smooth = interpolate.splrep(DNLO450_data[:,0],DNLO450_data[:,4])
y2_error = rn * interpolate.splev(x_new,f_smooth)


f_smooth = interpolate.splrep(DLO450_data[:,0],DLO450_data[:,2])
y3       = interpolate.splev(x_new,f_smooth)
f_smooth = interpolate.splrep(DLO450_data[:,0],DLO450_data[:,4])
y3_error = rn * interpolate.splev(x_new,f_smooth)


fig1  = plt.figure('fig1',figsize= (10,7))
plt.plot(x_new,y1,color = 'b',linestyle = '-.',linewidth=2, alpha=0.9, label=r'$\Delta$NNLO(500)',zorder=4)
plt.fill_between(x_new, y1 + y1_error, y1 - y1_error,color='b',alpha=0.3,zorder=1)

plt.plot(x_new,y2,color = 'g',linestyle = '--',linewidth=2, alpha=0.9, label=r'$\Delta$NLO(500)',zorder=5)
plt.fill_between(x_new, y2 + y2_error, y2 - y2_error,color='g',alpha=0.3,zorder=2)

plt.plot(x_new,y3,color = 'r',linestyle = ':',linewidth=2, alpha=0.9, label=r'$\Delta$LO(500)',zorder=6)
plt.fill_between(x_new, y3 + y3_error, y3 - y3_error,color='r',alpha=0.3,zorder=3)

plt.yticks(np.arange(0,60,10),fontsize = 13)
plt.xticks(np.arange(0.06,0.41,0.1),fontsize = 13)
plt.xlim((0.06,0.40))
plt.ylim((0,60))
#plt.yticks(np.arange(0,60,5),fontsize = 13)
#plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
#plt.xlim((0.05,0.20))
#plt.ylim((0,30))



plt.legend(loc='upper left',fontsize = 13)
plt.ylabel('$E/A$ [MeV]',fontsize=18)
plt.xlabel(r"$\rho$ [fm$^{-3}$]",fontsize=18)

plot_path = 'EOS_Delta450.pdf'
plt.savefig(plot_path, bbox_inches = 'tight')








