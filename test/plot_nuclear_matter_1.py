import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 

import re

from scipy import interpolate
from math import log 
from math import e

def input_file_2(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])
                raw_data[loop2][1] = float(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 
       # print (loop2)

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

file_path  = "christian_interaction.txt"
data_num   = input_file_count(file_path)
N_132_data = np.zeros((data_num,3),dtype = np.float)
N_28_data  = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,N_132_data)

#file_path  = "N_28_NNLO450.txt"
#input_file_2(file_path,N_28_data)
interpol_count = 1000
print(N_132_data)

X  = []
Y1 = []
Y2 = []
for i in range(0,N_132_data.shape[0],5):
    dens = N_132_data[i:i+5,0]
    temp_snm = N_132_data[i:i+5,1]
    temp_pnm = N_132_data[i:i+5,2]
    spl_ccdt_snm = interpolate.UnivariateSpline(dens,temp_snm,k=3)
    spl_ccdt_pnm = interpolate.UnivariateSpline(dens,temp_pnm,k=3)
    spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)
    interp_snm = spl_ccdt_snm(spldens)
    interp_pnm = spl_ccdt_pnm(spldens)
    for j in range(0,spldens.size):
            X.append(spldens[j])
            Y1.append(interp_snm[j])
            Y2.append(interp_pnm[j])
npX  = np.array(X)
npY1 = np.array(Y1)
npY2 = np.array(Y2)


N_132_interpolation = np.append(np.transpose([npX]),np.transpose([npY1]),1)
N_132_interpolation = np.append(N_132_interpolation,np.transpose([npY2]),1)
#print ("npY1"+str(npY1))
print ("data_interpolation="+str(N_132_interpolation))
#data_interpolation_backup = data_interpolation.copy()


#X  = []
#Y1 = []
#Y2 = []
#for i in range(0,N_28_data.shape[0],5):
#    dens = N_28_data[i:i+5,0]
#    temp_snm = N_28_data[i:i+5,1]
#    temp_pnm = N_28_data[i:i+5,2]
#    spl_ccdt_snm = interpolate.UnivariateSpline(dens,temp_snm,k=4)
#    spl_ccdt_pnm = interpolate.UnivariateSpline(dens,temp_pnm,k=4)
#    spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)
#    interp_snm = spl_ccdt_snm(spldens)
#    interp_pnm = spl_ccdt_pnm(spldens)
#    for j in range(0,spldens.size):
#            X.append(spldens[j])
#            Y1.append(interp_snm[j])
#            Y2.append(interp_pnm[j])
#npX  = np.array(X)
#npY1 = np.array(Y1)
#npY2 = np.array(Y2)
#
#
#N_28_interpolation = np.append(np.transpose([npX]),np.transpose([npY1]),1)
#N_28_interpolation = np.append(N_28_interpolation,np.transpose([npY2]),1)


#data analysis
#N_28_saturation_snm = np.min(N_28_interpolation[:,1])
#temp1 = N_28_interpolation[np.where(N_28_interpolation[:,1]==N_28_saturation_snm),0]
#N_28_saturation_dens = temp1[0]
#temp2 = N_28_interpolation[np.where(N_28_interpolation[:,1]==N_28_saturation_snm),2]
#N_28_saturation_pnm = temp2[0]
#saturation_energy_28 = N_28_saturation_pnm - N_28_saturation_snm

#df_28 = np.diff(N_28_interpolation[:,1])/np.diff(N_28_interpolation[:,0])
#ddf_28 = np.diff(df_28) /np.diff(N_28_interpolation[1:len(N_28_interpolation),0])
#temp3 = ddf_28[np.where(N_28_interpolation[:,1]==N_28_saturation_snm)]
#ddf_saturation_dens_28 = temp3[0]
#print ('ddf_saturation_dens_28=',ddf_saturation_dens_28)
#print ('saturation_dens_28=',N_28_saturation_dens)
#K0 = 9* pow(N_28_saturation_dens,2)*ddf_saturation_dens_28
#print ('K0_28=',K0)

N_132_saturation_snm = np.min(N_132_interpolation[:,1])
temp1 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),0]
N_132_saturation_dens = temp1[0]
temp2 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),2]
N_132_saturation_pnm = temp2[0]
print ('snm='+str(N_132_saturation_snm))
print ('dens='+str(N_132_saturation_dens))
print ('pnm='+str(N_132_saturation_pnm))
saturation_energy_132 = N_132_saturation_pnm- N_132_saturation_snm
print ('saturation_energy='+str(saturation_energy_132))


df_132 = np.diff(N_132_interpolation[:,1])/np.diff(N_132_interpolation[:,0])
ddf_132 = np.diff(df_132) /np.diff(N_132_interpolation[1:len(N_132_interpolation),0])
temp3 = ddf_132[np.where(N_132_interpolation[:,1]==N_132_saturation_snm)]
ddf_saturation_dens_132 = temp3[0]

K0 = 9* pow(N_132_saturation_dens,2)*ddf_saturation_dens_132
print ('K0_132=',K0)

S = N_132_interpolation[:,2] - N_132_interpolation[:,1]
u = N_132_interpolation[:,0] / N_132_saturation_dens
ds = np.diff(S) 
du = np.diff(u)
#u_0 is the position of saturation point
u_0 = np.where(u[:]== 1)
print('test='+str(u[688]))

print('u_0'+str(u_0))
L = 3 * u[u_0]*ds[u_0]/du[u_0]

print('L='+str(L)) 

#print('x='+str(len(N_132_interpolation[:,1])))
#print('S='+str(S)) 
#print('u='+str(u)) 





######################################################
######################################################
#####      write raw data file 
######################################################
######################################################
def cD_cE_area():
    particle_num = 28
    neutron_num = 14
    cE_min = -1
    cE_max = 1
    cE_gap = 0.25
    cE_count = int( (cE_max - cE_min) / cE_gap + 1 )
    cD_min = -3
    cD_max = 3
    cD_gap = 1
    cD_count = int( (cD_max - cD_min) / cD_gap + 1 )
    density_min = 0.12
    density_max = 0.20
    density_gap = 0.02
    density_count = int( (density_max - density_min) / density_gap +1 )
    data_num = cE_count*cD_count*density_count
    raw_data = np.zeros((data_num,7),dtype = np.float)  # cD,cE,density,snm,pnm
    file_path = './cD%.2f-%.2f_cE%.2f-%.2f.dat' % (cD_min,cD_max,cE_min,cE_max)
    
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])
                raw_data[loop2][1] = float(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
                raw_data[loop2][3] = float(temp_1[3])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 
        #print loop2





    print ('raw_data='+str(raw_data)) 
    interpol_count_ = 1000
    saturation_point = np.zeros((cE_count*cD_count,2))
    kind = "quadratic"
    
    #raw_data_1 = raw_data[np.where(raw_data[:,2]<0.20)]
    raw_data_2 = raw_data[np.where( (raw_data[:,0]==0) & (raw_data[:,1]==0) )]
    
    raw_data_2 = raw_data[np.where( (raw_data[:,0]==0) & (raw_data[:,1]==0) )]
    
    for loop1 in range(cE_count*cD_count):
        x = raw_data[loop1*density_count:loop1*density_count+density_count,2]
        y = raw_data[loop1*density_count:loop1*density_count+density_count,3]

        print ('x='+str(x))
        spldens = np.linspace(density_min,density_max,num=interpol_count_)
        f       = interpolate.interp1d(x,y,kind=kind)
        y_new   = np.zeros(interpol_count_)
        y_new   = f(spldens)
        saturation_point[loop1,1] = np.min(y_new)
        temp = spldens[np.where(y_new == np.min(y_new))]
        saturation_point[loop1,0] = temp[0]
    saturation_point_1=saturation_point[np.where(saturation_point[:,0]<0.2)]
    x_list = saturation_point_1[:,0]
    y_list = saturation_point_1[:,1]
    
    
    saturation_point_2 = np.zeros((int(len(raw_data_2)/density_count),2))
#    print ('raw_data_2='+str(raw_data_2))
    for loop1 in range(int(len(raw_data_2)/density_count)):
        x_2 = raw_data_2[loop1*density_count:loop1*density_count+density_count,2]
        y_2 = raw_data_2[loop1*density_count:loop1*density_count+density_count,3]
        spldens_2 = np.linspace(density_min,density_max,num=interpol_count_)
        f_2       = interpolate.interp1d(x_2,y_2,kind=kind)
        y_new_2   = np.zeros(interpol_count_)
        y_new_2   = f_2(spldens_2)
        saturation_point_2[loop1,1] = np.min(y_new_2)
        temp = spldens_2[np.where(y_new_2 == np.min(y_new_2))]
        saturation_point_2[loop1,0] = temp[0]
    x_list_2 = saturation_point_2[:,0]
    y_list_2 = saturation_point_2[:,1]
#    print ('satruation_point='+str(saturation_point_2))
    return x_list,y_list   
    



#x_list,y_list = cD_cE_area() 

#x_list_1   = N_28_interpolation[:,0]
#y_list_1   = N_28_interpolation[:,2]
x_list_2   = N_132_interpolation[:,0]
y_list_2   = N_132_interpolation[:,2]
#x_list_1_p = N_28_data[:,0]
#y_list_1_p = N_28_data[:,2]
x_list_2_p = N_132_data[:,0]
y_list_2_p = N_132_data[:,2]

#y_list_2   = N_132_data[:,2]
fig1 = plt.figure('fig1',figsize=(5,8))
plt.subplot(211)
#l1 = plt.plot(x_list_1,y_list_1,color = 'b',linestyle='--',label='N=14')
l2 = plt.plot(x_list_2,y_list_2,color = 'r',linestyle='--',label='N=66',zorder=1)
#l11 = plt.scatter(x_list_1_p,y_list_1_p,color = 'k',s = 10, marker = 'x')
l22 = plt.scatter(x_list_2_p,y_list_2_p,color = 'k',s = 10, marker = 's',zorder=2)


plt.title('pnm_E/A=%.2fMeV  snm_E/A=%.2fMeV\nsaturation_dens=%.3ffm$^{-3}$  saturation_energy=%.2fMeV\n K0=%.2f  L=%.2f'% (N_132_saturation_pnm,N_132_saturation_snm,N_132_saturation_dens,saturation_energy_132,K0,L))
plt.legend(loc='lower right')
plt.ylabel('pnm E/A (MeV)',fontsize=14)
plt.xlabel(r"$\rho$ (fm$^{-3}$)",fontsize=14)
#plt.xlim((0.1,0.22))
#plt.ylim((-25,-10))

#x_list_3   = N_28_interpolation[:,0]
#y_list_3   = N_28_interpolation[:,1]
x_list_4   = N_132_interpolation[:,0]
y_list_4   = N_132_interpolation[:,1]
#x_list_3_p = N_28_data[:,0]
#y_list_3_p = N_28_data[:,1]
x_list_4_p = N_132_data[:,0]
y_list_4_p = N_132_data[:,1]

plt.subplot(212)
#l3 = plt.plot(x_list_3,y_list_3,color = 'b',linestyle='--',label='A=28')
l4 = plt.plot(x_list_4,y_list_4,color = 'r',linestyle='--',label='A=132',zorder=1)
#l33 = plt.scatter(x_list_3_p,y_list_3_p,color = 'k',s = 10, marker = 'x')
l44 = plt.scatter(x_list_4_p,y_list_4_p,color = 'k',s = 10, marker = 's',zorder=2)
#l5  = plt.scatter(x_list,y_list,color = 'b', s = 20, marker='.')

plt.legend(loc='lower right')
plt.ylim((-18,-12))
plt.ylabel('snm E/A (MeV)',fontsize=14)
plt.xlabel(r"$\rho$ (fm$^{-3}$)",fontsize=14)


plot_path = 'christian_interaction.pdf'
plt.savefig(plot_path)
plt.show()



