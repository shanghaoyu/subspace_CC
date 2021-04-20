import numpy as np
import re
from validation import io_1
from validation import NM_emulator
import sympy
import pandas as pd

################################
## main
################################
density_min   = 0.12
density_max   = 0.20
density_gap   = 0.02
density_count = 5


#LEC_batch = io_1.read_LEC_batch("LEC_read5.txt")

sample_count             = 840889
samples_num              = np.zeros(sample_count)
saturation_density_batch = np.zeros(sample_count)
saturation_energy_batch  = np.zeros(sample_count) 
symmetry_energy_batch    = np.zeros(sample_count)
L_batch                  = np.zeros(sample_count)
K_batch                  = np.zeros(sample_count)

file_path = "NM_ccd_800k_samples.txt"
with open(file_path,"r") as f_1:
    count = len(open(file_path,"rU").readlines())
    data  = f_1.readlines()
    wtf = re.match('#','abc',flags=0)
    for loop1 in range(0,count-1):
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
        samples_num[loop1]              = float(temp_1[0])
        saturation_density_batch[loop1] = float(temp_1[1])
        saturation_energy_batch[loop1]  = float(temp_1[2]) 
        symmetry_energy_batch[loop1]    = float(temp_1[3])
        L_batch[loop1]                  = float(temp_1[4])
        K_batch[loop1]                  = float(temp_1[5])


samples_num_cut              = samples_num             [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
saturation_density_batch_cut = saturation_density_batch[np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
saturation_energy_batch_cut  = saturation_energy_batch [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
symmetry_energy_batch_cut    = symmetry_energy_batch   [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
L_batch_cut                  = L_batch                 [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
K_batch_cut                  = K_batch                 [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]


raw_data = np.vstack((saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,L_batch_cut,K_batch_cut))
#samples_num_cut              = samples_num_cut             [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#saturation_density_batch_cut = saturation_density_batch_cut[np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#saturation_energy_batch_cut  = saturation_energy_batch_cut [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#symmetry_energy_batch_cut    = symmetry_energy_batch_cut   [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#L_batch_cut                  = L_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#K_batch_cut                  = K_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
raw_data = raw_data.T
                  
print("samples_num_cut:"+str(len(samples_num_cut))) 
print("raw_data:"+str(raw_data.shape)) 

#####################################
## save sample results in pickle file
#####################################
observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']
df_nm = pd.DataFrame(raw_data,columns=observables)
df_nm.to_pickle('NM_ccdt_sampling_from_one_of_34.pickle')
pd.options.display.float_format = '{:.2f}'.format
print(df_nm)
#######################
# draw corner plot
#######################
io_1.plot_corner_plot(df_nm.loc[:,['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']])
#io_1.plot_corner_plot(df_nm.loc[:,['saturation_density', 'saturation_energy','symmetry_energy','rho','set_num']])

#io_1.plot_3(saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,K_batch_cut,L_batch_cut)


#rskin_batch = io_1.read_rskin_data("./pb208_rskin.txt",validation_count)
#
#io_1.plot_4(rskin_batch,symmetry_energy_batch,L_batch,K_batch)
#print(rskin_batch)


