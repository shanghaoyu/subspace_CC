import numpy as np
import re
from validation import io_1
from validation import NM_emulator
import sympy
import pandas as pd
from sklearn.preprocessing import minmax_scale

################################
## main
################################
density_min   = 0.12
density_max   = 0.20
density_gap   = 0.02
density_count = 5

# read 800k pickle file
#df_800k =  pd.read_pickle("/home/slime/subspace_CC/test/code/wave5_NM_fewbody/wave5_NM_fewbody_NI_IM1_cM3.0_pars.pickle")
#print(df_800k)
#print(df_800k.shape)
#print(df_800k.columns)

df_12k =  pd.read_pickle("/home/slime/subspace_CC/test/code/wave5_NM_fewbody/wave5_NM_fewbody_NI_IM1_cM3.0_select_scatt_IM1=5.5_IM2=3.0_pars.pickle")
#print(df_12k)
#print(df_12k.shape)
#print(df_12k.columns)
#print(df_12k.index.array)
index_12k = df_12k.index.array
#print(df_800k.iloc[index, :])

#df_logL_800k = pd.read_pickle("./wave5_NM_fewbody/wave5_NM_fewbody_NI_E1S0_IM1_4.0_IM2_3.0_A=2-16_logL.pickle")
df_logL_800k = pd.read_pickle("./wave5_NM_fewbody/wave5_NM_fewbody_NI_E1S0_IM1_4.0_IM2_3.0_A=2-16_logL_student-t_nocorr.pickle")
#print(df_logL_800k)
#print(df_logL_800k.shape)
#print(df_logL_800k.columns)
#LEC_batch = io_1.read_LEC_batch("LEC_read5.txt")


def txt_to_pickle():
    file_path = "./NM_ccd_800k_samples.txt"
    raw_data  = np.loadtxt(file_path)

    for loop in np.where((raw_data[:,1]==0.12) | (raw_data[:,1]==0.20) |  (raw_data[:,4] < 0)  | (raw_data[:,5] < 0) ):
        raw_data[loop,2:6] = np.nan

    A = np.where((raw_data[:,1]==0.12) | (raw_data[:,1]==0.20) |  (raw_data[:,4] < 0)  | (raw_data[:,5] < 0) )

    B = np.where((raw_data[:,2]==0)|(raw_data[:,3]==0)   |(raw_data[:,4]==0) |(raw_data[:,5]==0))
    #print("sample_killed="+str(sample_killed))
    #print('L<0'+str(np.where(raw_data[:,4] < 0)))
    #print('K<0'+str(np.where(raw_data[:,5] < 0)))
    colums_name =['saturation_density','saturation_energy','symmetry_energy','L','K',\
                  'pnm_0.12', 'pnm_0.14', 'pnm_0.16','pnm_0.18','pnm_0.20',\
                  'snm_0.12', 'snm_0.14', 'snm_0.16','snm_0.18','snm_0.20',\
                  'pnm_hf_0.12', 'pnm_hf_0.14', 'pnm_hf_0.16','pnm_hf_0.18','pnm_hf_0.20',\
                  'snm_hf_0.12', 'snm_hf_0.14', 'snm_hf_0.16','snm_hf_0.18','snm_hf_0.20']

    df = pd.DataFrame(raw_data[:,1::],columns=colums_name)
    df.to_pickle('NM_ccd_800k_samples.pickle')

txt_to_pickle()
df_800k_NM     = pd.read_pickle("NM_ccd_800k_samples.pickle")
df_800k_NM_new = df_800k_NM.dropna(axis=0,how='any') 

df_12k_NM  = df_800k_NM.iloc[index_12k,:]
df_8k_NM   = df_12k_NM.dropna(axis=0,how='any') 
index_8k   = df_8k_NM.index.array
df_logL_8k = df_logL_800k.iloc[index_8k,:]

#select_NI = df_logL_8k['select_scatt_E1S0']
#print(select_NI)
#df_logL_5k = df_logL_8k[select_NI] 
df_logL_5k     = df_logL_8k[df_logL_8k["select_scatt_E1S0"]==True] 
index_5k       = df_logL_5k.index.array
print(index_5k)
df_5k_NM_A2_4  = df_800k_NM.iloc[index_5k,:]
df_5k_NM_A2_4.insert(df_5k_NM_A2_4.shape[1],'weights_option','A2-4')
df_5k_NM_A16   = df_5k_NM_A2_4.copy()
df_5k_NM_A16['weights_option'] = 'A2-4, A16'

#df_5k_NM       = pd.concat([df_5k_NM_A2_4,df_5k_NM_A16],axis=0)
df_5k_NM       = pd.concat([df_5k_NM_A16,df_5k_NM_A2_4],axis=0)

print(df_logL_5k)
print(df_logL_5k.shape)
#print (df_8k_NM)
#print (df_8k_NM.shape)
#print (df_8k_NM.columns)

print (df_5k_NM_A2_4)
print (df_5k_NM_A16)


print (df_5k_NM)
print (df_5k_NM.shape)
# option 1
weights_0  = np.ones(len(df_800k_NM_new))
# option 2
weights_1  = np.exp(df_logL_5k['A2-4'].values)
#weights_1  = minmax_scale(weights_1)
weights_1  = weights_1 /  sum(weights_1)
#weights_1  = np.zeros(len(weights_1))
weights_2  = np.exp(df_logL_5k['A16'].values)
#weights_2  = minmax_scale(weights_2)
weights_2  = weights_2 /  sum(weights_2)
weights_3  = np.exp( df_logL_5k['A2-4'].values + df_logL_5k['A16'].values)
#weights_3  = minmax_scale(weights_3)
weights_3  = weights_3 /  sum(weights_3)

weights    = np.hstack((weights_3 , weights_1))
#print (df_logL_8k)

print(weights_1)
print(weights_2)
print(weights)
print(len(weights))


#observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']
#df_nm = pd.DataFrame(raw_data,columns=observables)
#df_nm.index = int(samples_num_cut)
#print(df_nm)

#sample_count             = 840889
#samples_num              = np.zeros(sample_count)
#saturation_density_batch = np.zeros(sample_count)
#saturation_energy_batch  = np.zeros(sample_count) 
#symmetry_energy_batch    = np.zeros(sample_count)
#L_batch                  = np.zeros(sample_count)
#K_batch                  = np.zeros(sample_count)

#file_path = "NM_ccd_800k_samples.txt"
#with open(file_path,"r") as f_1:
#    count = len(open(file_path,"rU").readlines())
#    data  = f_1.readlines()
#    wtf = re.match('#','abc',flags=0)
#    for loop1 in range(0,count-1):
#        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
#        samples_num[loop1]              = float(temp_1[0])
#        saturation_density_batch[loop1] = float(temp_1[1])
#        saturation_energy_batch[loop1]  = float(temp_1[2]) 
#        symmetry_energy_batch[loop1]    = float(temp_1[3])
#        L_batch[loop1]                  = float(temp_1[4])
#        K_batch[loop1]                  = float(temp_1[5])

#samples_num_cut              = samples_num             [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#saturation_density_batch_cut = saturation_density_batch[np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#saturation_energy_batch_cut  = saturation_energy_batch [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#symmetry_energy_batch_cut    = symmetry_energy_batch   [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#L_batch_cut                  = L_batch                 [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#K_batch_cut                  = K_batch                 [np.where((saturation_density_batch!=0.12) & (saturation_density_batch!=0.20) &(L_batch>0)&(L_batch<90)&(K_batch<800))]
#
#
#raw_data = np.vstack((saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,L_batch_cut,K_batch_cut))
##samples_num_cut              = samples_num_cut             [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
##saturation_density_batch_cut = saturation_density_batch_cut[np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
##saturation_energy_batch_cut  = saturation_energy_batch_cut [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
##symmetry_energy_batch_cut    = symmetry_energy_batch_cut   [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
##L_batch_cut                  = L_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
##K_batch_cut                  = K_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#raw_data = raw_data.T
#print("sample_num"+str(np.int(samples_num_cut)))                  
#print("samples_num_left:"+str(len(samples_num_cut))) 
#print("raw_data:"+str(raw_data.shape)) 

#####################################
## save sample results in pickle file
#####################################
#observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']
#df_nm = pd.DataFrame(raw_data,columns=observables)
#df_nm.to_pickle('NM_ccdt_sampling_from_one_of_34.pickle')
#pd.options.display.float_format = '{:.2f}'.format
#print(df_nm)



#######################
# draw corner plot
#######################

#io_1.plot_corner_plot_2(df_5k_NM.loc[:,['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K','weights_option']],weights)
io_1.plot_corner_plot_1(df_800k_NM_new.loc[:,['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']],weights_0)

#io_1.plot_3(saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,K_batch_cut,L_batch_cut)


#rskin_batch = io_1.read_rskin_data("./pb208_rskin.txt",validation_count)
#
#io_1.plot_4(rskin_batch,symmetry_energy_batch,L_batch,K_batch)
#print(rskin_batch)


