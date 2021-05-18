import pandas as pd
import numpy as np


def txt_to_pickle():
    file_path = "./NM_ccd_800k_samples.txt"
    raw_data  = np.loadtxt(file_path)
    print(raw_data)
    print(raw_data.shape)
    
    
    for loop in np.where((raw_data[:,1]==0.12) | (raw_data[:,1]==0.20) |  (raw_data[:,4] < 0)  | (raw_data[:,5] < 0) ):
        raw_data[loop,2:6] = np.nan
    
    A = np.where((raw_data[:,1]==0.12) | (raw_data[:,1]==0.20) |  (raw_data[:,4] < 0)  | (raw_data[:,5] < 0) )
    print(A)
    print(len(A[0]))
    
    B = np.where((raw_data[:,2]==0)|(raw_data[:,3]==0)   |(raw_data[:,4]==0) |(raw_data[:,5]==0))
    print("b"+str(B))
    #print("sample_killed="+str(sample_killed))
    print('L<0'+str(np.where(raw_data[:,4] < 0)))
    print('K<0'+str(np.where(raw_data[:,5] < 0)))
        
    
    
    
    colums_name =['sample_number','saturation_density','saturation_energy','symmetry_energy','L','K',\
                  'pnm_0.12', 'pnm_0.14', 'pnm_0.16','pnm_0.18','pnm_0.20',\
                  'snm_0.12', 'snm_0.14', 'snm_0.16','snm_0.18','snm_0.20',\
                  'pnm_hf_0.12', 'pnm_hf_0.14', 'pnm_hf_0.16','pnm_hf_0.18','pnm_hf_0.20',\
                  'snm_hf_0.12', 'snm_hf_0.14', 'snm_hf_0.16','snm_hf_0.18','snm_hf_0.20!']
    
    
    df = pd.DataFrame(raw_data,columns=colums_name)
    df.to_pickle('NM_ccd_800k_samples.pickle')
    
    
    print (df)
    print (df.shape)
    print (df.columns)

def pickle_to_txt(in_file_path,out_file_path):
    df_samples =  pd.read_pickle(in_file_path)

    print (df_samples)
    print (df_samples.shape)
    print (df_samples.columns )
    df_new = df_samples.values
    
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=12)
    np.savetxt(out_file_path,df_new,fmt='%.12f')

pickle_to_txt("wave5_NM_fewbody_NI_IM1_cM3.0_select_scatt_IM1=5.5_IM2=3.0_pars.pickle","LEC_read6.txt")


#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=12)
#np.savetxt(file_path,df_samples,fmt='%.12f')

#with open(file_path,'w') as f_1:
#    for loop1 in range(len(df_samples)):
#        f_1.write( " %.12f \n" % (df_samples[loop1][0]))



#df_cv = df_cv.values
#print(str(len(df_cv)))
#file_path = "./LEC_read3.txt"
#np.set_printoptions(suppress=True)
#np.set_printoptions(precision=12)
#np.savetxt(file_path,df_cv,fmt='%.12f')
#


