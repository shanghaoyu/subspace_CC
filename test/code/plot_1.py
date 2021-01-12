import numpy as np
import re
from validation import io_1
from validation import NM_emulator


sample_count             = 34
samples_num              = np.zeros(sample_count)
saturation_density_batch = np.zeros(sample_count)
saturation_energy_batch  = np.zeros(sample_count)
symmetry_energy_batch    = np.zeros(sample_count)
L_batch                  = np.zeros(sample_count)
K_batch                  = np.zeros(sample_count)

file_path = "pb208_NM_ccdt.txt"
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

print(saturation_density_batch)

rskin_batch_cc= io_1.read_rskin_data("./pb208_rskin.txt",sample_count,start_line=1,position=3)

rskin_batch_imsrg = io_1.read_rskin_data("./Pb208_DeltaGO394_sampleXXX_e14_E28_hw10_IMSRG.dat",sample_count,start_line =12 ,position=6)

alphaD_batch = io_1.read_rskin_data("./Pb208_DeltaGO394_sampleXXX_e14_E28_hw10_IMSRG.dat",sample_count,start_line =12 ,position=2)
print(alphaD_batch)
print(rskin_batch_cc)
print(rskin_batch_imsrg)
io_1.plot_10(symmetry_energy_batch,alphaD_batch,rskin_batch_cc,rskin_batch_imsrg)
