import os
import numpy as np
from my_package.ccd_emulator import eigenvector_diagonal
from my_package.ccd_emulator import validation
from my_package.io import inoutput

######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 64
LEC_num = 17
LEC_range = 0.2 
LEC = np.ones(LEC_num)
nucl_matt_exe = './prog_ccm.exe'
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/DNNLOgo450_20percent_64points_/'
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/'
database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/pnm_66_0.16_DNNLOgo_20percent_64points/'

# start validation 


for loop1 in range(1):
    file_path = "ccm_in_DNNLO450"
    LEC = inoutput.read_LEC_1(LEC_num,file_path)
    #LEC_random = generate_random_LEC(LEC, LEC_range)
    print ("LEC="+str(LEC))
    subtract = [14,20,23,38,41,50,53,56,57,59,62,63]
#    subtract = [0:32]
    #subtract = list(range(32,64))
    print('type'+str(type(subtract)))


    emulator_cal, ev_all_1, ev_all_2 = validation.emulator(database_dir,LEC,subtract)
    file_path = "validation_different_subspace.txt"
    with open(file_path,'a') as f_1:
        f_1.write('emulator = %.12f \n' % (emulator_cal))
    file_path = "validation_detail_test_different_subspace.txt"
    with open(file_path,'a') as f_2:
        f_2.write('emulator = %.12f   all =' % (emulator_cal))
        f_2.write(str(ev_all_1))
        f_2.write('\n')
        f_2.write(str(ev_all_2))
        f_2.write('\n')

