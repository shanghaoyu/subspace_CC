import os
import numpy as np
import math
import re
import scipy.linalg as spla
from scipy import interpolate
from scipy import linalg
from ..io import inoutput

def input_file(file_path,matrix):
    with open(file_path, 'r') as f_1:
        data = f_1.readlines()
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[0])
        subspace_dimension = int(temp_1[0])
        for loop1 in range (0, subspace_dimension):
            temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[1+loop1])
            matrix[loop1,:] = temp_2[:]


######################################################
######################################################
### generate subspace vector (subspace_cal)
######################################################
######################################################
def subspace_cal(vec_input,in_dir,out_dir):
    inoutput.read_LEC_1("ccm_in_DNNLO450")
    particle_num = 14
    density      = 0.16
    nmax         = 2
    cal_type     = "subsapce_cal"
    inoutput.generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax,cal_type)
    os.system('./'+nucl_matt_exe+' '+in_dir+' > '+out_dir)


######################################################
######################################################
### call solve_general_EV 
######################################################
######################################################
def call_solve_general_EV(vec_input,in_dir,out_dir):
    neutron_num  = 14  #test
    particle_num = 28
    density      = 0.16
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2 #test

    inoutput.generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax,"solve_general_EV")
    os.system('./'+nucl_matt_exe+' '+in_dir+' > '+out_dir)



######################################################
######################################################
### print H matrix for individual LEC
######################################################
######################################################
def print_LEC_matrix(out_dir,subspace_dimension,matrix):
    with open(out_dir,'w')  as f_1:
        f_1.write(matrix)
       # for loop1 in range (subspace_dimension):
       #     f_1.write(matrix[loop1,:]+'\n')





######################################################
######################################################
### generate emulator_matrix
######################################################
######################################################
def generate_emulator_matrix(subspace_dimension):
    C_matrix = np.zeros((subspace_dimension,subspace_dimension))
    N_matrix = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((subspace_dimension,subspace_dimension))
    K_matrix = np.zeros((subspace_dimension,subspace_dimension))
    LEC_all_matrix = np.zeros((LEC_number,subspace_dimension,subspace_dimension))

    LEC     = np.zeros(LEC_number)
    call_solve_general_EV(LEC,"ccm_in_test","a.out")
    N_matrix = np.loadtxt("N_matrix.txt")
    H_matrix = np.loadtxt("H_matrix.txt")
    K_matrix = np.loadtxt("K_matrix.txt")
    out_dir = "./emulator/N_matrix.txt"
    np.savetxt(out_dir,N_matrix)
 
    C_matrix = H_matrix + K_matrix
    out_dir = "./emulator/C_matrix.txt"
    np.savetxt(out_dir,C_matrix)

    for loop1 in range(LEC_number):
        LEC = np.zeros(LEC_number)
        LEC[loop1] = 1 
        call_solve_general_EV(LEC,"ccm_in_test","a.out")
        H_matrix = np.loadtxt("H_matrix.txt")
        K_matrix = np.loadtxt("K_matrix.txt")
        LEC_all_matrix[loop1,:,:] = H_matrix + K_matrix - C_matrix
        out_dir = "./emulator/LEC_"+str(loop1+1)+"_matrix"
        np.savetxt(out_dir,LEC_all_matrix[loop1,:,:])


def test_1():
    print("test_1")
    inoutput.test_2()    
######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 64
LEC_number = 17
nucl_matt_exe = './prog_ccm.exe'
#
#generate_emulator_matrix(subspace_dimension)


