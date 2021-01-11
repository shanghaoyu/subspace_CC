import os
import numpy as np
import math
import re
import scipy.linalg as spla
from scipy import interpolate
from scipy import linalg


def input_file(file_path,matrix):
    with open(file_path, 'r') as f_1:
        data = f_1.readlines()
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[0])
        subspace_dimension = int(temp_1[0])
        for loop1 in range (0, subspace_dimension):
            temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[1+loop1])
            matrix[loop1,:] = temp_2[:]
#            print(loop1)
#            print(matrix[loop1,:])
#LECs = [200,-91.85]

######################################################
######################################################
### generate infile for solve_general_EV
######################################################
######################################################
def generate_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write('3, 394\n')
        f_1.write('! cE and cD 3nf parameters:'+'\n' )
        f_1.write('%.12f, %.12f\n' % (vec_input[0],vec_input[1]))
        f_1.write('! LEC ci \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f \n' % (vec_input[2],vec_input[3],vec_input[4],vec_input[5]))
        f_1.write('!c1s0 & c3s1 \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f, %.12f, %.12f \n' % (vec_input[6],vec_input[7],vec_input[8],vec_input[9],vec_input[9],vec_input[9]))
        f_1.write('! cnlo(7) \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f, %.12f, %.12f, %.12f \n' % (vec_input[10],vec_input[11],vec_input[12],vec_input[13],vec_input[14],vec_input[15],vec_input[16]))
        f_1.write('! number of particles'+'\n')
        f_1.write('%d\n' % (particle_num) )
        f_1.write('! specify: pnm/snm, input type: density/kfermi'+'\n')
        f_1.write(matter_type+', density'+'\n')
        f_1.write('! specify boundary conditions (PBC/TABC/TABCsp/subspace_cal/subspace_cal_dens/solve_general_EV)'+'\n')
        f_1.write('hf_emulator'+'\n')
        f_1.write('! dens/kf, ntwist,  nmax'+'\n')
        f_1.write('%.12f, 1, %d\n' % (density, nmax))
        f_1.write('! specify cluster approximation: CCD, CCDT'+'\n')
        f_1.write('CCD'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 3'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('394, 4'+'\n')



######################################################
######################################################
### call hf_emulator
######################################################
######################################################
def call_hf_emulator(vec_input,in_dir,out_dir):
    neutron_num  = 14  #test
    particle_num = 28
    density      = 0.16
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2 #test

    generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax)
    os.system('./'+nucl_matt_exe+' '+in_dir+' > '+out_dir)

######################################################
######################################################
### call hf_emulator
######################################################
######################################################
def read_hf_energy_per_A(in_dir):
    with open(in_dir,'r') as f_1:
        count = len(open(in_dir,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range (0, count):
            if ( re.search('3NF vacuum expectation value', data[loop1],flags=0) != wtf):
                print(data[loop1+1]) 
                temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                hf_energy = float(temp_2[1])
                break
    return  hf_energy

######################################################
######################################################
### save hf_energy
######################################################
######################################################
def save_hf_energy(out_dir,hf_energy):
    with open(out_dir,'w') as f_1:
        f_1.write(str(hf_energy))



######################################################
######################################################
### generate emulator_matrix
######################################################
######################################################
def generate_emulator_matrix():
    LEC_all_matrix = np.zeros(LEC_number) # hf one-dimension matrix
    LEC      = np.zeros(LEC_number)
    call_hf_emulator(LEC,"ccm_in_test","a.out")
    C_matrix = read_hf_energy_per_A("a.out")
    out_dir = "./emulator/LEC_hf_C_matrix"
    save_hf_energy(out_dir,C_matrix)

    for loop1 in range(LEC_number):
        LEC = np.zeros(LEC_number)
        LEC[loop1] = 1 
        call_hf_emulator(LEC,"ccm_in_test","a.out")
        H_matrix = read_hf_energy_per_A("a.out")
        LEC_all_matrix[loop1] = H_matrix - C_matrix
        out_dir = "./emulator/LEC_hf_"+str(loop1+1)+"_matrix"
        save_hf_energy(out_dir,LEC_all_matrix[loop1])

######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def hf_emulator(LEC_target):
    LEC_all_matrix = np.zeros(LEC_number) # hf one-dimension matrix
    C = np.loadtxt("./emulator/LEC_hf_C_matrix")
    for loop1 in range(LEC_number):
        LEC_all_matrix[loop1] = np.loadtxt("./emulator/LEC_hf_"+str(loop1+1)+"_matrix")
    H = 0 
    for loop1 in range(LEC_number):
        H = H + LEC_target[loop1] * LEC_all_matrix[loop1]
    H = H + C
    return H
   
def read_LEC(file_path):
    LEC = np.zeros(LEC_number)
    with open(file_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            if ( re.search('cE and cD', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                LEC[0] = float(temp_1[0])
                LEC[1] = float(temp_1[1])
            if ( re.search('LEC ci', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                LEC[2] = float(temp_1[0])
                LEC[3] = float(temp_1[1])
                LEC[4] = float(temp_1[2])
                LEC[5] = float(temp_1[3])
            if ( re.search('c1s0 & c3s1', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                LEC[6] = float(temp_1[0])
                LEC[7] = float(temp_1[1])
                LEC[8] = float(temp_1[2])
                LEC[9] = float(temp_1[3])
            if ( re.search('cnlo', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                LEC[10] = float(temp_1[0])
                LEC[11] = float(temp_1[1])
                LEC[12] = float(temp_1[2])
                LEC[13] = float(temp_1[3])
                LEC[14] = float(temp_1[4])
                LEC[15] = float(temp_1[5])
                LEC[16] = float(temp_1[6])
    return LEC    

######################################################
######################################################
#### MAIN
######################################################
######################################################
#subspace_dimension = 64
LEC_number = 17
nucl_matt_exe = './prog_ccm.exe'

#generate_emulator_matrix()
LEC = read_LEC("ccm_in_DNNLO394")
print(hf_emulator(LEC))

