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
magic_no = 2


######################################################
######################################################
### generate infile for solve_general_EV
######################################################
######################################################
def generate_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write('3, 450\n')
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
        f_1.write('solve_general_EV'+'\n')
        f_1.write('! dens/kf, ntwist,  nmax'+'\n')
        f_1.write('%.12f, 1, %d\n' % (density, nmax))
        f_1.write('! specify cluster approximation: CCD, CCDT'+'\n')
        f_1.write('CCD(T)'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 3'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('450, 3'+'\n')



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

    generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax)
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



######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator(LEC_target):
    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_number,subspace_dimension,subspace_dimension))
    N = np.loadtxt("H_matrix.txt")
    C = np.loadtxt("C_matrix.txt")
    for loop1 in range(LEC_number):
        input_file("H_matrix.txt",H_matrix[loop1,:,:])
    
    #H = LECs[0]*H_matrix + K_matrix
    for loop1 in range(LEC_number):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C

    print("H="+str(H))
    print("rank of N ="+str(np.linalg.matrix_rank(N_matrix)))
    N = np.matrix(N_matrix)
    #Ni = N.I
    #print (N)
    #Ni = np.linalg.inv(N)
    #
    #print (np.dot(Ni,N_matrix))
    #print (Ni*N_matrix)
    
    #Ni_dot_H = np.dot(Ni,H)
    #D,V = np.linalg.eig(Ni_dot_H)
    #print (Ni_dot_H)
    #print ("D="+str(D))
    #print ("V="+str(V))
    
    eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    
    loop2 = 0
    for loop1 in range(subspace_dimension):
        ev = eigvals[loop1] 
        if ev.imag != 0:
            continue
    #    if ev.real < 0:
    #        continue
        loop2 = loop2+1
    
    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(subspace_dimension):
        ev = eigvals[loop1] 
        if ev.imag != 0:
            continue
    #    if ev.real < 0:
    #        continue
        ev_all[loop2] = ev.real
        loop2 = loop2+1
    
    
    ev_sorted = sorted(ev_all)
    print('eigvals='+str (ev_sorted))
    #print('eigvec_L='+str (eigvec_L))
    #print('eigvec_0='+str (eigvec_0))
    
    
    print('eigvals_gs='+str (ev_sorted[1]))
    
    
    
    #D,V = np.linalg.eig(H_matrix)
    #print ("D="+str(D))
    #print(np.linalg.matrix_rank(N_matrix))
    #print(np.linalg.matrix_rank(H_matrix))
    
    
    #print(N_matrix)
    #print(H_matrix)   



######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 64
LEC_number = 17
nucl_matt_exe = './prog_ccm.exe'

generate_emulator_matrix(subspace_dimension)


