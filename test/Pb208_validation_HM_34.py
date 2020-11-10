import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
from scipy import interpolate
from scipy import optimize
import seaborn as sns
import random
from random import choice
import matplotlib.colors as mcolors
colors=list(mcolors.TABLEAU_COLORS.keys())

######################################################
######################################################
### read LECs set from file
######################################################
######################################################
def read_LEC(file_path):
    LEC = np.zeros(LEC_num)
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
#### read in H N C matries 
######################################################
######################################################
def read_emulator_matrices(database_dir):
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"norm_wave4_208Pb_A2_16_HF_c1_nnlo394_delta_N12E28_hw10_Pb208.dat"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"hbar_wave4_208Pb_A2_16_HF_const_nnlo394_delta_N12E28_hw10_Pb208.dat"
    C = np.loadtxt(in_dir)

    H_matrix[0,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_cE_nnlo394_delta_N12E28_hw10_Pb208.dat") - C
    H_matrix[1,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_cD_nnlo394_delta_N12E28_hw10_Pb208.dat") - C
    H_matrix[2,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_c1_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[3,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_c2_nnlo394_delta_N12E28_hw10_Pb208.dat") - C      
    H_matrix[4,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_c3_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[5,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_c4_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[6,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_Ct1S0pp_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[7,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_Ct1S0np_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[8,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_Ct1S0nn_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[9,:,:]  = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_Ct3S1_nnlo394_delta_N12E28_hw10_Pb208.dat") -C
    H_matrix[10,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C1S0_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[11,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C3P0_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[12,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C1P1_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[13,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C3P1_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[14,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C3S1_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[15,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_CE1_nnlo394_delta_N12E28_hw10_Pb208.dat")-C
    H_matrix[16,:,:] = np.loadtxt(database_dir+"hbar_wave4_208Pb_A2_16_HF_C3P2_nnlo394_delta_N12E28_hw10_Pb208.dat") -C           
    return H_matrix, C, N


######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator(LEC_target,subtract):
    H = np.zeros((subspace_dimension,subspace_dimension))
#    N = np.zeros((subspace_dimension,subspace_dimension))
#    C = np.zeros((subspace_dimension,subspace_dimension))
#    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
#    in_dir = database_dir+"N_matrix.txt"
#    N = np.loadtxt(in_dir)
#    in_dir = database_dir+"C_matrix.txt"
#    C = np.loadtxt(in_dir)
#    for loop1 in range(LEC_num):
#        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
#        H_matrix[loop1,:,:] = np.loadtxt(in_dir)
#    #H = LECs[0]*H_matrix + K_matrix
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
        print("LEC="+str(LEC_target[loop1]))
        print("matrix="+str(H_matrix[loop1,:,:]))
        print("\n")
    H = H + C

    print("H="+str(H))
#    eigvals,eigvec = spla.eig(N)
#    print ("N eigvals = "+str(sorted(eigvals)))


##### without subtract
    subtract_1 = subtract
    H[subtract] = 0
    H[:,subtract] = 0
    N[subtract] = 0
    N[:,subtract] = 0
    N[subtract,subtract] = 1

    print("shape of H ="+str(H.shape))
    print("rank of N ="+str(np.linalg.matrix_rank(N)))

#    np.savetxt('H.test',H,fmt='%.10f')
#    np.savetxt('N.test',N,fmt='%.10f')
#    H = np.loadtxt('H.test')
#    N = np.loadtxt('N.test')
### solve the general eigval problem
    eigvals,eigvec_L, eigvec_R = spla.eig(H,N,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals)
    eigvals  = eigvals[x]
    eigvec_R = eigvec_R.T
    eigvec_R = eigvec_R[x]

### drop states with imaginary part
    eigvals_new   = eigvals[np.where(abs(eigvals.imag) < 0.01)]
    eigvec_R_new =  eigvec_R[np.where(abs(eigvals.imag)< 0.01)]

    print(eigvals)

    with open("emulator.wf",'w') as f_1:
        #f_2.write('ccd = %.12f     emulator = %.12f   all =' % (ccd_cal, emulator_cal))
        f_1.write('################################\n')
        f_1.write('#### emulator wave function ####\n')
        f_1.write('################################\n')
        f_1.write('all eigvals: \n')
        f_1.write(str(eigvals))
        f_1.write('\n')
        f_1.write('\n')
        for loop1 in range(len(eigvals)):
            if (eigvals[loop1].real != 0):
                f_1.write('################################\n')
                f_1.write('state %d -- eigvals: %r \n' % (loop1,eigvals[loop1]))
                for loop2 in range(np.size(eigvec_R,1)):
                    f_1.write('%2d: %.5f%%  ' % (loop2+1,abs(eigvec_R[loop1,loop2])**2*100))
                    if ((loop2+1)%5==0): f_1.write('\n')
                f_1.write('\n################################\n')
    return eigvals_new , eigvec_R_new

######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator2(subtract_count,LEC_target,tolerance):
    split = 5
    #vote_need = 5

    H = np.zeros((subspace_dimension,subspace_dimension))
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C

    subtract = [subtract_count]
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)
#    H[subtract] = 0
#    H[:,subtract] = 0
#    N[subtract] = 0
#    N[:,subtract] = 0
#    N[subtract,subtract] = 1
#    H1 = H
#    N1 = N
### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

### drop states with imaginary part
    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]
### divide training samples into few parts

    each_split = round(len(H1)/split)
    eigvals       = []
    eigvals_len = np.zeros(split)
    for loop in range(split):
        if (loop == split-1):
            subtract = range(loop*each_split,len(H1))
        else:
            subtract = range(loop*each_split,(loop+1)*each_split)
        H2 = np.delete(H1,subtract,axis = 0)
        H2 = np.delete(H2,subtract,axis = 1)
        N2 = np.delete(N1,subtract,axis = 0)
        N2 = np.delete(N2,subtract,axis = 1)

###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals.append(eigvals_2[:])

    score  = np.zeros((len(eigvals_1),split))
#    score1 = np.zeros(len(eigvals_1))
#    score2 = np.zeros(len(eigvals_1))
#    score3 = np.zeros(len(eigvals_1))
#    score4 = np.zeros(len(eigvals_1))
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < tolerance ):
                    #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                    score[loop2,loop] = score[loop2,loop] + 1
#    # vote for the lowest ture state
#    for loop in range(len(eigvals_1)):
#        vote = 0
#        for loop1 in range(split):
#            if score[loop,loop1] > 0:
#                vote = vote + 1
#
#    for loop in range(len(eigvals_1)):
#        if vote[loop] == max(vote):
#            ev_ultra = eigvals_1[loop]
#            break
    # vote for the lowest ture state
    vote = np.zeros(len(eigvals_1))
    for loop in range(len(eigvals_1)):
        for loop1 in range(split):
            if score[loop,loop1] > 0:
                vote[loop] = vote[loop]+1

    for loop in range(len(eigvals_1)):
        if vote[loop] == max(vote) :
            ev_ultra = eigvals_1[loop]
            break

### drop states with imaginary part
    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]
    #return eigvals_new , eigvals_R_new
    eigvals_1 = eigvals_1[np.where(eigvals_1.real!=0)]
    eigvals_new_1 = eigvals_new_1[np.where(eigvals_new_1.real!=0)]
    #return eigvals_1[0].real

    return ev_ultra.real, eigvals_1, vote 






######################################################
######################################################
#### emulator3
######################################################
######################################################
def emulator3(LEC_target,tolerance):
    split = 3
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C

### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H,N,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

### drop states with imaginary part
    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

### divide training samples into few parts

    each_split = round(len(H)/split)
    eigvals       = []
    eigvals_len = np.zeros(split)
    for loop in range(split):
        if (loop == split-1):
            remain = range(loop*each_split,len(H))
        else:
            remain = range(loop*each_split,(loop+1)*each_split)
        subtract = np.delete(range(subspace_dimension),remain)
        H2 = np.delete(H,subtract,axis = 0)
        H2 = np.delete(H2,subtract,axis = 1)
        N2 = np.delete(N,subtract,axis = 0)
        N2 = np.delete(N2,subtract,axis = 1)
        #print("remain"+str(remain))
###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals.append(eigvals_2[:])

    score  = np.zeros((len(eigvals_1),split))
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < tolerance ):
                    score[loop2,loop] = score[loop2,loop] + 1

    # vote for the lowest ture state
    for loop in range(len(eigvals_1)):
        flag = 1
        for loop1 in range(split):
            if score[loop,loop1] <= 0:
                flag = flag * 0
        if flag == 1 :
            ev_ultra = eigvals_1[loop]
            break

    return ev_ultra.real

######################################################
######################################################
#### emulator4
######################################################
######################################################
def emulator4(subtract_count,LEC_target,tolerance):
    split = 4
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    subtract = [subtract_count]
    #print(subtract)

#    print(H)
###########################################################################
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)

#    print(H1)
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)
### sort with eigval
    eigvals_1 = eigvals_1[np.where(eigvals_1 != 0 )]
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

#    print(eigvals_1)
#############################################################################
#    H1 = H.copy()
#    N1 = N.copy()
#
#    H1[subtract] = 0
#    H1[:,subtract] = 0
#    N1[subtract] = 0
#    N1[:,subtract] = 0
#    N1[subtract,subtract] = 1
#    print(H1)
### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    eigvals_1 = eigvals_1[np.where(eigvals_1 != 0 )]
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]
############################################################################
### drop states with imaginary part
   # eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
   # eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

### divide training samples into few parts

    each_split = round(len(H1)/split)
    eigvals       = []
    eigvals_len = np.zeros(split)
    for loop in range(split):
        if (loop == split-1):
            remain = range(loop*each_split,len(H1))
        else:
            remain = range(loop*each_split,(loop+1)*each_split)
        subtract = np.delete(range(len(H1)),remain)

        H2 = np.delete(H1,subtract,axis = 0)
        H2 = np.delete(H2,subtract,axis = 1)
        N2 = np.delete(N1,subtract,axis = 0)
        N2 = np.delete(N2,subtract,axis = 1)
      #  H2 = H1.copy()
      #  N2 = N1.copy()
    
      #  H2[subtract] = 0
      #  H2[:,subtract] = 0
      #  N2[subtract] = 0
      #  N2[:,subtract] = 0
      #  N2[subtract,subtract] = 1


        #print("remain"+str(remain))
###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
        eigvals_2 = eigvals_2[np.where(eigvals_2 != 0 )]
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
      #  print(eigvals_2)
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals.append(eigvals_2[:])

    score  = np.zeros((len(eigvals_1),split))
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < tolerance ):
                    score[loop2,loop] = score[loop2,loop] + 1

    # vote for the lowest ture state
    vote = np.zeros(len(eigvals_1))
    for loop in range(len(eigvals_1)):
        for loop1 in range(split):
            if score[loop,loop1] > 0:
                vote[loop] = vote[loop]+1

    for loop in range(len(eigvals_1)):
        if vote[loop] == max(vote) :
            ev_ultra = eigvals_1[loop]
            break
    return ev_ultra.real, eigvals_1, vote


######################################################
######################################################
#### emulator5
######################################################
######################################################
def emulator5(subtract_count,LEC_target,tolerance,ccsdt_1):
    split = 100
    sample_each_slice = 30
    vote_need = round(0.70*split)

    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    subtract = [subtract_count]
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)


### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

### drop states with imaginary part
#    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
#    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

### divide training samples into few parts
    remain = range(0,subspace_dimension-1)

    eigvals       = []
    eigvals_len = np.zeros(split)
    for loop in range(split):
        slice_1 = random.sample(remain,sample_each_slice)
        subtract = np.delete(range(subspace_dimension),slice_1)
        H2 = np.delete(H1,subtract,axis = 0)
        H2 = np.delete(H2,subtract,axis = 1)
        N2 = np.delete(N1,subtract,axis = 0)
        N2 = np.delete(N2,subtract,axis = 1)
        #print("remain"+str(remain))
###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals.append(eigvals_2[:])

    score  = np.zeros((len(eigvals_1),split))
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < tolerance ):
                    score[loop2,loop] = score[loop2,loop] + 1

#    # vote for the lowest ture state
#    for loop in range(len(eigvals_1)):
#        vote = 0
#        for loop1 in range(split):
#            if score[loop,loop1] > 0:
#                vote = vote + 1
#        if vote >= vote_need :
#            ev_ultra = eigvals_1[loop]
#            break

    # vote for the lowest ture state
    vote = np.zeros(len(eigvals_1))
    for loop in range(len(eigvals_1)):
        for loop1 in range(split):
            if score[loop,loop1] > 0:
                vote[loop] = vote[loop]+1
   # print("ccsdt-1"+str(ccsdt_1))
   # print("eigvals_1="+str(eigvals_1))
   # print("vote="+str(vote))
#    for loop in range(len(eigvals_1)):
#        if ( np.abs((ccsdt_1 - eigvals_1[loop].real )/ccsdt_1) < 0.01 ):
#            print("emulator_should_be="+str(eigvals_1[loop]))
#            print("the vote it gets="+str(vote[loop]))

    for loop in range(len(eigvals_1)):
        if vote[loop] >= vote_need :
            ev_ultra = eigvals_1[loop]
            break
            

    return ev_ultra.real, eigvals_1,vote




######################################################
######################################################
#### validation
######################################################
######################################################
def validation(tolerance):
    emulator_data=[]
    ccsdt_data=[]
    file_path       = my_path + "LEC_read3.txt"
    validation_path = my_path + "validation/Pb208/ccsdt_Pb208_34_points.txt" 
    with open(validation_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        file_count = np.zeros(count-1)
        ccsdt_1      = np.zeros(count-1)
        for loop1 in range(0,count-1):
            temp_1     = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count[loop1] = round(float(temp_1[5]))
            ccsdt_1[loop1]      = float(temp_1[7])
    ccsdt_1 = ccsdt_1[np.argsort(file_count)]    
    #print(file_count)
    #print(ccsdt_1)


    with open(file_path,'r') as f_2:
        count = len(open(file_path,'rU').readlines())
        data = f_2.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(1,count):
            temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            LEC[0]    = temp_1[16]
            LEC[1]    = temp_1[15]
            LEC[2:6]  = temp_1[11:15]
            LEC[6:10] = temp_1[0:4]
            LEC[10:17]= temp_1[4:11]
            #gs = emulator3(LEC,0.015)
            gs,eigvals,vote = emulator5(loop1-1,LEC,tolerance,ccsdt_1[loop1-1])
            #gs,eigvals,vote1 = emulator4(loop1-1,LEC,tolerance)
            #gs,eigvals,vote2 = emulator2(loop1-1,LEC,tolerance)
            #print("ccsdt-1"+str(ccsdt_1[loop1-1]))
            #print("eigvals_1="+str(eigvals[0:15]))
            #print("vote1="+str(vote1[0:15]))
            #print("vote2="+str(vote2[0:15]))
            #for loop2 in range(len(eigvals)):
            #    if (vote1[loop2]== max(vote1)) and (vote2[loop2] == max(vote2)):
            #        gs = eigvals.real[loop2]
            #        break             

            #print("emulator ="+str(eigvals.real[loop2]))
            emulator_data.append(gs)
            ccsdt_data.append(ccsdt_1[loop1-1])
    return emulator_data,ccsdt_data

######################################################
######################################################
#### plot
######################################################
######################################################
def plot_1(emulator_data,ccsdt_data,tolerance):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(12,5))
    #plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(121)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208")

    x_list_1 =  ccsdt_data
    y_list_1 =  emulator_data

    uper_range   = -1000
    lower_range  = -1800
    gap          = 200

    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

#    plt.xlim((lower_range,uper_range))
#    plt.ylim((lower_range,uper_range))
#
#    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
#    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(122)
    ax.set_title("Pb208")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    x_list_1 = ((np.array(emulator_data) - np.array(ccsdt_data))/abs(np.array(ccsdt_data)))

    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=100,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
#    plt.xlim((-0.2,0.2))
#    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)
    plot_path =  'Pb208_CV_emulator5_tolerance_%s_test3.pdf' %(str(tolerance))
    plt.savefig(plot_path)

######################################################
######################################################
#### plot 2
######################################################
######################################################
def plot_2(len_1,count, emulator_data_,ccsdt_data_,tolerances):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(12,5))
    #plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(121)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208")
    colors= ['darkblue', 'darkorange', 'seagreen', 'r','violet']
    for loop in range(count):
        x_list_1 =  ccsdt_data_[loop]
        y_list_1 =  emulator_data_[loop]

        uper_range   = -1000
        lower_range  = -1800
        gap          = 200
        ax.scatter (x_list_1, y_list_1,color=colors[loop],marker = 's',zorder=0.5, label="tolerance="+str(tolerances[loop])  )

    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
        #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

    plt.legend(loc="lower right")
    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))

    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(122)
    ax.set_title("Pb208")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    for loop in range(count):
        x_list_1 = ((np.array(emulator_data_[loop]) - np.array(ccsdt_data_[loop]))/abs(np.array(ccsdt_data_[loop] )))
        sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
        sns.distplot(x_list_1,bins=10,kde_kws={"color":colors[loop], "lw":0 }, hist_kws={ "color": colors[loop]},label="tolerance="+str(tolerances[loop]))
        print("std = "+str(np.std(x_list_1,ddof=1)))
    plt.legend()
    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
#    plt.xlim((-0.2,0.2))
#    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)
    plot_path =  'Pb208_CV_emulator5_all_tolerance.pdf'
    plt.savefig(plot_path)



######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 34
LEC_num = 17
LEC = np.zeros(LEC_num)
subtract = []
remain   = []

N = np.zeros((subspace_dimension,subspace_dimension))
C = np.zeros((subspace_dimension,subspace_dimension))
H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))

my_path ="./"
database_dir = "/home/slime/subspace_CC/test/Pb208/Pb208_spcc/cc_output/split_files_wave4_208Pb_A2_16_HF/"

H_matrix, C, N = read_emulator_matrices(database_dir)

file_path   = "ccm_in_DNNLO394"
LEC         = read_LEC(file_path)

tolerance   = 0.01
gs_DNNLO394 = emulator4(0,LEC_target=LEC,tolerance = tolerance)

emulator_data_ = [] 
ccsdt_data_ = []
tolerances  = [] 
count = 3
for loop in [0.005,0.01,0.03]:
    #tolerance = 0.005 + 0.0025 *(loop+1) 
    tolerance = loop
    print(tolerance)
    emulator_data,ccsdt_data = validation(tolerance=tolerance)
    emulator_data_.append(emulator_data)
    ccsdt_data_.append(ccsdt_data)
    tolerances.append(tolerance)
    


len_1 = len(ccsdt_data)
print(len_1)
    #plot_1(emulator_data,ccsdt_data,tolerance=tolerance)
print(emulator_data_[0])
print(emulator_data_[1])
plot_2(len_1,count,emulator_data_,ccsdt_data_,tolerances)

#gs_DNNLO394,test = emulator(LEC,subtract)
#print(LEC)
print(gs_DNNLO394)


