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
### read LECs set from file
######################################################
######################################################
def read_LEC_batch(file_path): 
    LEC_batch = []
    my_LEC_label = ['cE','cD','c1','c2','c3','c4','Ct1S0pp','Ct1S0np','Ct1S0nn','Ct3S1','C1S0','C3P0','C1P1','C3P1','C3S1','CE1','C3P2']    
    with open(file_path,'r') as f:
        count = len(open(file_path,'rU').readlines())
        data = f.readlines()
        wtf = re.match('#', 'abc',flags=0)
        LEC_label = data[0].split()
        LEC_label = LEC_label[1::]

        #print("LEC_label"+str(LEC_label))
        x = []
        for loop1 in range(len(my_LEC_label)):
            for loop2 in range(len(my_LEC_label)):
                if LEC_label[loop2]==my_LEC_label[loop1]:
                    x.append(loop2)
        #print("x=",x)
        for loop1 in range(1,count):
            myarray = np.fromstring(data[loop1],dtype=float, sep=' ')
            myarray = myarray[x]
            LEC_batch.append(myarray)
            #temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            #LEC_batch[loop1-1][0:16]    = temp_1[0:16]
        #print("LEC_batch"+str(LEC_batch))
    return LEC_batch


############################################
## read energy validation data 
############################################
def read_validation_data(validation_path):
    with open(validation_path,'r') as f_1:
        count = len(open(validation_path,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        file_count = np.zeros(count)
        validation_data = np.zeros(count)
        for loop1 in range(0,count):
            temp_1     = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count[loop1]      = round(float(temp_1[0]))
            validation_data[loop1] = float(temp_1[1])
    validation_data = validation_data[np.argsort(file_count)]
    return validation_data

######################################################
######################################################
#### read in H N C matries 
######################################################
######################################################
def read_emulator_matrices(database_dir):
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"norm_5percent34_c1_nnlo394_delta_mass_208_N10E22_hw12.dat"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"hbar_5percent34_const_nnlo394_delta_mass_208_N10E22_hw12.dat"
    C = np.loadtxt(in_dir)
    LEC_label = ['cE','cD','c1','c2','c3','c4','Ct1S0pp','Ct1S0np','Ct1S0nn','Ct3S1','C1S0','C3P0','C1P1','C3P1','C3S1','CE1','C3P2']    

    str1="hbar_5percent34_"
    str2="_nnlo394_delta_mass_208_N10E22_hw12.dat"
    for loop in range(17):
        H_matrix[loop,:,:]  = np.loadtxt(database_dir+str1+LEC_label[loop]+str2) - C
    
    return H_matrix, C, N


######################################################
######################################################
### calculate Rn2 Rp2
######################################################
######################################################
def observable_batch_cal(eigvec_R,N,observable_matrix):
    norm = []
    observable_batch = []
    for loop in range(len(eigvec_R)):
        norm.append(np.dot(np.dot(eigvec_R[loop].T,N),eigvec_R[loop]))
    for loop in range(len(eigvec_R)):
        observable_batch.append(np.dot( np.dot(eigvec_R[loop].T, observable_matrix), eigvec_R[loop])/norm[loop])
    return observable_batch

def observable_batch_cal2(eigvec_L,eigvec_R,N,observable_matrix):
    norm = []
    observable_batch = []
    for loop in range(len(eigvec_R)):
        norm.append(np.dot(np.dot(eigvec_L[loop].T,N),eigvec_R[loop]))
    for loop in range(len(eigvec_R)):
        observable_batch.append(np.dot( np.dot(eigvec_L[loop].T, observable_matrix), eigvec_R[loop])/norm[loop])
    return observable_batch




######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator(LEC_target):
    #subtract = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 18, 20, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]
    subtract = []
    H = np.zeros((subspace_dimension,subspace_dimension))
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C

    eigvals_all,eigvec_L_all, eigvec_R_all = spla.eig(H,N,left =True,right=True)
### sort with eigval
    x = np.argsort(eigvals_all)
    eigvals_all  = eigvals_all[x]
    eigvec_L_all = eigvec_L_all.T
    eigvec_L_all = eigvec_L_all[x]
    eigvec_R_all = eigvec_R_all.T
    eigvec_R_all = eigvec_R_all[x]


##### with subtract
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)

    #print("shape of H ="+str(H.shape))
    #print("rank of N ="+str(np.linalg.matrix_rank(N)))

#    np.savetxt('H.test',H,fmt='%.10f')
#    np.savetxt('N.test',N,fmt='%.10f')
#    H = np.loadtxt('H.test')
#    N = np.loadtxt('N.test')
### solve the general eigval problem
    eigvals,eigvec_L, eigvec_R = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals)
    eigvals  = eigvals[x]
    eigvec_L = eigvec_L.T
    eigvec_L = eigvec_L[x]


    eigvec_R = eigvec_R.T
    eigvec_R = eigvec_R[x]

### drop states with imaginary part
   # eigvals_new   = eigvals[np.where(abs(eigvals.imag) < 0.01)]
   # eigvec_R_new =  eigvec_R[np.where(abs(eigvals.imag)< 0.01)]

#    print(eigvals)

#    with open("emulator.wf",'w') as f_1:
#        #f_2.write('ccd = %.12f     emulator = %.12f   all =' % (ccd_cal, emulator_cal))
#        f_1.write('################################\n')
#        f_1.write('#### emulator wave function ####\n')
#        f_1.write('################################\n')
#        f_1.write('all eigvals: \n')
#        f_1.write(str(eigvals))
#        f_1.write('\n')
#        f_1.write('\n')
#        for loop1 in range(len(eigvals)):
#            if (eigvals[loop1].real != 0):
#                f_1.write('################################\n')
#                f_1.write('state %d -- eigvals: %r \n' % (loop1,eigvals[loop1]))
#                for loop2 in range(np.size(eigvec_R,1)):
#                    f_1.write('%2d: %.5f%%  ' % (loop2+1,abs(eigvec_R[loop1,loop2])**2*100))
#                    if ((loop2+1)%5==0): f_1.write('\n')
#                f_1.write('\n################################\n')
    #return eigvals_new , eigvec_R_new
    #print(eigvals)


########################################################
### calcualte Rn2 Rp2
########################################################
    xx = np.where( abs((eigvals_all.real - eigvals.real[0])/eigvals.real[0]) < 0.02 )
    #print("eigvals_all"+str(eigvals_all.real))
    #print("eigvals"+str(eigvals.real[0]))

    print("xx="+str(xx))
    yy = int(round(len(xx[0])/2))
    print("yy="+str(yy))
    xxx = xx[0][yy]
    print("xxx="+str(xxx))
    ev_ultra = eigvals_all[xxx]
    ev_ultra_vec_R = eigvec_R_all[xxx]
    ev_ultra_vec_L = eigvec_L_all[xxx]

    #print("ev_ultra_vec_R="+str(ev_ultra_vec_R))
    subtract = []
    norm = np.dot(np.dot(ev_ultra_vec_R.T,N),ev_ultra_vec_R)
    Rn2_matrix_1 = np.delete(Rn2_matrix,subtract,axis = 0)
    Rn2_matrix_1 = np.delete(Rn2_matrix_1,subtract,axis = 1)
    Rp2_matrix_1 = np.delete(Rp2_matrix,subtract,axis = 0)
    Rp2_matrix_1 = np.delete(Rp2_matrix_1,subtract,axis = 1)
 
    Rn2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix_1), ev_ultra_vec_R)/norm
    Rp2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix_1), ev_ultra_vec_R)/norm
    ev_ultra_1 =np.dot( np.dot(ev_ultra_vec_R.T, H), ev_ultra_vec_R)/norm
########################################################


########################################################
### calcualte Rn2 Rp2
########################################################
#    ev_ultra_vec_R = eigvec_R[0]
#    ev_ultra_vec_L = eigvec_L[0]
#
#    norm = np.dot(np.dot(ev_ultra_vec_R.T,N1),ev_ultra_vec_R)
#    Rn2_matrix_1 = np.delete(Rn2_matrix,subtract,axis = 0)
#    Rn2_matrix_1 = np.delete(Rn2_matrix_1,subtract,axis = 1)
#    Rp2_matrix_1 = np.delete(Rp2_matrix,subtract,axis = 0)
#    Rp2_matrix_1 = np.delete(Rp2_matrix_1,subtract,axis = 1)
# 
#    Rn2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix_1), ev_ultra_vec_R)/norm
#    Rp2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix_1), ev_ultra_vec_R)/norm
#    ev_ultra_1 =np.dot( np.dot(ev_ultra_vec_R.T, H1), ev_ultra_vec_R)/norm
     #ev_ultra   = eigvals.real[0]
########################################################


    #print("norm="+str(norm))
    #print("Rn2="+str(Rn2))
    #print("ev_ultra_vec_R="+str(ev_ultra_vec_R))
    #print("ev_ultra_vec_L="+str(ev_ultra_vec_L))
    #print("ev_ultra_1="+str(ev_ultra_1))
    #print("ev_ultra.real="+str(eigvals.real[0]))

    return ev_ultra,Rn2, Rp2

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
    split = 5
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
def emulator4(subtract_count,LEC_target,tolerance,cc_gs,cc_Rn2_temp,cc_Rp2_temp):
    split = 4 
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    #print(subtract)

###########################################################################
#   with subtract
###########################################################################
#    subtract = [subtract_count]
#    H1 = np.delete(H,subtract,axis = 0)
#    H1 = np.delete(H1,subtract,axis = 1)
#    N1 = np.delete(N,subtract,axis = 0)
#    N1 = np.delete(N1,subtract,axis = 1)
    H1 = H
    N1 = N
#    print(H1)
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvals_1_real_x = np.where(abs(eigvals_1.imag)<0.01)
    eigvals_1_real = eigvals_1[eigvals_1_real_x]

    eigvec_L_1 = eigvec_L_1.T
    eigvec_L_1 = eigvec_L_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

    #Rn2_matrix_c = np.array(Rn2_matrix ,dtype=np.complex)
    #Rp2_matrix_c = np.array(Rp2_matrix ,dtype=np.complex)

    Rn2_all = observable_batch_cal(eigvec_R_1,N1,Rn2_matrix)
    Rn2_all = np.array(Rn2_all)
    Rp2_all = observable_batch_cal(eigvec_R_1,N1,Rp2_matrix)
    Rp2_all = np.array(Rp2_all)
    #Rs_matrix = np.power(Rn2_matrix_c,0.5)-np.power(Rp2_matrix_c,0.5)
    #Rs_all = observable_batch_cal2(eigvec_L_1,eigvec_R_1,N1,Rs_matrix)
    #print("Rn2_matrix ="+str((Rn2_matrix_c)))
    #print("Rn2_matrix_size ="+str(Rn2_matrix.shape))
    #print("Rs_matrix ="+str(Rs_matrix))
    #print("Rs_matrix_size ="+str(Rs_matrix.shape))
    #print("Rs_all="+str(Rs_all))
#    eigvals_all = observable_batch_cal(eigvec_R_1,N1,H1)

    #print("#####################")
    #print("eigvals_all="+str(eigvals_all) )


#############################################################################
### drop states with imaginary part
############################################################################
   # eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
   # eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

#############################################################################
### divide training samples into few parts
#############################################################################
    each_split = round(len(H1)/split)
    eigvals_vote    = []
    Rn2_vote        = []
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
        Rn2_matrix_2 = np.delete(Rn2_matrix,subtract,axis = 0)
        Rn2_matrix_2 = np.delete(Rn2_matrix_2,subtract,axis = 1)

      #  H2 = H1.copy()
      #  N2 = N1.copy()
    
      #  H2[subtract] = 0
      #  H2[:,subtract] = 0
      #  N2[subtract] = 0
      #  N2[:,subtract] = 0
      #  N2[subtract,subtract] = 1

###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals_vote.append(eigvals_2[:])
        
        
        Rn2_2 = observable_batch_cal(eigvec_R_2,N2,Rn2_matrix_2)
        #Rp2_all = observable_batch_cal(eigvec_R_1,N1,Rp2_matrix)
        Rn2_vote.append(Rn2_2)

# score for gs energy
    score  = np.zeros((len(eigvals_1),split))
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals_vote[loop][loop1].real - eigvals_1[loop2].real )/eigvals_vote[loop][loop1]) < tolerance ):
                    score[loop2,loop] = score[loop2,loop] + 1

# score for radii 
    score_Rn2  = np.zeros((len(eigvals_1),split))
    Rn2_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((Rn2_vote[loop][loop1].real - Rn2_all[loop2].real )/Rn2_vote[loop][loop1]) < tolerance*0.1 ):
                    score_Rn2[loop2,loop] = score_Rn2[loop2,loop] + 1



#############################################################################
# vote for the lowest true state
#############################################################################
    vote = np.zeros(len(eigvals_1))
    for loop in range(len(eigvals_1)):
        for loop1 in range(split):
            if score[loop,loop1] > 0:
                vote[loop] = vote[loop]+1

    for loop in range(len(eigvals_1)):
        if vote[loop] == max(vote) :
            ev_ultra = eigvals_1[loop]
            #Rs_ultra = Rs_all[loop]
            if abs(ev_ultra.imag) > 0.01:
                xx = np.where((abs(eigvals_1.real) < abs(ev_ultra.real)*(1+0.03)) &\
                              (abs(eigvals_1.real) > abs(ev_ultra.real)*(1-0.03)) &\
                              (abs(eigvals_1.imag) < 0.01))
                xx = xx[0]
                if len(xx) == 0:
                    Rn2_ultra = Rn2_all[loop]
                    Rp2_ultra = Rp2_all[loop]
                    break    
                #print("xx"+str(xx))
                #print("eigvals_real"+str(eigvals_1[xx]))
                #print("Rn2_real"+str(Rn2_all[xx])) 
                Rn2_1 = Rn2_all[np.array(xx)] 
                Rp2_1 = Rp2_all[np.array(xx)]
                #print("Rn2_1"+str(Rn2_1[0])) 
                Rn2_ultra = Rn2_1[0]
                Rp2_ultra = Rp2_1[0]
                #Rn2_1 = Rn2_1.mean()
                #Rp2_1 = Rp2_1.mean()
                break 
            #ev_ultra_vec_R = eigvec_R_1[loop]
            #ev_ultra_vec_L = eigvec_L_1[loop]

            Rn2_ultra = Rn2_all[loop]
            Rp2_ultra = Rp2_all[loop]
            #ev_ultra_1 = eigvals_all[loop]
            #print("loop_energy="+str(loop))
            break

#############################################################################
# vote for the radii
#############################################################################
#    vote_Rn2 = np.zeros(len(eigvals_1))
#    for loop in range(len(eigvals_1)):
#        for loop1 in range(split):
#            if score_Rn2[loop,loop1] > 0:
#                vote_Rn2[loop] = vote_Rn2[loop]+1
#
#    for loop in range(len(eigvals_1)):
#        if vote_Rn2[loop] == max(vote_Rn2) :
#            Rn2_ultra = Rn2_all[loop]
#
#    #        print("loop_Rn2="+str(loop))
#            break


   # norm = np.dot(np.dot(ev_ultra_vec_R.T,N1),ev_ultra_vec_R) 
   # Rn2_2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix), ev_ultra_vec_R)/norm
   # Rp2_2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix), ev_ultra_vec_R)/norm
   # ev_ultra_2 =np.dot( np.dot(ev_ultra_vec_R.T, H1), ev_ultra_vec_R)/norm

    cc_Rs = np.sqrt(cc_Rn2_temp) - np.sqrt(cc_Rp2_temp)
    #print("cc_Rn2_temp="+str(cc_Rn2_temp))
    #print("cc_Rp2_temp="+str(cc_Rp2_temp))
    emulator_Rs = np.sqrt(Rn2_ultra) - np.sqrt(Rp2_ultra)
    print("cc_Rs="+str(cc_Rs))
    print("emulator_Rs="+str(emulator_Rs))
    #print("emulator_Rs_all="+str(Rs_ultra))
   # print("norm="+str(norm))
    if (abs(cc_Rs - emulator_Rs)/abs(cc_Rs)> 0.10):
        print("cc_gs="+str(cc_gs)) 
        print("ev_emulator="+str(ev_ultra)) 
        print("eigvals_1="+str(eigvals_1))
        print("cc_Rn2 = "+str(cc_Rn2_temp))
        print("Rn2_emulator="+str(Rn2_ultra))
        print("cc_Rp2 = "+str(cc_Rp2_temp))
        print("Rp2_emulator="+str(Rp2_ultra))
        #Rn2_all = Rn2_all[np.where(abs(Rn2_all.imag) < 0.01)]

        print("Rn2="+str(Rn2_all))
        # temp
       # Rn2_1 = 

        #return ev_ultra.real,100,100, ev_ultra, vote
   # print("ev_ultra_vec_R_1="+str(eigvec_R_1[loop]))
   # print("ev_ultra_vec_R_2="+str(ev_ultra_vec_R))
   ## #print("ev_ultra_vec_L="+str(ev_ultra_vec_L))
   # print("ev_ultra_1="+str(ev_ultra_1))
   # print("ev_ultra_2="+str(ev_ultra_2))
   # print("ev_ultra.real="+str(ev_ultra))

    return ev_ultra.real,Rn2_ultra,Rp2_ultra, ev_ultra, vote


######################################################
######################################################
#### emulator5
######################################################
######################################################
def emulator5(subtract_count,LEC_target,tolerance,cc_gs,cc_Rn2_temp,cc_Rp2_temp):
    split = 100
    sample_each_slice = 30
    vote_need = round(0.75*split)

    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    subtract = [subtract_count]
#    H1 = np.delete(H,subtract,axis = 0)
#    H1 = np.delete(H1,subtract,axis = 1)
#    N1 = np.delete(N,subtract,axis = 0)
#    N1 = np.delete(N1,subtract,axis = 1)
    H1 = H
    N1 = N

### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H,N,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvals_1_real_x = np.where(abs(eigvals_1.imag)<0.01)
    eigvals_1_real = eigvals_1[eigvals_1_real_x]

    eigvec_L_1 = eigvec_L_1.T
    eigvec_L_1 = eigvec_L_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

    Rn2_all = observable_batch_cal(eigvec_R_1,N1,Rn2_matrix)
    Rn2_all = np.array(Rn2_all)
    Rn2_all_real_x = np.where(abs(Rn2_all.imag)<0.0001)
    Rn2_all_real = Rn2_all[Rn2_all_real_x]


    Rp2_all = observable_batch_cal(eigvec_R_1,N1,Rp2_matrix)
    Rp2_all = np.array(Rp2_all)
    Rp2_all_real_x = np.where(abs(Rp2_all.imag)<0.0001)
    Rp2_all_real = Rp2_all[Rp2_all_real_x]


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

    for loop in range(len(eigvals_1)):
        if vote[loop] >= vote_need:
            ev_ultra = eigvals_1[loop]
            if abs(ev_ultra.imag) > 0.01:
                xx = np.where((abs(eigvals_1.real) < abs(ev_ultra.real)*(1+0.02)) &\
                              (abs(eigvals_1.real) > abs(ev_ultra.real)*(1-0.02)) &\
                              (abs(eigvals_1.imag) < 0.01))
                xx = xx[0]
                if len(xx) == 0:
                    Rn2_ultra = Rn2_all[loop]
                    Rp2_ultra = Rp2_all[loop]
         # temp test
                    Rn2_ultra = Rn2_all_real[1]
                    Rp2_ultra = Rp2_all_real[1]

                    break    
                #print("xx"+str(xx))
                #print("eigvals_real"+str(eigvals_1[xx]))
                #print("Rn2_real"+str(Rn2_all[xx])) 
                Rn2_1 = Rn2_all[np.array(xx)] 
                Rp2_1 = Rp2_all[np.array(xx)]
                #print("Rn2_1"+str(Rn2_1[0])) 
                Rn2_ultra = Rn2_1[0]
                Rp2_ultra = Rp2_1[0]
 # temp test
                Rn2_ultra = Rn2_all_real[1]
                Rp2_ultra = Rp2_all_real[1]


                #Rn2_1 = Rn2_1.mean()
                #Rp2_1 = Rp2_1.mean()
                break 
            #ev_ultra_vec_R = eigvec_R_1[loop]
            #ev_ultra_vec_L = eigvec_L_1[loop]

            Rn2_ultra = Rn2_all[loop]
            Rp2_ultra = Rp2_all[loop]
            #ev_ultra_1 = eigvals_all[loop]
            #print("loop_energy="+str(loop))
            break
#    for loop in range(len(eigvals_1)):
#        if vote[loop] >= vote_need :
#            ev_ultra = eigvals_1[loop]
#            ev_ultra_vec_R = eigvec_R_1[loop]
#            ev_ultra_vec_L = eigvec_L_1[loop]
#            break
#
#    norm = np.dot(np.dot(ev_ultra_vec_R.T,N),ev_ultra_vec_R) 
#    Rn2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix), ev_ultra_vec_R)/norm
#    Rp2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix), ev_ultra_vec_R)/norm

    cc_Rs = np.sqrt(cc_Rn2_temp) - np.sqrt(cc_Rp2_temp)
    #print("cc_Rn2_temp="+str(cc_Rn2_temp))
    #print("cc_Rp2_temp="+str(cc_Rp2_temp))
    emulator_Rs = np.sqrt(Rn2_ultra) - np.sqrt(Rp2_ultra)
    print("cc_Rs="+str(cc_Rs))
    print("emulator_Rs="+str(emulator_Rs))
    #print("emulator_Rs_all="+str(Rs_ultra))
   # print("norm="+str(norm))
    if (abs(cc_Rs - emulator_Rs)/abs(cc_Rs)> 0.10):
        print("cc_gs="+str(cc_gs)) 
        print("ev_emulator="+str(ev_ultra)) 
        print("eigvals_1="+str(eigvals_1))
        print("cc_Rn2 = "+str(cc_Rn2_temp))
        print("Rn2_emulator="+str(Rn2_ultra))
        print("cc_Rp2 = "+str(cc_Rp2_temp))
        print("Rp2_emulator="+str(Rp2_ultra))
        #Rn2_all = Rn2_all[np.where(abs(Rn2_all.imag) < 0.01)]

        print("Rn2="+str(Rn2_all))

    return ev_ultra.real,Rn2_ultra,Rp2_ultra,ev_ultra,vote


######################################################
######################################################
#### emulator6
######################################################
######################################################
def emulator6(subtract_count,LEC_target,tolerance):
    split = 3
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    subtract = [subtract_count]
    #print(subtract)

#    print(H)
###########################################################################
#    H1 = np.delete(H,subtract,axis = 0)
#    H1 = np.delete(H1,subtract,axis = 1)
#    N1 = np.delete(N,subtract,axis = 0)
#    N1 = np.delete(N1,subtract,axis = 1)
    H1 = H
    N1 = N
#    print(H1)
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    #eigvals_1 = eigvals_1[np.where(eigvals_1 != 0 )]
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_L_1 = eigvec_L_1.T
    eigvec_L_1 = eigvec_L_1[x]

    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

### drop states with imaginary part
   # eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
   # eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

### divide training samples into few parts
    each_split = round(len(H1)/split)
    eigvals       = []
    eigvals_len = np.zeros(split)
    Rn2_all  = []
    Rp2_all  = []
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
        
        Rn2_matrix2 = np.delete(Rn2_matrix,subtract,axis = 0)
        Rn2_matrix2 = np.delete(Rn2_matrix,subtract,axis = 1)
        Rp2_matrix2 = np.delete(Rp2_matrix,subtract,axis = 0)
        Rp2_matrix2 = np.delete(Rp2_matrix,subtract,axis = 1)



###     solve the general eigval problem
        eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

###     sort with eigval
       # eigvals_2 = eigvals_2[np.where(eigvals_2 != 0 )]
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
      #  print(eigvals_2)
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]

        for loop1 in range(len(eigvals_2)):
            norm = np.dot(np.dot(eigvec_R_2[loop1].T,N2),eigvec_R_2[loop1]) 
            Rn2 =np.dot( np.dot(eigvec_R_2[loop1].T, Rn2_matrix),eigvec_R_2[loop1] )/norm
            Rp2 =np.dot( np.dot(eigvec_R_2[loop1].T, Rp2_matrix),eigvec_R_2[loop1] )/norm

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
            ev_ultra_vec_R = eigvec_R_1[loop]
            ev_ultra_vec_L = eigvec_L_1[loop]
            break
    norm = np.dot(np.dot(ev_ultra_vec_R.T,N),ev_ultra_vec_R) 
    Rn2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix), ev_ultra_vec_R)/norm
    Rp2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix), ev_ultra_vec_R)/norm
    ev_ultra_1 =np.dot( np.dot(ev_ultra_vec_R.T, H), ev_ultra_vec_R)/norm
    #if ( abs(ev_ultra_1.real - ev_ultra.real) > 5 ):    

    print("norm="+str(norm))
    print("Rn2="+str(Rn2))
    print("ev_ultra_vec_R="+str(ev_ultra_vec_R))
    #print("ev_ultra_vec_L="+str(ev_ultra_vec_L))
    print("ev_ultra_1="+str(ev_ultra_1))
    print("ev_ultra.real="+str(ev_ultra))

    return ev_ultra.real,Rn2,Rp2, eigvals_1, vote


######################################################
######################################################
#### emulator7
######################################################
######################################################
def emulator7(subtract_count,LEC_target,tolerance,cc_gs,cc_Rn2_temp,cc_Rp2_temp):
    split = 100
    sample_each_slice = 30
    vote_need = round(0.75*split)

    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    subtract = [subtract_count]
#    H1 = np.delete(H,subtract,axis = 0)
#    H1 = np.delete(H1,subtract,axis = 1)
#    N1 = np.delete(N,subtract,axis = 0)
#    N1 = np.delete(N1,subtract,axis = 1)
    H1 = H
    N1 = N

### solve the general eigval problem
    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H,N,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvals_1_real_x = np.where(abs(eigvals_1.imag)<0.01)
    eigvals_1_real = eigvals_1[eigvals_1_real_x]

    eigvec_L_1 = eigvec_L_1.T
    eigvec_L_1 = eigvec_L_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

    Rn2_all = observable_batch_cal(eigvec_R_1,N1,Rn2_matrix)
    Rn2_all = np.array(Rn2_all)
    Rn2_all_real_x = np.where(abs(Rn2_all.imag)<0.0001)
    Rn2_all_real = Rn2_all[Rn2_all_real_x]


    Rp2_all = observable_batch_cal(eigvec_R_1,N1,Rp2_matrix)
    Rp2_all = np.array(Rp2_all)
    Rp2_all_real_x = np.where(abs(Rp2_all.imag)<0.0001)
    Rp2_all_real = Rp2_all[Rp2_all_real_x]


### drop states with imaginary part
#    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
#    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

### divide training samples into few parts
    serial_all = range(0,subspace_dimension)
    eigvals       = []
    eigvals_len = np.zeros(split)
    for loop in range(split):
        wf_all   = np.zeros((subspace_dimension,subspace_dimension)) 
        slice_1  = random.sample(serial_all,sample_each_slice)
        subtract = np.delete(range(subspace_dimension),slice_1)
        remain   = np.delete(serial_all,subtract)
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

    for loop in range(len(eigvals_1)):
        if vote[loop] >= vote_need:
            ev_ultra = eigvals_1[loop]
            if abs(ev_ultra.imag) > 0.01:
                xx = np.where((abs(eigvals_1.real) < abs(ev_ultra.real)*(1+0.02)) &\
                              (abs(eigvals_1.real) > abs(ev_ultra.real)*(1-0.02)) &\
                              (abs(eigvals_1.imag) < 0.01))
                xx = xx[0]
                if len(xx) == 0:
                    Rn2_ultra = Rn2_all[loop]
                    Rp2_ultra = Rp2_all[loop]
         # temp test
                    Rn2_ultra = Rn2_all_real[1]
                    Rp2_ultra = Rp2_all_real[1]

                    break    
                #print("xx"+str(xx))
                #print("eigvals_real"+str(eigvals_1[xx]))
                #print("Rn2_real"+str(Rn2_all[xx])) 
                Rn2_1 = Rn2_all[np.array(xx)] 
                Rp2_1 = Rp2_all[np.array(xx)]
                #print("Rn2_1"+str(Rn2_1[0])) 
                Rn2_ultra = Rn2_1[0]
                Rp2_ultra = Rp2_1[0]
 # temp test
                Rn2_ultra = Rn2_all_real[1]
                Rp2_ultra = Rp2_all_real[1]


                #Rn2_1 = Rn2_1.mean()
                #Rp2_1 = Rp2_1.mean()
                break 
            #ev_ultra_vec_R = eigvec_R_1[loop]
            #ev_ultra_vec_L = eigvec_L_1[loop]

            Rn2_ultra = Rn2_all[loop]
            Rp2_ultra = Rp2_all[loop]
            #ev_ultra_1 = eigvals_all[loop]
            #print("loop_energy="+str(loop))
            break
#    for loop in range(len(eigvals_1)):
#        if vote[loop] >= vote_need :
#            ev_ultra = eigvals_1[loop]
#            ev_ultra_vec_R = eigvec_R_1[loop]
#            ev_ultra_vec_L = eigvec_L_1[loop]
#            break
#
#    norm = np.dot(np.dot(ev_ultra_vec_R.T,N),ev_ultra_vec_R) 
#    Rn2 =np.dot( np.dot(ev_ultra_vec_R.T, Rn2_matrix), ev_ultra_vec_R)/norm
#    Rp2 =np.dot( np.dot(ev_ultra_vec_R.T, Rp2_matrix), ev_ultra_vec_R)/norm

    cc_Rs = np.sqrt(cc_Rn2_temp) - np.sqrt(cc_Rp2_temp)
    #print("cc_Rn2_temp="+str(cc_Rn2_temp))
    #print("cc_Rp2_temp="+str(cc_Rp2_temp))
    emulator_Rs = np.sqrt(Rn2_ultra) - np.sqrt(Rp2_ultra)
    print("cc_Rs="+str(cc_Rs))
    print("emulator_Rs="+str(emulator_Rs))
    #print("emulator_Rs_all="+str(Rs_ultra))
   # print("norm="+str(norm))
    if (abs(cc_Rs - emulator_Rs)/abs(cc_Rs)> 0.10):
        print("cc_gs="+str(cc_gs)) 
        print("ev_emulator="+str(ev_ultra)) 
        print("eigvals_1="+str(eigvals_1))
        print("cc_Rn2 = "+str(cc_Rn2_temp))
        print("Rn2_emulator="+str(Rn2_ultra))
        print("cc_Rp2 = "+str(cc_Rp2_temp))
        print("Rp2_emulator="+str(Rp2_ultra))
        #Rn2_all = Rn2_all[np.where(abs(Rn2_all.imag) < 0.01)]

        print("Rn2="+str(Rn2_all))

    return ev_ultra.real,Rn2_ultra,Rp2_ultra,ev_ultra,vote





######################################################
######################################################
#### validation
######################################################
######################################################
def validation2(tolerance):
    emulator_data=[]
    emulator_Rn2=[]
    emulator_Rp2=[]
    ccsdt_data=[]
    cc_Rn2=[]
    cc_Rp2=[]
    #file_path       = my_path + "LEC_read3.txt"
    #validation_path = my_path + "validation/Pb208/ccsdt_Pb208_34_points.txt" 
    file_path = "/home/slime/subspace_CC/test/Pb208/Pb208_spcc_2/Pb208_spcc/cc_input/cc_lhs_input_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10_percent_68_points/list_of_points_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10percent_68_points.txt"
    validation_path = "/home/slime/subspace_CC/test/Pb208/Pb208_spcc_2/Pb208_spcc/cc_input/cc_lhs_input_Delta_go_394_mass_208_hw_10_NO_ISOBREAK_10_percent_68_points/ccsd_energies.txt"
    validation_Rn2_path ="/home/slime/subspace_CC/test/Pb208/Pb208_spcc_2/Pb208_spcc/cc_output/cross_validation_data/cc_Rn2.txt"
    validation_Rp2_path ="/home/slime/subspace_CC/test/Pb208/Pb208_spcc_2/Pb208_spcc/cc_output/cross_validation_data/cc_Rp2.txt"
    #subtract=[2, 9, 15, 17, 21, 22, 24, 26, 31, 33, 36, 40, 41, 50, 51, 55, 59, 61, 66]
    subtract=[]

############################################
## read energy validation data 
############################################
    ccsdt_temp = read_validation_data(validation_path)
    ccsdt_temp = np.delete (ccsdt_temp,subtract)
    
    #print("read in ccsd calculation \n file_count ="+str(file_count))
    #print("ccsd ="+str(ccsdt_temp))
    
############################################
### read Rn2 Rp2 validation data
############################################

    cc_Rn2_temp = read_validation_data(validation_Rn2_path)
    cc_Rp2_temp = read_validation_data(validation_Rp2_path)
   
    cc_Rn2_temp = np.delete (cc_Rn2_temp,subtract)
    cc_Rp2_temp = np.delete (cc_Rp2_temp,subtract)
   # ccsdt_temp = np.delete (ccsdt_temp,subtract)

############################################
### emulator prediction
############################################
    LEC_batch = read_LEC_batch(file_path)

    LEC_batch = np.delete(LEC_batch,subtract,axis=0)
#    print("LEC_batch"+str(LEC_batch))


    for loop in range(len(LEC_batch)):
        print("###############loop"+str(loop)+"###############")
        #gs = emulator4(LEC_batch[loop])
        gs,Rn2,Rp2,eigvals,vote = emulator7(0,LEC_batch[loop],tolerance,ccsdt_temp[loop],cc_Rn2_temp[loop],cc_Rp2_temp[loop])
        #gs,Rn2,Rp2 = emulator(LEC_batch[loop])

        emulator_data.append(gs)
        emulator_Rn2.append(Rn2)
        emulator_Rp2.append(Rp2)
        ccsdt_data.append(ccsdt_temp[loop])
        cc_Rn2.append(cc_Rn2_temp[loop])
        cc_Rp2.append(cc_Rp2_temp[loop])
    return emulator_data,ccsdt_data,emulator_Rn2,cc_Rn2,emulator_Rp2,cc_Rp2

######################################################
######################################################
#### plot
######################################################
######################################################
def plot_1(emulator_data,ccsdt_data,emulator_Rn2,cc_Rn2,emulator_Rp2,cc_Rp2   , tolerance):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(10, 21))
    plt.subplots_adjust( hspace =0.3)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(421)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208_gs")

    x_list_1 =  ccsdt_data
    y_list_1 =  emulator_data

    uper_range   = -1000
    lower_range  = -2000
    gap          = 200
   # uper_range   = 6
   # lower_range  = 5
   # gap          = 0.2


    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))

    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(422)
    ax.set_title("Pb208")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    x_list_1 = ((np.array(emulator_data) - np.array(ccsdt_data))/abs(np.array(ccsdt_data)))

    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=100,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
    plt.xlim((-0.2,0.2))
    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)
######################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(423)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208_Rn")

    x_list_1 =  cc_Rn2
    y_list_1 =  emulator_Rn2

    uper_range   = 6
    lower_range  = 5
    gap          = 0.2


    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))

    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(424)
    ax.set_title("Pb208_Rn")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    x_list_1 = ((np.array(emulator_Rn2) - np.array(cc_Rn2))/abs(np.array(cc_Rn2)))

    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=100,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
    plt.xlim((-0.2,0.2))
    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)

######################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(425)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208_Rp")

    x_list_1 =  cc_Rp2
    y_list_1 =  emulator_Rp2

    uper_range   = 5.8
    lower_range  = 4.8
    gap          = 0.2


    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))

    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(426)
    ax.set_title("Pb208_Rp")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    x_list_1 = ((np.array(emulator_Rp2) - np.array(cc_Rp2))/abs(np.array(cc_Rp2)))

    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=100,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
    plt.xlim((-0.2,0.2))
    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(427)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("Pb208_Rs")

    x_list_1 =  cc_Rn2 - cc_Rp2
    y_list_1 =  emulator_Rn2 - emulator_Rp2
    #x_list_1  = emulator_Rn2 
    #y_list_1  = emulator_Rp2

    #x_list_1  = cc_Rn2 
    #y_list_1  = cc_Rp2


    uper_range   = 0.16
    lower_range  = 0.10
    gap          = 0.01

    #uper_range   = 5.8
    #lower_range  = 4.8
    #gap          = 0.2





    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    plt.xlabel("CCSDT [MeV]" ,fontsize=10)
    plt.ylabel("emulator [MeV]",fontsize=10)

    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))

    #plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    #plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)


#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(428)
    ax.set_title("Pb208_Rs")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    x_list_1 = ((np.array(emulator_Rn2- emulator_Rp2) - np.array( cc_Rn2 - cc_Rp2
))/abs(np.array( cc_Rn2 - cc_Rp2
)))

    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=200,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccsdt)/abs(ccsdt)",fontsize=10)
    plt.xlim((-0.2,0.2))
    plt.xticks(np.arange(-0.2,0.201,0.05),fontsize = 10)
######################################################


    #plot_path =  'Pb208_CV_emulator8_tolerance_%s_test_3_9sample_subtract.pdf' %(str(tolerance))
    plot_path =  'Pb208_CV_emulator9_tolerance_%s.pdf' %(str(tolerance))
    plt.savefig(plot_path)

#####################################################
#####################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 34
validation_count   = 68
LEC_num = 17
LEC = np.zeros(LEC_num)
subtract = []
remain   = []

N = np.zeros((subspace_dimension,subspace_dimension))
C = np.zeros((subspace_dimension,subspace_dimension))
H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))

my_path ="./"
database_dir = "/home/slime/subspace_CC/test/Pb208/Pb208_spcc_2/Pb208_spcc/cc_input/cc_lhs_input_Delta_go_394_mass_208_hw_12_NO_ISOBREAK_5_percent_34_points/"

H_matrix, C, N = read_emulator_matrices(database_dir)
file_path =database_dir+"Rn2_5percent34_nnlo394_delta_mass_208_N10E22_hw12.dat"
Rn2_matrix     = np.loadtxt(file_path)
file_path =database_dir+"Rp2_5percent34_nnlo394_delta_mass_208_N10E22_hw12.dat"
Rp2_matrix     = np.loadtxt(file_path)
print("Rn2="+str(Rn2_matrix))

file_path   = "ccm_in_DNNLO394"
LEC         = read_LEC(file_path)
#
tolerance   = 0.02
#gs_DNNLO394 = emulator4(0,LEC_target=LEC,tolerance = tolerance)
gs_DNNLO394 = emulator(LEC)
#
for loop in range(3):
    tolerance = 0.01 *(loop+1) 
    emulator_data,ccsdt_data,emulator_Rn2,cc_Rn2,emulator_Rp2,cc_Rp2 = validation2(tolerance=tolerance)
    #plot_1(emulator_data,ccsdt_data,tolerance=tolerance)
    plot_1(emulator_data,ccsdt_data,np.sqrt(emulator_Rn2),np.sqrt(cc_Rn2), np.power(emulator_Rp2,0.5),np.power(cc_Rp2,0.5),tolerance=tolerance)

#gs_DNNLO394,test = emulator(LEC,subtract)
##print(LEC)
print(gs_DNNLO394)


