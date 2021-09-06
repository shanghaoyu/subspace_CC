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
from . import io_1

################################################
################################################
### Emulator tool
################################################
################################################
class emulator_tool:
    def __init__(self):
        self.path = "/home/slime/subspace_CC/test/emulator/DNNLO394/"
        self.density_count =  5
        self.density_min   =  0.12
        self.density_gap   =  0.02
        self.domain_dimension =  17
        self.subspace_dimension = 64
        self.subspace_matrices_NM_snm = []
        self.subspace_matrices_NM_pnm = []
        self.subspace_norm_matrix_NM_snm = []
        self.subspace_norm_matrix_NM_pnm = []
        self.subtract_snm = []
        self.subtract_pnm = []
        self.hf_matrices_snm  = []
        self.hf_matrices_pnm  = []
        self.hyperparameter   = []

    def load_emulator_matrix(self):
        
        for loop in range(self.density_count):
            matter_type = 'pnm'
            particle_num = 66
            density = self.density_min + loop * self.density_gap
            database_dir = self.path + '%s_%d_%.2f_DNNLOgo_christian_64points/' % (matter_type,particle_num,density)
            input_dir = self.path + "%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
            in_dir = database_dir+"N_matrix.txt"
            N = np.loadtxt(in_dir)
            self.subspace_norm_matrix_NM_pnm.append([N])
            in_dir = database_dir+"C_matrix.txt"
            C = np.loadtxt(in_dir)
            self.subspace_matrices_NM_pnm.append([C])
            for loop1 in range(self.domain_dimension):
                in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
                mtx = np.loadtxt(in_dir)
                self.subspace_matrices_NM_pnm.append([mtx])
        
            # load hf matrix
            database_dir = self.path + 'hf_emulator/%s_%d_%.2f_hf/' % (matter_type,particle_num,density)
            in_dir = database_dir+"LEC_hf_C_matrix"
            self.hf_matrices_pnm.append(np.loadtxt(in_dir))
            for loop1 in range(self.domain_dimension):
                in_dir = database_dir+"LEC_hf_"+str(loop1+1)+"_matrix"
                self.hf_matrices_pnm.append(np.loadtxt(in_dir))

        for loop in range(self.density_count):
            matter_type = 'snm'
            particle_num = 132
            density = self.density_min + loop * self.density_gap
            database_dir = self.path + '%s_%d_%.2f_DNNLOgo_christian_64points/' % (matter_type,particle_num,density)
            input_dir = self.path + "%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
        
            in_dir = database_dir+"N_matrix.txt"
            N = np.loadtxt(in_dir)
            self.subspace_norm_matrix_NM_snm.append([N])
            in_dir = database_dir+"C_matrix.txt"
            C = np.loadtxt(in_dir)
            self.subspace_matrices_NM_snm.append([C])
            for loop1 in range(self.domain_dimension):
                in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
                mtx = np.loadtxt(in_dir)
                self.subspace_matrices_NM_snm.append([mtx])

            # load hf matrix
            database_dir = self.path + 'hf_emulator/%s_%d_%.2f_hf/' % (matter_type,particle_num,density)
            in_dir = database_dir+"LEC_hf_C_matrix"
            self.hf_matrices_snm.append(np.loadtxt(in_dir))
            for loop1 in range(self.domain_dimension):
                in_dir = database_dir+"LEC_hf_"+str(loop1+1)+"_matrix"
                self.hf_matrices_snm.append(np.loadtxt(in_dir))



    def evaluate_NM_batch(self,domain_points):
        observable_batch = []
        for loop in range(len(domain_points)):
            observable_batch.append(evaluate_NM(domain_points[loop]))
        return observable_batch
    
    def evaluate_NM(self,domain_point):
        """
        Evaluate NM observable with emulator at a single domain point
    
        Returns:
            observables (array of floats): eigvals[0], obs_vals
        """
        emulator_switch= 5
        interpolation_choice = 'GP'
        hyperparameter  = self.hyperparameter         
 
        density_batch = np.array( [0.12,0.14,0.16,0.18,0.20])
        pnm_data      = np.zeros(self.density_count)
        snm_data      = np.zeros(self.density_count)
        pnm_hf_data   = np.zeros(self.density_count)
        snm_hf_data   = np.zeros(self.density_count)
        LEC_target    = np.zeros(self.domain_dimension)
        #LEC_target    = domain_point[1:18]
        LEC_target    = domain_point
        # test
        #LEC_target    =  read_LEC("/home/slime/history_matching/O28_prediction/history_matching/emulators/ccm_in_DNNLO394")
    
        ###########################################
        ##### pnm calculation for different density
        ###########################################
        H_matrix = np.zeros((self.domain_dimension,self.subspace_dimension,self.subspace_dimension))
        LEC_all_matrix = np.zeros(self.domain_dimension)
    
    
        for loop in range(self.density_count):
            for loop1 in range(self.domain_dimension):
                H_matrix[loop1,:,:] = self.subspace_matrices_NM_pnm[loop*18+ loop1+1][0]
            C  = self.subspace_matrices_NM_pnm[loop*18][0]
            N  = self.subspace_norm_matrix_NM_pnm[loop][0]
            pnm_data[loop],eigvec_temp,vote_temp = self.emulator(emulator_switch,'pnm',H_matrix, C, N, [],LEC_target,hyperparameter)
            pnm_data[loop]    = pnm_data[loop]/66
    
    
            C = self.hf_matrices_pnm[loop*18]
            LEC_all_matrix = self.hf_matrices_pnm[loop*18+1:loop*18+19]
            pnm_hf_data[loop] = self.hf_emulator(LEC_all_matrix,C,LEC_target)
    
    
        ###########################################
        ##### snm calculation for different density
        ###########################################
        for loop in range(self.density_count):
            for loop1 in range(self.domain_dimension):
    
                H_matrix[loop1,:,:] = self.subspace_matrices_NM_snm[loop*18+ loop1+1][0]
            C  = self.subspace_matrices_NM_snm[loop*18][0]
            N  = self.subspace_norm_matrix_NM_snm[loop][0]
            snm_data[loop],eigvec_temp,vote_temp = self.emulator(emulator_switch,'snm',H_matrix, C, N, [],LEC_target,hyperparameter)
            snm_data[loop] = snm_data[loop]/132
    
            C = self.hf_matrices_snm[loop*18]
            LEC_all_matrix = self.hf_matrices_snm[loop*18+1:loop*18+19]
            snm_hf_data[loop] = self.hf_emulator(LEC_all_matrix,C,LEC_target)
        #print(pnm_hf_data)
        #print(snm_hf_data)
    
        #print("pnm："+str(pnm_data))
        #print("snm: "+str(snm_data))
        ###########################################
        ##### calculate NM observables
        ###########################################
        saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = io_1.generate_NM_observable(pnm_data,snm_data,density_batch,"GP")
    
        #print("saturation_density="+str(saturation_density))
        #print("saturation_energy="+str(saturation_energy))
        #print("symmetry_energy="+str(symmetry_energy))
        #print("L="+str(L))
        #print("K="+str(K))
    
        #observables = np.array([saturation_density]+ [saturation_energy]+ [symmetry_energy]+[L]+[K])
        observables = np.zeros(26)
        observables[0] =  domain_point[0]
        observables[1] =  saturation_density
        observables[2] =  saturation_energy
        observables[3] =  symmetry_energy
        observables[4] =  L
        observables[5] =  K
        observables[6:11]  = pnm_data[0:5]
        observables[11:16] = snm_data[0:5]
        observables[16:21] = pnm_hf_data[0:5]
        observables[21:26] = snm_hf_data[0:5]
    
        #print("observables="+str(observables))
        return observables #, eigvec_temp, raw_data





######################################################
######################################################
### hf emulator!!!
######################################################
######################################################
    def hf_emulator(self,LEC_all_matrix,C,LEC_target):
        LEC_num = len(LEC_target)
        H = 0
        for loop1 in range(LEC_num):
            H = H + LEC_target[loop1] * LEC_all_matrix[loop1]
        H = H + C
        return H


######################################################
######################################################
### Emulator!!!
######################################################
######################################################
    def emulator(self,switch,matter_type,H_matrix,C,N,subtract,LEC_target,hyperparameter):
        if(switch == 1):
            ev_ultra,eigvals,vote = emulator1(H_matrix,C,N,subtract,LEC_target,0)
        elif(switch == 2):
            ev_ultra,eigvals,vote = emulator2(H_matrix,C,N,subtract,LEC_target,0.03)
        elif(switch == 3):
            ev_ultra,eigvals,vote = emulator4(H_matrix,C,N,subtract,LEC_target,0.04)
        elif(switch == 4):
            ev_ultra,eigvals,vote = emulator4(H_matrix,C,N,subtract,LEC_target,0.04)
        elif(switch == 5):
            if matter_type == "pnm":
                ev_ultra,eigvals,vote = emulator5(H_matrix,C,N,subtract,LEC_target,0.01)
            if matter_type == "snm":
                ev_ultra,eigvals,vote = emulator5(H_matrix,C,N,subtract,LEC_target,0.01)
    
    
        else:
            print("NM emulator choice error.")
    
        return ev_ultra, eigvals, vote


######################################################
######################################################
### Emulator1
######################################################
######################################################
def emulator1(H_matrix,C,N,subtract,LEC_target,tolerance):
    subspace_dimension = np.size(H_matrix,1)
    LEC_num            = np.size(H_matrix,0)
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
    H = H + C

    #print("H="+str(H))
#    eigvals,eigvec = spla.eig(N)
#    print ("N eigvals = "+str(sorted(eigvals)))


##### without subtract
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)


#    H[subtract] = 0
#    H[:,subtract] = 0
#    N[subtract] = 0
#    N[:,subtract] = 0
#    N[subtract,subtract] = 1

#    print("shape of H ="+str(H.shape))
#    print("rank of N ="+str(np.linalg.matrix_rank(N)))

#    np.savetxt('H.test',H,fmt='%.10f')
#    np.savetxt('N.test',N,fmt='%.10f')
#    H = np.loadtxt('H.test')
#    N = np.loadtxt('N.test')
### solve the general eigval problem
    eigvals,eigvec_L, eigvec_R = spla.eig(H1,N1,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals)
    eigvals  = eigvals[x]
    eigvec_R = eigvec_R.T
    eigvec_R = eigvec_R[x]

### drop states with imaginary part
#    eigvals_new   = eigvals[np.where(abs(eigvals.imag) < 0.01)]
#    eigvec_R_new =  eigvec_R[np.where(abs(eigvals.imag)< 0.01)]
    eigvals_new  = eigvals
    eigvec_R_new = eigvec_R


### drop states with 0 
#    eigvals_new  =  eigvals_new[np.where( eigvals_new != 0)]
#    eigvec_R_new =  eigvec_R_new[np.where(eigvals_new != 0)]


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
    vote = 0
    return eigvals_new[0] , eigvec_R_new[0], vote


######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator2(H_matrix,C,N,subtract,LEC_target,tolerance):

    split = 5
    #vote_need = 5
    subspace_dimension = np.size(H_matrix,1)
    LEC_num            = np.size(H_matrix,0)

    H = np.zeros((subspace_dimension,subspace_dimension))
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C

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
    vote = 0
    return ev_ultra.real, eigvals_1, vote


######################################################
######################################################
#### emulator3
######################################################
######################################################
def emulator3(H_matrix,C,N,subtract,LEC_target,tolerance):
    split = 4
    subspace_dimension = np.size(H_matrix,1)
    LEC_num            = np.size(H_matrix,0)
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    #subtract = [subtract_count]

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
    vote = 0
    return ev_ultra.real, eigvals_1, vote


######################################################
######################################################
#### emulator4
######################################################
######################################################
def emulator4(H_matrix,C,N,subtract,LEC_target,tolerance):
    split = 4
    subspace_dimension = np.size(H_matrix,1)
    LEC_num            = np.size(H_matrix,0)
    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    #subtract = [subtract_count]

###########################################################################
#   if needed subtract some of the trainning samples
###########################################################################
    H1 = np.delete(H,subtract,axis = 0)
    H1 = np.delete(H1,subtract,axis = 1)
    N1 = np.delete(N,subtract,axis = 0)
    N1 = np.delete(N1,subtract,axis = 1)

    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)
### sort with eigval
    eigvals_1 = eigvals_1[np.where(eigvals_1 != 0 )]
    x = np.argsort(eigvals_1)
    eigvals_1  = eigvals_1[x]
    eigvec_R_1 = eigvec_R_1.T
    eigvec_R_1 = eigvec_R_1[x]

#############################################################################
#    other way of subtracting 
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
#    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H1,N1,left =True,right=True)
#
#### sort with eigval
#    eigvals_1 = eigvals_1[np.where(eigvals_1 != 0 )]
#    x = np.argsort(eigvals_1)
#    eigvals_1  = eigvals_1[x]
#    eigvec_R_1 = eigvec_R_1.T
#    eigvec_R_1 = eigvec_R_1[x]
############################################################################
### drop states with imaginary part
############################################################################
   # eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)]
   # eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)]

############################################################################
### divide training samples into small batches
############################################################################
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

#############################################################################
#    other way of subtracting 
#############################################################################
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
        eigvals_2 = eigvals_2[np.where(eigvals_2 != 0 )]
        x = np.argsort(eigvals_2)
        eigvals_2  = eigvals_2[x]
        eigvec_R_2 = eigvec_R_2.T
        eigvec_R_2 = eigvec_R_2[x]
        eigvals_len[loop] = len(eigvals_2)
        eigvals.append(eigvals_2[:])

#############################################################################
#   voting for each state
#############################################################################
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
    vote = 0
    return ev_ultra.real, eigvals_1, vote


######################################################
######################################################
#### emulator5
######################################################
######################################################
def emulator5(H_matrix,C,N,subtract,LEC_target,tolerance):
    split = 100
    sample_each_slice = 30
    vote_need = round(0.7*split)
    #vote_need = round(1*split)
    subspace_dimension = np.size(H_matrix,1)
    LEC_num            = np.size(H_matrix,0)

    H = np.zeros((subspace_dimension,subspace_dimension))

    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C
    #subtract = [subtract_count]
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
    if vote.max() < vote_need:
        vote_need_new = vote.max()
    else:
        vote_need_new = vote_need
    for loop in range(len(eigvals_1)):
        if vote[loop] >= vote_need_new :
            ev_ultra = eigvals_1[loop]
            vote_ultra = vote[loop]
            break


    return ev_ultra.real, eigvals_1,vote_ultra

