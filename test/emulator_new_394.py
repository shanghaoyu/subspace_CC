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
import pandas as pd

#quadratic_curve
def f_2(x, A, B, C):
    return A*x*x + B*x + C
 
#cubic_curve
def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D


######################################################
######################################################
### GP tool
######################################################
######################################################
class GP_test:

    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.sigma  = 100
        self.length = 0.25
        self.optimize = optimize
        self.gaussian_noise = 1

    def fit_data(self, x, y, gaussian_noise):
        # store train data
        self.train_x = np.asarray(x)
        self.train_y = np.asarray(y)
        self.gaussian_noise = gaussian_noise

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.l, self.sigma = params[0], params[1]
            Kyy = self.kernel(self.train_x, self.train_x) + 1e-8 * np.eye(len(self.train_x))
            return 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1    ] + 0.5 * len(self.train_x) * np.log(2 * np.pi)

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.length, self.sigma],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.length, self.sigma = res.x[0], res.x[1]

        self.is_fit = True
    def predict(self, x):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        x = np.asarray(x)
        # gaussian_noise**2 here is the variance of the gaussian like of noise in y ( y = f(x) + noise)  (noise =     N (0, gaussian_noise**2))
        Kff = self.kernel(self.train_x, self.train_x) + self.gaussian_noise**2 * np.eye(len(self.train_x))  # (N,     N)
        Kyy = self.kernel(x, x)  # (k, k)
        Kfy = self.kernel(self.train_x, x)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_x)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma ** 2 * np.exp(-0.5 / self.length ** 2 * dist_matrix)



######################################################
######################################################
### generate random LECs set
######################################################
######################################################
def generate_random_LEC(LEC,LEC_range):
    LEC_max = LEC * ( 1 + LEC_range)
    LEC_min = LEC * ( 1 - LEC_range)
    #np.random.seed(seed)
    LEC_random = np.zeros(LEC_num)
    for loop1 in range (LEC_num):
        LEC_random[loop1] = LEC_min[loop1] + np.random.rand(1) * (LEC_max[loop1] - LEC_min[loop1])
    return LEC_random


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

def read_LEC_1(file_path):
    LEC = np.zeros(LEC_num)
    with open(file_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data = f_1.readline()
        print(data)
        wtf = re.match('#', 'abc',flags=0)
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data)
        print("temp_1 "+str(temp_1))
        for loop2 in range(17):
            LEC[loop2] = float(temp_1[loop2])
    return LEC
def read_LEC_2(file_path):
    LEC = np.zeros(LEC_num)
    with open(file_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            if ( re.search('cD,cE', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                LEC[0] = float(temp_1[0])
                LEC[1] = float(temp_1[1])
            if ( re.search('LEC=', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                LEC[2] = float(temp_1[0])
                LEC[3] = float(temp_1[1])
                LEC[4] = float(temp_1[2])
                LEC[5] = float(temp_1[3])
            if ( re.search('c1s0, c3s1', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                LEC[6] = float(temp_1[4])
                LEC[7] = float(temp_1[5])
                LEC[8] = float(temp_1[6])
                LEC[9] = float(temp_1[7])
            if ( re.search('cnlo_pw', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                LEC[10] = float(temp_1[2])
                LEC[11] = float(temp_1[3])
                LEC[12] = float(temp_1[4])
                LEC[13] = float(temp_1[5])
                LEC[14] = float(temp_1[6])
                LEC[15] = float(temp_1[7])
                LEC[16] = float(temp_1[8])
    return LEC

def read_LEC_3(file_path):
    LEC = np.zeros(LEC_num)
    with open(file_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data = f_1.readline()
        print(data)
        wtf = re.match('#', 'abc',flags=0)
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data)
        print("temp_1 "+str(temp_1))
        for loop2 in range(17):
            LEC[loop2] = float(temp_1[loop2])
    return LEC


######################################################
######################################################
### generate nuclear matter infile
######################################################
######################################################
def output_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax):
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
        f_1.write('PBC'+'\n')
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
### read ccm_nuclear_matter output
######################################################
######################################################
def read_nucl_matt_out(file_path):  # converge: flag = 1    converge: flag =0
    with open(file_path,'r') as f_1:
        converge_flag = int (1)
        count = len(open(file_path,'rU').readlines())
        #if ( count > 1500 ):
        #    converge_flag =int (0)
        data =  f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        ccd = 0
        for loop1 in range(0,count):
            if ( re.search('CCD energy', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                ccd = float(temp_1[0])
        return ccd   #,converge_flag
        #print ('No "E/A" found in the file:'+file_path)
        #return float('nan')



######################################################
######################################################
### call CCM_nuclear_matter
######################################################
######################################################
def nuclear_matter(vec_input):
    neutron_num  = 14
    particle_num = 28
    density      = 0.16
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2
    #snm_dens    = np.zeros(5)
    #snm_energy_per_nucleon = np.zeros(5)
    #snm_dens_new = np.zeros(interpolation_count)
    #snm_energy_per_nucleon_new = np.zeros(interpolation_count)

    nucl_matt_in_dir   = './ccm_in_pnm_%.2f' % (density)
    nucl_matt_out_dir  = './pnm_rho_%.2f.out' % (density)

    output_ccm_in_file(nucl_matt_in_dir,vec_input,neutron_num,'pnm',density,nmax)
    os.system('./'+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir) 
    ccd = read_nucl_matt_out(nucl_matt_out_dir)
    print ("ccd energy from real CC calculation: "+str(ccd))
    return ccd

######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def find_notconverge(database_dir,converge_flag):
    for loop1 in range(subspace_dimension):
        file_path = database_dir+ '/'+ str(loop1+1)+'.txt'
        with open(file_path,'r') as f_1:
            count = len(open(file_path,'rU').readlines())
            data = f_1.readlines()
            wtf = re.match('#', 'abc',flags=0)
            for loop2 in range(0,count):
                if ( re.search('converge_flag', data[loop2],flags=0) != wtf):
                    temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop2])
                    converge_flag[loop1] = float(temp_1[0])
 
######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator0(database_dir,LEC_target,subtract):
    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    for loop1 in range(LEC_num):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        H_matrix[loop1,:,:] = np.loadtxt(in_dir) 
    #H = LECs[0]*H_matrix + K_matrix
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C 

#   print("H="+str(H))
#    eigvals,eigvec = spla.eig(N)
#    print ("N eigvals = "+str(sorted(eigvals)))


##### without subtract 
    H[subtract] = 0
    H[:,subtract] = 0
    N[subtract] = 0
    N[:,subtract] = 0
    N[subtract,subtract] = 1
     
#    H = np.delete(H,subtract_1,axis = 0)
#    H = np.delete(H,subtract_1,axis = 1) 
#    N = np.delete(N,subtract_1,axis = 0)
#    N = np.delete(N,subtract_1,axis = 1)


 
    #print("shape of H ="+str(H.shape))
    #print("rank of N ="+str(np.linalg.matrix_rank(N)))

#    np.savetxt('H.test',H,fmt='%.10f')
#    np.savetxt('N.test',N,fmt='%.10f')
#    H = np.loadtxt('H.test')
#    N = np.loadtxt('N.test')

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
    #return eigvals_new , eigvals_R_new
    eigvals_1 = eigvals_1[np.where(eigvals_1.real!=0)]
    eigvals_new_1 = eigvals_new_1[np.where(eigvals_new_1.real!=0)]
    #return eigvals_1[0].real
    return eigvals_new_1[0].real
 
######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator1(database_dir,LEC_target,subtract):
    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    for loop1 in range(LEC_num):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        H_matrix[loop1,:,:] = np.loadtxt(in_dir) 
    #H = LECs[0]*H_matrix + K_matrix
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

##### with subtract
    subtract = range(0,round(len(H)/2.))
    H2 = np.delete(H,subtract,axis = 0)
    H2 = np.delete(H2,subtract,axis = 1) 
    N2 = np.delete(N,subtract,axis = 0)
    N2 = np.delete(N2,subtract,axis = 1) 

### solve the general eigval problem
    eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_2)
    eigvals_2  = eigvals_2[x]
    eigvec_R_2 = eigvec_R_2.T
    eigvec_R_2 = eigvec_R_2[x]

### drop states with imaginary part
    eigvals_new_2   = eigvals_2[np.where(abs(eigvals_2.imag) < 0.01)] 
    eigvec_R_new_2 =  eigvec_R_2[np.where(abs(eigvals_2.imag)< 0.01)] 


##### with subtract
    subtract = range(round(len(H)/2.),len(H))
    H3 = np.delete(H,subtract,axis = 0)
    H3 = np.delete(H3,subtract,axis = 1) 
    N3 = np.delete(N,subtract,axis = 0)
    N3 = np.delete(N3,subtract,axis = 1) 

### solve the general eigval problem
    eigvals_3,eigvec_L_3, eigvec_R_3 = spla.eig(H3,N3,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_3)
    eigvals_3  = eigvals_3[x]
    eigvec_R_3 = eigvec_R_3.T
    eigvec_R_3 = eigvec_R_3[x]

### drop states with imaginary part
    eigvals_new_3   = eigvals_3[np.where(abs(eigvals_3.imag) < 0.01)] 
    eigvec_R_new_3 =  eigvec_R_3[np.where(abs(eigvals_3.imag)< 0.01)] 

    score1 = np.zeros(len(eigvals_1))  
    score2 = np.zeros(len(eigvals_1))  
    ev_ultra = 0
    for loop1 in range(len(eigvals_2)):
        if ev_ultra != 0:
            break
        for loop2 in range(len(eigvals_1)):
            if ( np.abs((eigvals_2[loop1].real - eigvals_1[loop2].real )/eigvals_2[loop1]) < 0.05 ):
                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                score1[loop2] = score1[loop2] + 1 
                #ev_ultra = eigvals_2[loop1]

    for loop1 in range(len(eigvals_3)):
        if ev_ultra != 0:
            break
        for loop2 in range(len(eigvals_1)):
            if ( np.abs((eigvals_3[loop1].real - eigvals_1[loop2].real )/eigvals_3[loop1]) < 0.05 ):
                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                score2[loop2] = score2[loop2] + 1 
                #ev_ultra = eigvals_2[loop1]

    #ev_ultra_temp = eigvals_1[np.where((score1>0) and (score2>0))]
    #ev_ultra = ev_ultra_temp[0]
    for loop in range(len(eigvals_1)):
        if score1[loop]>0 and score2[loop]>0:
            ev_ultra = eigvals_1[loop]
            break

    #return eigvals_new , eigvals_R_new
    return ev_ultra.real

 
######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator2(database_dir,LEC_target,subtract):
    split = 4

    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    for loop1 in range(LEC_num):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        H_matrix[loop1,:,:] = np.loadtxt(in_dir) 
    #H = LECs[0]*H_matrix + K_matrix
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
            subtract = range(loop*each_split,len(H))
        else:
            subtract = range(loop*each_split,(loop+1)*each_split)
        H2 = np.delete(H,subtract,axis = 0)
        H2 = np.delete(H2,subtract,axis = 1) 
        N2 = np.delete(N,subtract,axis = 0)
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

#    print("wtf3")
#    print(eigvals_len)
#    print(eigvals[0])
#    print(eigvals[0][0].real)
###     drop states with imaginary part
#        eigvals_new_2  =  eigvals_2 [np.where(abs(eigvals_2.imag)< 0.01)] 
#        eigvec_R_new_2 =  eigvec_R_2[np.where(abs(eigvals_2.imag)< 0.01)] 
       
    

###### with subtract
#    subtract = range(0,round(len(H)/4.))
#    H2 = np.delete(H,subtract,axis = 0)
#    H2 = np.delete(H2,subtract,axis = 1) 
#    N2 = np.delete(N,subtract,axis = 0)
#    N2 = np.delete(N2,subtract,axis = 1) 
#
#### solve the general eigval problem
#    eigvals_2,eigvec_L_2, eigvec_R_2 = spla.eig(H2,N2,left =True,right=True)
#
#### sort with eigval
#    x = np.argsort(eigvals_2)
#    eigvals_2  = eigvals_2[x]
#    eigvec_R_2 = eigvec_R_2.T
#    eigvec_R_2 = eigvec_R_2[x]
#
#### drop states with imaginary part
#    eigvals_new_2   = eigvals_2[np.where(abs(eigvals_2.imag) < 0.01)] 
#    eigvec_R_new_2 =  eigvec_R_2[np.where(abs(eigvals_2.imag)< 0.01)] 
#
#
###### with subtract
#    subtract = range(round(len(H)/4.),round(len(H)/4.*2))
#    H3 = np.delete(H,subtract,axis = 0)
#    H3 = np.delete(H3,subtract,axis = 1) 
#    N3 = np.delete(N,subtract,axis = 0)
#    N3 = np.delete(N3,subtract,axis = 1) 
#
#### solve the general eigval problem
#    eigvals_3,eigvec_L_3, eigvec_R_3 = spla.eig(H3,N3,left =True,right=True)
#
#### sort with eigval
#    x = np.argsort(eigvals_3)
#    eigvals_3  = eigvals_3[x]
#    eigvec_R_3 = eigvec_R_3.T
#    eigvec_R_3 = eigvec_R_3[x]
#
#### drop states with imaginary part
#    eigvals_new_3   = eigvals_3[np.where(abs(eigvals_3.imag) < 0.01)] 
#    eigvec_R_new_3 =  eigvec_R_3[np.where(abs(eigvals_3.imag)< 0.01)] 
#
###### with subtract
#    subtract = range(round(len(H)/4.*2),round(len(H)/4.*3))
#    H4 = np.delete(H,subtract,axis = 0)
#    H4 = np.delete(H4,subtract,axis = 1) 
#    N4 = np.delete(N,subtract,axis = 0)
#    N4 = np.delete(N4,subtract,axis = 1) 
#
#### solve the general eigval problem
#    eigvals_4,eigvec_L_4, eigvec_R_4 = spla.eig(H4,N4,left =True,right=True)
#
#### sort with eigval
#    x = np.argsort(eigvals_4)
#    eigvals_4  = eigvals_4[x]
#    eigvec_R_4 = eigvec_R_4.T
#    eigvec_R_4 = eigvec_R_4[x]
#
#### drop states with imaginary part
#    eigvals_new_4   = eigvals_4[np.where(abs(eigvals_4.imag) < 0.01)] 
#    eigvec_R_new_4 =  eigvec_R_4[np.where(abs(eigvals_4.imag)< 0.01)] 
#
###### with subtract
#    subtract = range(round(len(H)/4.*3),len(H))
#    H5 = np.delete(H,subtract,axis = 0)
#    H5 = np.delete(H5,subtract,axis = 1) 
#    N5 = np.delete(N,subtract,axis = 0)
#    N5 = np.delete(N5,subtract,axis = 1) 
#
#### solve the general eigval problem
#    eigvals_5,eigvec_L_5, eigvec_R_5 = spla.eig(H5,N5,left =True,right=True)
#
#### sort with eigval
#    x = np.argsort(eigvals_5)
#    eigvals_5  = eigvals_5[x]
#    eigvec_R_5 = eigvec_R_5.T
#    eigvec_R_5 = eigvec_R_5[x]
#
#### drop states with imaginary part
#    eigvals_new_5   = eigvals_5[np.where(abs(eigvals_5.imag) < 0.01)] 
#    eigvec_R_new_5 =  eigvec_R_5[np.where(abs(eigvals_5.imag)< 0.01)] 


    score  = np.zeros((len(eigvals_1),split))
#    score1 = np.zeros(len(eigvals_1))  
#    score2 = np.zeros(len(eigvals_1))  
#    score3 = np.zeros(len(eigvals_1))  
#    score4 = np.zeros(len(eigvals_1))  
    ev_ultra = 0

    for loop in range(split):
        for loop1 in range(int(eigvals_len[loop])):
            for loop2 in range(len(eigvals_1)):
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < 0.05 ):
                    #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                    score[loop2,loop] = score[loop2,loop] + 1 
#    for loop1 in range(len(eigvals_3)):
#        for loop2 in range(len(eigvals_1)):
#            if ( np.abs((eigvals_3[loop1].real - eigvals_1[loop2].real )/eigvals_3[loop1]) < 0.05 ):
#                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
#                score2[loop2] = score2[loop2] + 1 
#
#    for loop1 in range(len(eigvals_4)):
#        for loop2 in range(len(eigvals_1)):
#            if ( np.abs((eigvals_4[loop1].real - eigvals_1[loop2].real )/eigvals_4[loop1]) < 0.05 ):
#                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
#                score3[loop2] = score3[loop2] + 1 
#
#    for loop1 in range(len(eigvals_5)):
#        for loop2 in range(len(eigvals_1)):
#            if ( np.abs((eigvals_5[loop1].real - eigvals_1[loop2].real )/eigvals_5[loop1]) < 0.05 ):
#                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
#                score4[loop2] = score4[loop2] + 1 
#
#
#
#    #ev_ultra_temp = eigvals_1[np.where((score1>0) and (score2>0))]
#    #ev_ultra = ev_ultra_temp[0]
    # vote for the lowest ture state
    for loop in range(len(eigvals_1)):
        flag = 1
        for loop1 in range(split):
            if score[loop,loop1] <= 0:
                flag = flag * 0
        if flag == 1 :
            ev_ultra = eigvals_1[loop]
            break

### drop states with imaginary part
    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)] 
    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)] 
    #return eigvals_new , eigvals_R_new
    eigvals_1 = eigvals_1[np.where(eigvals_1.real!=0)]
    eigvals_new_1 = eigvals_new_1[np.where(eigvals_new_1.real!=0)]
    #return eigvals_1[0].real

    return ev_ultra.real
    #return eigvals_new_1[0].real



def emulator3(database_dir,LEC_target,subtract):
    split = 4

    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    for loop1 in range(LEC_num):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        H_matrix[loop1,:,:] = np.loadtxt(in_dir) 
    #H = LECs[0]*H_matrix + K_matrix
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
        subtract = np.delete(range(64),remain)
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
                if ( np.abs((eigvals[loop][loop1].real - eigvals_1[loop2].real )/eigvals[loop][loop1]) < 0.05 ):
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
 
#def emulator3(database_dir,LEC_target,subtract):
#    H = np.zeros((subspace_dimension,subspace_dimension))
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
#    for loop1 in range(LEC_num):
#        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
#    H = H + C 
#
#    for loop1 in range():
#        for loop2 in range():
#            H[loop1,loop2] = H[]
#
##   print("H="+str(H))
##    eigvals,eigvec = spla.eig(N)
##    print ("N eigvals = "+str(sorted(eigvals)))
#
#
###### without subtract 
#    H[subtract] = 0
#    H[:,subtract] = 0
#    N[subtract] = 0
#    N[:,subtract] = 0
#    N[subtract,subtract] = 1
#     
##    H = np.delete(H,subtract_1,axis = 0)
##    H = np.delete(H,subtract_1,axis = 1) 
##    N = np.delete(N,subtract_1,axis = 0)
##    N = np.delete(N,subtract_1,axis = 1)
#
#
# 
#    #print("shape of H ="+str(H.shape))
#    #print("rank of N ="+str(np.linalg.matrix_rank(N)))
#
##    np.savetxt('H.test',H,fmt='%.10f')
##    np.savetxt('N.test',N,fmt='%.10f')
##    H = np.loadtxt('H.test')
##    N = np.loadtxt('N.test')
#
#### solve the general eigval problem
#    eigvals_1,eigvec_L_1, eigvec_R_1 = spla.eig(H,N,left =True,right=True)
#
#### sort with eigval
#    x = np.argsort(eigvals_1)
#    eigvals_1  = eigvals_1[x]
#    eigvec_R_1 = eigvec_R_1.T
#    eigvec_R_1 = eigvec_R_1[x]
#
#### drop states with imaginary part
#    eigvals_new_1   = eigvals_1[np.where(abs(eigvals_1.imag) < 0.01)] 
#    eigvec_R_new_1 =  eigvec_R_1[np.where(abs(eigvals_1.imag)< 0.01)] 
#    #return eigvals_new , eigvals_R_new
#    eigvals_1 = eigvals_1[np.where(eigvals_1.real!=0)]
#    return eigvals_1[0].real

 
######################################################
######################################################
#### subtract
######################################################
######################################################
def find_subtract(input_dir,expectation,remain_sample_num):
    subtract = []
    remain   = []
    with open(input_dir,'r') as f_1:
        count = len(open(input_dir,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        file_count  = np.zeros(count)
        ccd_1       = np.zeros(count)
        error       = np.zeros(count)
        for loop1 in range(0,count):
            temp_1       = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count[loop1]= round(float(temp_1[0]))-1
            ccd_1[loop1] = float(temp_1[1])
            error[loop1] = abs(ccd_1[loop1] - expectation)
        xx  = np.argsort(error)            
        ccd_1 = ccd_1[xx]    
        file_count = file_count[xx]
        error= error[xx]
    remain   = file_count[0:remain_sample_num].astype(int)
    subtract = np.delete(range(count),remain)
            #if matter_type =="snm":
            #    error = error_sample
            #else:
            #    error = -error_sample
            #if(ccd_1> (expectation*(1-error) ) or ccd_1< (expectation*(1+error))):
            #    subtract.append(file_count-1)       
    return subtract,remain

# 
def validation(matter_type,particle_num):
    emulator_data = [] 
    density_data  = []
    ccd_data      = [] 
    for loop in range(5):
    #for loop in [1]:
        density = round(density_min + loop*density_gap,2)
        database_dir = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/" % (matter_type,particle_num,density)
        #print(database_dir)
        file_path    = my_path + "LEC_read3.txt"
        input_dir    = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
        print(input_dir)
        validation_dir=my_path + "validation/DNNLO394_validation/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
        if (matter_type == "snm") :
            expectation = [-1831,-1921,-1955,-1932,-1852]
        else:
            expectation = [763,896,1044,1206,1378]
        subtract,remain = find_subtract(input_dir,expectation[loop],remain_sample_num)
        print("remain sample:"+str(remain))
        print("remain_sample_number:"+str(len(remain)))
    
        with open(validation_dir,'r') as f_1:
            count = len(open(file_path,'rU').readlines())
            data = f_1.readlines()
            wtf = re.match('#', 'abc',flags=0)
            file_count = np.zeros(count-1)
            ccd_1      = np.zeros(count-1)
            for loop1 in range(0,count-1):
                temp_1     = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                file_count[loop1] = round(float(temp_1[0]))
                ccd_1[loop1]      = float(temp_1[1])
            ccd_1 = ccd_1[np.argsort(file_count)]
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
                ccd_data.append(ccd_1[loop1-1])
                #eigvalue, eigvec = emulator1(database_dir,LEC,subtract)
    #            #gs = eigvalue[0]
                gs = emulator3(database_dir,LEC,subtract)
                emulator_data.append(gs)
                density_data.append(density)
    return emulator_data,ccd_data,density_data
#print(density_data)
#print(ccd_data)
#print(emulator_data)

###print(sm_cal_new)
def plot_(matter_type,emulator_data,ccd_data,density_data):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,6))
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(111)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    
    if (matter_type == "snm") :
        y_list_1 =  emulator_data
        x_list_1 =  ccd_data
    else :    
        y_list_1 =  emulator_data
        x_list_1 =  ccd_data

    if (matter_type == "snm") :
        uper_range =  500
        lower_range  = -4000
    else:
        uper_range  = 3000
        lower_range = -500
    
    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    
    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))
    
    if (matter_type == "pnm") :
        plt.xticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)
        plt.yticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)
    else:
        plt.xticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)
        plt.yticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)
       
    
    #plt.legend(loc='upper left',fontsize = 15)
    plt.xlabel(r"$\rm{CCD} \ [\rm{MeV}]$",fontsize=20)
    plt.ylabel(r"$\rm{emulator} \ [\rm{MeV}]$",fontsize=20)
    
    plot_path = 'DNNLO394_%s_1.pdf' % (matter_type)
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')
    #
    #
    
    x_list_1 = ((np.array(emulator_data) - np.array(ccd_data))/abs(np.array(ccd_data)))
    fig_2 = plt.figure('fig_2')
    
    sns.set_palette("hls")
    matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=200,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})
    
    plt.ylabel("count")
    plt.xlabel("error: (emulator - ccd)/abs(ccd)")
    #plt.xlim((-0.1,0.1))
    plot_path = 'DNNLO394_%s_1_hist.pdf' % (matter_type)
    plt.savefig(plot_path)

def generate_observable(pnm_data,snm_data,density_sequence,switch):
    density_accuracy = 0.0001
    if (switch == "GP"):
        dens_count = len(density_sequence)
        #######################################################
        #######################################################
        ####  use GP to find the saturation point
        #######################################################
        #######################################################
        #t1 = time.time()
        train_x = np.arange(0.12,0.12+dens_count*0.02,0.02)
        train_x = train_x.reshape(-1,1)
        train_y_1 = snm_data
        test_x  = np.arange(0.12,0.20,density_accuracy).reshape(-1,1)
        
        gpr = GP_test()
        gaussian_noise = 0.02
        
        gpr.fit_data(train_x, train_y_1, gaussian_noise)
        
        snm, snm_cov = gpr.predict(test_x)
        
        iX=np.argmin(snm)
        test_y_1 = snm.ravel()
        confidence_1 = 1.96 * np.sqrt(np.diag(snm_cov))
        
        density_range = test_x[np.where((snm[:]<(snm[iX]+confidence_1[iX]))&(snm[:]>(snm[iX]-confidence_1[iX])))]
        
        
        #print("saturation density: %.3f +/- %.3f" % (test_x[iX], 0.5*(np.max(density_range)-np.min(density_range))))
        #print("saturation energy:  %.3f +/- %.3f" % (snm[iX] , confidence_1[iX]))
        
        
        train_y_2 = pnm_data
        gpr = GP_test()
        gpr.fit_data(train_x, train_y_2, gaussian_noise)
        
        pnm, pnm_cov = gpr.predict(test_x)
        
        test_y_2 = pnm.ravel()
        confidence_2 = 1.96 * np.sqrt(np.diag(pnm_cov))
        
        #print("pnm energy:  %.3f +/- %.3f" % ( pnm[iX], confidence_2[iX]))
        #print("symmetry energy:  %.3f +/- %.3f" % (pnm[iX]-snm[iX],(confidence_1[iX]+confidence_2[iX])))
        
        #t2 = time.time()
        #print("time for GP : "+ str(t2-t1))
        saturation_density = test_x[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]
    elif (switch =="fit_curve_quadratic"):
        A2,B2,C2 = optimize.curve_fit(f_2,density_sequence,snm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        snm = A2*x2*x2 + B2*x2 + C2
        iX=np.argmin(snm)

        A2,B2,C2 = optimize.curve_fit(f_2,density_sequence,pnm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        pnm = A2*x2*x2 + B2*x2 + C2

        saturation_density = x2[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]

    elif (switch =="fit_curve_cubic"):
        A3,B3,C3,D3 = optimize.curve_fit(f_3,density_sequence,snm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        snm = A3*x2*x2*x2 + B3*x2*x2 + C3*x2 +D3
        iX=np.argmin(snm)

        A3,B3,C3,D3 = optimize.curve_fit(f_3,density_sequence,pnm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        pnm = A3*x2*x2*x2 + B3*x2*x2 + C3*x2 +D3

        saturation_density = x2[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]

    elif (switch =="interpolate"):
        interpol_count = 1000
        spl_ccd_snm    = interpolate.UnivariateSpline(density_sequence,snm_data)
        spl_ccd_pnm    = interpolate.UnivariateSpline(density_sequence,pnm_data)
        spldens        = np.linspace(density_sequence[0],density_sequence[len(density_sequence)-1],num=interpol_count)
        snm = spl_ccd_snm(spldens)
        pnm = spl_ccd_pnm(spldens)
        iX=np.argmin(snm)
        saturation_density = spldens[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]
    else:
        print("error_1")
    return saturation_density, saturation_energy, symmetry_energy


#####################################################
#####################################################
#####################################################
def plot_2(observable,emulator_data_1,ccd_data_1,emulator_data_2,ccd_data_2,emulator_data_3,ccd_data_3):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(321)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax.set_title("saturation density")
   
    x_list_1 =  ccd_data_1
    y_list_1 =  emulator_data_1
    
    uper_range   = 0.24
    lower_range  = 0.08
    gap          = 0.04

    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    
    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))
    
    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
       
    
    #plt.legend(loc='upper left',fontsize = 15)
    plt.xlabel(r"$\rm{CCD} \ [\rm{MeV}]$",fontsize=10)
    plt.ylabel(r"$\rm{emulator} \ [\rm{MeV}]$",fontsize=10)
  
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(322)
    ax.set_title("saturation density")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
 
    x_list_1 = ((np.array(emulator_data_1) - np.array(ccd_data_1))/abs(np.array(ccd_data_1)))
    
    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=20,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})
    
    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccd)/abs(ccd)",fontsize=10)
    plt.xlim((-0.1,0.1))
    plt.xticks(np.arange(-0.1,0.11,0.05),fontsize = 10)
 

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(323)
    ax.set_title("saturation energy")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
   
    x_list_1 =  ccd_data_2
    y_list_1 =  emulator_data_2
    
    uper_range   =  -10
    lower_range  =  -24
    gap          =  4

    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    
    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))
    
    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
       
    
    #plt.legend(loc='upper left',fontsize = 15)
    plt.xlabel(r"$\rm{CCD} \ [\rm{MeV}]$",fontsize=10)
    plt.ylabel(r"$\rm{emulator} \ [\rm{MeV}]$",fontsize=10)
  

  
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(324)
    ax.set_title("saturation energy")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
 
    x_list_1 = ((np.array(emulator_data_2) - np.array(ccd_data_2))/abs(np.array(ccd_data_2)))
    
    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=20,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})
    
    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccd)/abs(ccd)",fontsize=10)
    plt.xlim((-0.1,0.1))
    plt.xticks(np.arange(-0.1,0.11,0.05),fontsize = 10)
 

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(325)
    ax.set_title("symmetry energy")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
   
    x_list_1 =  ccd_data_3
    y_list_1 =  emulator_data_3
    
    uper_range   =  35
    lower_range  =  10
    gap          =  5
 
    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
    l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
    
    plt.xlim((lower_range,uper_range))
    plt.ylim((lower_range,uper_range))
    
    plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
       
    
    #plt.legend(loc='upper left',fontsize = 15)
    plt.xlabel(r"$\rm{CCD} \ [\rm{MeV}]$",fontsize=10)
    plt.ylabel(r"$\rm{emulator} \ [\rm{MeV}]$",fontsize=10)
  
  
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(326)
    ax.set_title("symmetry energy")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
 
    x_list_1 = ((np.array(emulator_data_3) - np.array(ccd_data_3))/abs(np.array(ccd_data_3)))
    
    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=20,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})
    
    plt.ylabel("count" ,fontsize=10)
    plt.xlabel("relative error\n(emulator-ccd)/abs(ccd)",fontsize=10)
    plt.xlim((-0.1,0.1))
    plt.xticks(np.arange(-0.1,0.11,0.05),fontsize = 10)

    plot_path = 'DNNLO394_NM_observable_test_GP.pdf' 
    plt.savefig(plot_path)

def plot_3(saturation_density,saturation_energy,symmetry_energy):
    fig1 = plt.figure('fig1')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = -27.7
    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = saturation_density
    y_list = saturation_energy
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(saturation_energy),2))
    for loop1 in range(0,len(saturation_energy)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])


    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=20, linewidth=1, marker="x",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = 9)
    l2 = plt.scatter (0.16196, -14.812,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{saturation \ density} \ [\rm{fm}^{-3}]$",fontsize=10)
    plt.ylabel(r"$\rm{saturation \ energy} \ [\rm{MeV}]$",fontsize=10)


    plot_path = 'Pb208_34_sample_test.pdf' 
    plt.savefig(plot_path)
    plt.close('all')


   
#    x_list_1 = 
#    y_list_1 = 
#    
#    uper_range_x   = 0.22
#    lower_range_x  = 0.10
#    gap_x          = 0.02
#
#    uper_range_y   = -10
#    lower_range_y  = -20
#    gap_y          = 2
#
#    l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5, label = r"$\rm{HM \ (34\ samples)}$")
#    l2 = plt.scatter (0.16196, -14.812,color = 'red' ,marker = 'o',zorder=1,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
#    #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
#    #l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)
#    
#    plt.xlim((lower_range_x,uper_range_x))
#    plt.ylim((lower_range_y,uper_range_y))
#    
#    plt.xticks(np.arange(lower_range_x,uper_range_x+0.0001,gap_x),fontsize = 10)
#    plt.yticks(np.arange(lower_range_y,uper_range_y+0.0001,gap_y),fontsize = 10)
#       

    fig2 = plt.figure('fig2')
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(111)
    #ax.set_title("symmetry energy")
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    x_list_1 = symmetry_energy 
    
    #plt.vlines(30.866,0,0.3,colors ="k",ls= "--")
    sns.set_palette("hls")
    #matplotlib.rc("figure", figsize=(6,4))
    sns.distplot(x_list_1,bins=8,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})
    plt.vlines(30.866,0,0.18,colors ="red",ls= "--",label=r"$\rm{DNNLO}_{\rm{GO}}(394)$")
 
    plt.legend(loc='upper right',fontsize = 9)
    plt.ylabel("count" ,fontsize=10)
    plt.xlabel(r"$\rm{symmetry \ energy} \ [\rm{MeV}]$",fontsize=10)
    #plt.xlim((-0.1,0.1))
    plt.ylim((0,0.18))
    plt.yticks(np.arange(0,0.181,0.02),[0,1,2,3,4,5,6,7,8,9],fontsize = 10)

    plot_path = 'Pb208_34_sample.pdf' 
    plt.savefig(plot_path)




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
#database_dir = '/home/slime/subspace_CC/test/emulator/DNNLOgo450_20percent_64points_/'
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/'
#database_dir = '/home/slime/subspace_CC/test/emulator/'
#database_dir = '/home/slime/subspace_CC/test/emulator/snm_132_0.12_DNNLOgo_20percent_64points/'
#database_dir = '/home/slime/subspace_CC/test/emulator/pnm_66_0.20_DNNLOgo_20percent_64points/'

# pick out not converge CCD results
#converge_flag = np.zeros(subspace_dimension)
#find_notconverge('./',converge_flag)
#subtract = converge_flag.nonzero()
#print("converge_flag"+str(converge_flag))
#print(converge_flag.nonzero())
#subtract = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#subtract = [2,3,5,33,41,55,59]
#subtract = range(30,45)
#subtract = [0,2,3,5,9,10,11,16,17,18,22,23,26,27,28,29,30,32,33,34,36,37,39,40,42,43,44,45,47,48,51,52,54,55,61,62]
#subtract = [4,6,9,12,16,18,19,20,21,22,23,24,25,26,28,30,34,35,37,38,39,40,42,43,45,46,47,48,49,51,52,54,57,62]
#subtract = []
# start validation 
subtract = []
remain   = []
LEC = np.zeros(17)


my_path      = "./"
density_min  = 0.12
density_gap  = 0.02
density_count= 5
matter_type  = "pnm"
particle_num = 66
remain_sample_num = 64 



emulator_data,ccd_data,density_data = validation("pnm",66)
#plot_("pnm",emulator_data,ccd_data,density_data) 

validation_count=int(len(emulator_data)/density_count)
emulator_data_snm = np.zeros((validation_count,density_count))
emulator_data_pnm = np.zeros((validation_count,density_count))
ccd_data_snm      = np.zeros((validation_count,density_count))
ccd_data_pnm      = np.zeros((validation_count,density_count))
density_data_snm  = np.zeros((validation_count,density_count))
density_data_pnm  = np.zeros((validation_count,density_count))

for loop1 in range(validation_count):
    for loop2 in range(density_count):
        emulator_data_pnm[loop1,loop2] = emulator_data[loop2*validation_count+loop1] 
        ccd_data_pnm[loop1,loop2]      = ccd_data[loop2*validation_count+loop1] 
        density_data_pnm[loop1,loop2]  = density_data[loop2*validation_count+loop1]

emulator_data,ccd_data,density_data = validation("snm",132)
#plot_("snm",emulator_data,ccd_data,density_data) 

for loop1 in range(validation_count):
    for loop2 in range(density_count):
        emulator_data_snm[loop1,loop2] = emulator_data[loop2*validation_count+loop1] 
        ccd_data_snm[loop1,loop2]      = ccd_data[loop2*validation_count+loop1] 
        density_data_snm[loop1,loop2]  = density_data[loop2*validation_count+loop1]

print("test:"+str(emulator_data_snm[0][0]))
print("test:"+str(emulator_data_snm[1][0]))
print("test:"+str(emulator_data_snm[2][0]))
print("test:"+str(emulator_data_snm[3][0]))

saturation_density_emulator = np.zeros(validation_count)
saturation_energy_emulator  = np.zeros(validation_count)
symmetry_energy_emulator    = np.zeros(validation_count)

saturation_density_ccd = np.zeros(validation_count)
saturation_energy_ccd  = np.zeros(validation_count)
symmetry_energy_ccd    = np.zeros(validation_count)

for loop1 in range(validation_count):
    saturation_density_emulator[loop1], saturation_energy_emulator[loop1], symmetry_energy_emulator[loop1] = generate_observable(emulator_data_pnm[loop1,:]/66,emulator_data_snm[loop1,:]/132,density_data_pnm[loop1,:],"GP")#"fit_curve_quadratic")

    saturation_density_ccd[loop1], saturation_energy_ccd[loop1], symmetry_energy_ccd[loop1] = generate_observable(ccd_data_pnm[loop1,:]/66,ccd_data_snm[loop1,:]/132,density_data_pnm[loop1,:],"interpolate")


plot_3(saturation_density_emulator,saturation_energy_emulator,symmetry_energy_emulator)

#plot_2("saturation_density",saturation_density_emulator,saturation_density_ccd, saturation_energy_emulator/132,saturation_energy_ccd/132, symmetry_energy_emulator/132,symmetry_energy_ccd/132)
print("sample number"+str(validation_count))
print("saturation_density")
print(saturation_density_emulator)
#print(saturation_density_ccd)


print("saturation_energy")
print(saturation_energy_emulator)
#print(saturation_energy_ccd/132)

print("symmetry_energy")
print(symmetry_energy_emulator)
#print(symmetry_energy_ccd/132)


print(emulator_data_pnm)
print(ccd_data_pnm)
print(density_data_pnm)
#
DNNLO394_pnm =np.array([763,896,1044,1206,1378])
DNNLO394_snm =np.array([-1831,-1921,-1955,-1932,-1852])
DNNLO394_dens=np.array([0.12,0.14,0.16,0.18,0.20])
a,b,c = generate_observable(DNNLO394_pnm/66,DNNLO394_snm/132,DNNLO394_dens,"interpolate")
print ("DNNLO394_saturation_density",a)
print ("DNNLO394_saturation_energy",b)
print ("DNNLO394_symmetry_energy",c)



 
