import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
from scipy import interpolate
import seaborn as sns
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
    return eigvals_1[0].real
 
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
    subtract = range(0,round(len(H)/3.))
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
    subtract = range(round(len(H)/3.),round(len(H)/3.*2))
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

##### with subtract
    subtract = range(round(len(H)/3.*2),len(H))
    H4 = np.delete(H,subtract,axis = 0)
    H4 = np.delete(H4,subtract,axis = 1) 
    N4 = np.delete(N,subtract,axis = 0)
    N4 = np.delete(N4,subtract,axis = 1) 

### solve the general eigval problem
    eigvals_4,eigvec_L_4, eigvec_R_4 = spla.eig(H4,N4,left =True,right=True)

### sort with eigval
    x = np.argsort(eigvals_4)
    eigvals_4  = eigvals_4[x]
    eigvec_R_4 = eigvec_R_4.T
    eigvec_R_4 = eigvec_R_4[x]

### drop states with imaginary part
    eigvals_new_4   = eigvals_4[np.where(abs(eigvals_4.imag) < 0.01)] 
    eigvec_R_new_4 =  eigvec_R_4[np.where(abs(eigvals_4.imag)< 0.01)] 


    score1 = np.zeros(len(eigvals_1))  
    score2 = np.zeros(len(eigvals_1))  
    score3 = np.zeros(len(eigvals_1))  
    ev_ultra = 0
    for loop1 in range(len(eigvals_2)):
        for loop2 in range(len(eigvals_1)):
            if ( np.abs((eigvals_2[loop1].real - eigvals_1[loop2].real )/eigvals_2[loop1]) < 0.05 ):
                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                score1[loop2] = score1[loop2] + 1 

    for loop1 in range(len(eigvals_3)):
        for loop2 in range(len(eigvals_1)):
            if ( np.abs((eigvals_3[loop1].real - eigvals_1[loop2].real )/eigvals_3[loop1]) < 0.05 ):
                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                score2[loop2] = score2[loop2] + 1 

    for loop1 in range(len(eigvals_4)):
        for loop2 in range(len(eigvals_1)):
            if ( np.abs((eigvals_4[loop1].real - eigvals_1[loop2].real )/eigvals_4[loop1]) < 0.05 ):
                #print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
                score3[loop2] = score3[loop2] + 1 


    #ev_ultra_temp = eigvals_1[np.where((score1>0) and (score2>0))]
    #ev_ultra = ev_ultra_temp[0]
    for loop in range(len(eigvals_1)):
        if score1[loop]>0 and score2[loop]>0 and score3[loop]>0:
            ev_ultra = eigvals_1[loop]
            break

    #return eigvals_new , eigvals_R_new
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
def find_subtract(input_dir,expectation):
    subtract = []
    with open(input_dir,'r') as f_1:
        count = len(open(input_dir,'rU').readlines())
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            temp_1     = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count = round(float(temp_1[0]))
            ccd_1      = float(temp_1[1])
            if matter_type =="snm":
                error = error_sample
            else:
                error = -error_sample
            if(ccd_1> (expectation*(1-error) ) or ccd_1< (expectation*(1+error))):
                subtract.append(file_count-1)       
    return subtract





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

#print ("ev_all="+str(ev_all))


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
ccd_data      = [] 
emulator_data = [] 
density_data  = []
LEC = np.zeros(17)


my_path      = "./"
density_min  = 0.12
density_gap  = 0.02
matter_type  = "pnm"
particle_num = 66
error_sample = 0.30



for loop in range(5):
    density = density_min + loop*density_gap
    database_dir = my_path + "emulator/DNNLO450/%s_%d_%.2f_DNNLOgo_20percent_64points/" % (matter_type,particle_num,density)
    file_path    = my_path + "validation/%s_%.2f/validation_different_subspace.txt" % (matter_type,density)
    input_dir    = my_path + "emulator/DNNLO450/%s_%d_%.2f_DNNLOgo_20percent_64points/ccd.out" % (matter_type,particle_num,density)
    if (matter_type == "snm") :
        expectation = [-1809,-1902,-1940,-1919,-1838] 
    else:
        expectation = [726,857,1009,1181,1373] 
    subtract = find_subtract(input_dir,expectation[loop])
    print(subtract)
    print(64-len(subtract))
    with open(file_path,'r') as f_2:
        count = len(open(file_path,'rU').readlines())
        data = f_2.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            if ( re.search('ccd =', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                ccd_data.append(float(temp_1[0]))
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1-3])
                LEC[0:6]  = temp_1[0:6]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1-2])
                LEC[6:12] = temp_1[0:6]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1-1])
                LEC[12:17]= temp_1[0:5]
                #eigvalue, eigvec = emulator1(database_dir,LEC,subtract)
                #gs = eigvalue[0]
                gs = emulator0(database_dir,LEC,subtract)
                emulator_data.append(gs)
                density_data.append(density_min + loop*density_gap)


###print(sm_cal_new)
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


y_list_1 =  emulator_data
x_list_1 =  ccd_data

if (matter_type == "snm") :
    uper_range  = -6000
    lower_range =  500
else:
    uper_range  = 3000
    lower_range = -500

l1 = plt.scatter (x_list_1, y_list_1,color = 'darkblue' ,marker = 's',zorder=0.5)
l2 = plt.plot([lower_range ,uper_range], [lower_range, uper_range], ls="--",color = 'k', lw = 2, zorder = 1)

plt.xlim((lower_range,uper_range))
plt.ylim((lower_range,uper_range))
plt.xticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)
plt.yticks(np.arange(lower_range,uper_range+1,500),fontsize = 10)


#plt.legend(loc='upper left',fontsize = 15)
plt.xlabel(r"$\rm{CCD} \ [\rm{MeV}]$",fontsize=20)
plt.ylabel(r"$\rm{emulator} \ [\rm{MeV}]$",fontsize=20)

plot_path = 'test1.pdf'
plt.savefig(plot_path,bbox_inches='tight')
plt.close('all')


x_list_1 = ((np.array(emulator_data) - np.array(ccd_data))/np.array(ccd_data))
fig_2 = plt.figure('fig_2')

sns.set_palette("hls")
matplotlib.rc("figure", figsize=(6,4))
sns.distplot(x_list_1,bins=1200,kde_kws={"color":"seagreen", "lw":0 }, hist_kws={ "color": "lightblue"})

plt.ylabel("count")
plt.xlabel("error_0")
plt.xlim((-0.1,0.1))
plot_path = 'test2.pdf'
plt.savefig(plot_path)
fig_2.show()

file_path = "PNM.txt" 
with open(file_path,'w') as f_1:
    f_1.write('#density          E/Awith CCD          E/A with emulator'+'\n')
    for loop in range(len(density_data)):
        f_1.write('%.2f              %6.3f              %6.3f\n' % (density_data[loop], ccd_data[loop]/particle_num,emulator_data[loop]/particle_num))
 


