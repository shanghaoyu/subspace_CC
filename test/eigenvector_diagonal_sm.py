import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
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

def read_sm_vec(vec_num,vec_dimension,sm_vec,database_dir):
    for loop1 in range(vec_num):
        file_path = database_dir + str(loop1+1)+"_sm.txt"
        with open(file_path,'r') as f_1:
           count = len(open(file_path,'rU').readlines())
           data = f_1.readlines()
           wtf = re.match('#', 'abc',flags=0)
           for loop2 in range(0,vec_dimension):
               temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop2])
               sm_vec[loop1][loop2] = float(temp_1[0])

######################################################
######################################################
### generate infile for solve_general_EV
######################################################
######################################################
def generate_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax,cal_type):
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
        f_1.write('%s\n' %(cal_type) )
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
    neutron_num  = 2  #test
    particle_num = 28
    density      = 0.16
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 1 #test

    generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax,'solve_general_EV_sm')
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
### sm calculation with different LECs
######################################################
######################################################
def sm_calculation(vec_input,in_dir,out_dir):
    neutron_num  = 2  #test
    particle_num = 28
    density      = 0.16
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 1 #test
    generate_ccm_in_file(in_dir,vec_input,neutron_num,'pnm',density,nmax,'sm')
    os.system('./'+nucl_matt_exe+' '+in_dir+' > '+out_dir)

    with open('sm_result.txt','r') as f_1:
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[0])
        sm_cal = float(temp_1[0])
    return sm_cal


######################################################
######################################################
### generate emulator_matrix
######################################################
######################################################
def generate_emulator_matrix(subspace_dimension):
    C_matrix = np.zeros((subspace_dimension,subspace_dimension))
    N_matrix = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((subspace_dimension,subspace_dimension))
    LEC_all_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))

    LEC     = np.zeros(LEC_num)
    call_solve_general_EV(LEC,"ccm_in_test","a.out")
    N_matrix = np.loadtxt("N_matrix_sm.txt")
    H_matrix = np.loadtxt("H_matrix_sm.txt")
    #out_dir = "./N_matrix_sm.txt"
    #np.savetxt(out_dir,N_matrix)
 
    C_matrix = H_matrix
    out_dir = "./C_matrix_sm.txt"
    np.savetxt(out_dir,C_matrix)

    for loop1 in range(LEC_num):
        LEC = np.zeros(LEC_num)
        LEC[loop1] = 1 
        call_solve_general_EV(LEC,"ccm_in_test","a.out")
        H_matrix = np.loadtxt("H_matrix_sm.txt")
        #K_matrix = np.loadtxt("K_matrix_sm.txt")
        LEC_all_matrix[loop1,:,:] = H_matrix + - C_matrix
        out_dir = "./emulator/LEC_"+str(loop1+1)+"_matrix_sm"
        np.savetxt(out_dir,LEC_all_matrix[loop1,:,:])



######################################################
######################################################
### solve_general_EV_sm!!!
######################################################
######################################################
def solve_general_EV_sm(LEC_target,database_dir):
    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
 #   C = np.zeros((subspace_dimension,subspace_dimension))
    H_matrix = np.zeros((LEC_num,subspace_dimension,subspace_dimension))
    N = np.loadtxt(database_dir+"N_matrix_sm.txt")
    H = np.loadtxt(database_dir+"H_matrix_sm.txt")

   #subtract = [4,30, 56 ]
    subtract = []
    H = np.delete(H,subtract,axis = 0)
    H = np.delete(H,subtract,axis = 1)  
    N = np.delete(N,subtract,axis = 0)
    N = np.delete(N,subtract,axis = 1)  

    eigvals,eigvec = spla.eig(N)
    print ("N eigvals = "+str(sorted(eigvals)))
    
    #np.set_printoptions(suppress=True)
    #np.set_printoptions(precision=6) 
   # np.savetxt('H.test',H,fmt='%.01f')
#    np.savetxt('H.test',H)

    # test 
#    N_new = np.zeros((subspace_dimension-2,subspace_dimension-2))
#    H_new = np.zeros((subspace_dimension-2,subspace_dimension-2))
#
#    for loop1 in range(subspace_dimension-2):
#        for loop2 in range(subspace_dimension-2):
#            loop3 = loop1
#            loop4 = loop2
#            if (loop3 >= 13):
#                loop3 = loop1+1
#            if (loop4 >= 13):
#                loop4 = loop2+1
#            if (loop3 >= 33):
#                loop3 = loop1+1
#            if (loop4 >= 33):
#                loop4 = loop2+1
#            N_new[loop1,loop2] = N[loop3,loop4]
#
#    for loop1 in range(subspace_dimension-2):
#        for loop2 in range(subspace_dimension-2):
#            loop3 = loop1
#            loop4 = loop2
#            if (loop3 >= 13):
#                loop3 = loop1+1
#            if (loop4 >= 13):
#                loop4 = loop2+1
#            if (loop3 >= 33):
#                loop3 = loop1+1
#            if (loop4 >= 33):
#                loop4 = loop2+1
#            H_new[loop1,loop2] = H[loop3,loop4]
##
##
#    np.savetxt("./N_new.txt",N)
#    np.savetxt("./H_new.txt",H)
#    print("H="+str(H))
#    print("rank of N ="+str(np.linalg.matrix_rank(N)))
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
    print('H',np.size(H,1)) 
    eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    print('eigvalsize,', eigvals.shape) 
    loop2 = 0
    for loop1 in range(np.size(H,1)):
        ev = eigvals[loop1] 
        if ev.imag != 0:
            continue
    #    if ev.real < 0:
    #        continue
        loop2 = loop2+1
    
    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(np.size(H,1)):
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
    print(eigvec_L[np.where(eigvals==ev_sorted[0])]) 
    
    print('eigvals_gs='+str (ev_sorted[0]))
    
    
    
    #D,V = np.linalg.eig(H_matrix)
    #print ("D="+str(D))
    #print(np.linalg.matrix_rank(N_matrix))
    #print(np.linalg.matrix_rank(H_matrix))
    
    
    #print(N_matrix)
    #print(H_matrix)   

def test_1():
    vec_num =64
    vec_dimension = 53
    sm_vec = np.zeros((vec_num,vec_dimension))
    LEC = read_LEC("ccm_in_DNNLO450")
    #LEC_14th = read_LEC_2("/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_test_sm_vs_ccd_nmax1_n_2/14_sm.txt") 
    #print("LEC_14th"+str(LEC_14th))
    sm_calculation(LEC,"ccm_in_test","a.out")
    H_1   = np.loadtxt("H_temp_real.txt")
    
    print ("H= "+str(H_1.shape))
    vec_num = 64
    read_sm_vec(vec_num,vec_dimension,sm_vec,"/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_test_sm_vs_ccd_nmax1_n_2/")

    #eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    eigvals, eigvec = spla.eig(H_1)

    loop2 = 0
    for loop1 in range(np.size(H_1,1)):
        ev = eigvals[loop1]
        if ev.imag != 0:
            continue
        loop2 = loop2+1
    
    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(np.size(H_1,1)):
        ev = eigvals[loop1] 
        if ev.imag != 0:
            continue
    #    if ev.real < 0:
    #        continue
        ev_all[loop2] = ev.real
        loop2 = loop2+1
    ev_sorted = sorted(ev_all)
#    print ("H eigvals = "+str(ev_sorted))
#    print("sm_vec="+str( sm_vec.shape))
    H_temp = np.dot(sm_vec,H_1)
    H = np.dot(H_temp,sm_vec.T)
    N = np.dot(sm_vec,sm_vec.T)
    print("H="+str( H.shape))
#    print("N="+str( N))
    np.savetxt("H_matrix_sm_test.txt",H,fmt='%.15f')
    np.savetxt("N_matrix_sm_test.txt",N,fmt='%.15f')

    H = np.loadtxt("./H_matrix_sm_test.txt")
    N = np.loadtxt("./N_matrix_sm_test.txt")
    

    eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    print('eigvalsize,', eigvals.shape) 
    loop2 = 0
    for loop1 in range(np.size(H,1)):
        ev = eigvals[loop1] 
        if ev.imag != 0:
            continue
    #    if ev.real < 0:
    #        continue
        loop2 = loop2+1
    
    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(np.size(H,1)):
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
 
 

    H_2   = np.loadtxt("H_matrix_sm.txt")
    N_2   = np.loadtxt("N_matrix_sm.txt")

    np.savetxt("H-H_2.txt",H-H_2)
    np.savetxt("N-N_2.txt",N-N_2)


######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 64
LEC_num = 17
nucl_matt_exe = './prog_ccm.exe'


#database_dir= "/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_test_sm_vs_ccd_nmax1_n_2/"
database_dir= "./"
file_path = "ccm_in_DNNLO450"
LEC = read_LEC(file_path)
#solve_general_EV_sm(LEC,database_dir)

generate_emulator_matrix(subspace_dimension)

#test_1()







###LEC_new = np.zeros(LEC_num)
####sm_cal_new = np.zeros(LEC_num)
###
###LEC_new = LEC.copy()
###sm_count   = 10
###sm_cal_new = np.zeros(sm_count)
###LEC_new_shift = np.zeros(sm_count)
###
###count = 0
###which_LEC = 10
###for loop1 in np.arange(0,1,1./sm_count):
###    LEC_range = 10
###    LEC_max = LEC * ( 1 + LEC_range)
###    LEC_min = LEC * ( 1 - LEC_range)
###    LEC_new[which_LEC] = LEC_min[which_LEC] + loop1 * (LEC_max[which_LEC] - LEC_min[which_LEC])
####    print(LEC_new[which_LEC])
###    LEC_new_shift[count] = LEC_new[which_LEC]
###    sm_cal_new[count]    = sm_calculation(LEC_new,"ccm_in_test","a.out")
###    count  = count + 1
###
###print(sm_cal_new)
###
###fig1 = plt.figure('fig1')
###
###matplotlib.rcParams['xtick.direction'] = 'in'
###matplotlib.rcParams['ytick.direction'] = 'in'
###ax1 = plt.subplot(111)
###plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
###ax1.spines['bottom'].set_linewidth(2)
###ax1.spines['top'].set_linewidth(2)
###ax1.spines['left'].set_linewidth(2)
###ax1.spines['right'].set_linewidth(2)
###
###
#### sm calculation
###y_list_1 =  sm_cal_new
###x_list_1 =  LEC_new_shift
###
###
####l0 = plt.scatter (x_list_0,y_list_0,color = 'k', marker = 's',s = 200 ,zorder = 4, label=r'$\Delta$NNLO$_{\rm{go}}$(450)')
###l1 = plt.scatter (x_list_1, y_list_1,color = 'cornflowerblue', edgecolor = 'k', marker = 'o',s = 120 ,zorder=2,label = 'sm_cal')
####l2 = plt.plot([-10, 40], [-10, 40], ls="-",color = 'k', lw = 3, zorder = 3)
###
####plt.xlim((-10,40))
####plt.ylim((-10,40))
####plt.xticks(np.arange(-10,41,10),fontsize = 15)
####plt.yticks(np.arange(-10,41,10),fontsize = 15)
###
###
###plt.legend(loc='upper left',fontsize = 15)
###plt.xlabel(r"$\rm{CCSD} \ [\rm{MeV}]$",fontsize=20)
###plt.ylabel(r"$\rm{SP-CC} \ [\rm{MeV}]$",fontsize=20)
###
###plot_path = 'sm_test.pdf'
###plt.savefig(plot_path,bbox_inches='tight')
###plt.close('all')
###
###

