import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
from scipy import interpolate
from ..io import inoutput

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
    ccd = inoutput.read_nucl_matt_out(nucl_matt_out_dir)
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
def emulator(database_dir,LEC_target,subtract):
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

#    print("rank of N ="+str(np.linalg.matrix_rank(N)))
    #print("N= "+str(N))

##### without subtract 
    subtract_1 = []
    H_new = np.delete(H,subtract_1,axis = 0)
    H_new = np.delete(H_new,subtract_1,axis = 1) 
    N_new = np.delete(N,subtract_1,axis = 0)
    N_new = np.delete(N_new,subtract_1,axis = 1) 

    #np.savetxt('H.test',H,fmt='%.10f')
    #np.savetxt('N.test',N,fmt='%.10f')
    #H = np.loadtxt('H.test')
    #N = np.loadtxt('N.test')
    eigvals,eigvec_L, eigvec_0 = spla.eig(H_new,N_new,left =True,right=True)
    loop2 = 0
    for loop1 in range(np.size(H_new,1)):
        ev = eigvals[loop1]
        if ev.imag > 0.01:
            continue
    #    if ev.real < 0:
    #        continue
        loop2 = loop2+1

    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(np.size(H_new,1)):
        ev = eigvals[loop1]
        if ev.imag >0.01 :
            continue
    #    if ev.real < 0:
    #        continue
        ev_all[loop2] = ev.real
        loop2 = loop2+1

    ev_sorted_1 = sorted(ev_all)
 
##### with subtract
    H = np.delete(H,subtract,axis = 0)
    H = np.delete(H,subtract,axis = 1) 
    N = np.delete(N,subtract,axis = 0)
    N = np.delete(N,subtract,axis = 1) 
    np.savetxt('N_test',)
    #np.savetxt('H.test',H,fmt='%.12f')
    #np.savetxt('N.test',N,fmt='%.12f')
    #H = np.loadtxt('H.test')
    #N = np.loadtxt('N.test')
    eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    loop2 = 0
    for loop1 in range(np.size(H,1)):
        ev = eigvals[loop1]
        if ev.imag > 0.01:
            continue
    #    if ev.real < 0:
    #        continue
        loop2 = loop2+1

    ev_all = np.zeros(loop2)
    loop2 = 0
    for loop1 in range(np.size(H,1)):
        ev = eigvals[loop1]
        if ev.imag > 0.01 :
            continue
    #    if ev.real < 0:
    #        continue
        ev_all[loop2] = ev.real
        loop2 = loop2+1

    ev_sorted_2 = sorted(ev_all)
    #print('eigvals='+str (ev_sorted))
    #print ("ccd energy from emulator:"+str(ev_sorted[0]))
  
    ev_ultra = 0
    for loop1 in range(len(ev_sorted_2)):
        if ev_ultra != 0:
            break
        for loop2 in range(len(ev_sorted_1)):
            if ( np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1] < 0.01 ):
                ev_ultra = ev_sorted_2[loop2]
                break
    #print(ev_ultra)
    #print(ev_sorted_1)
    #print(ev_sorted_2)

    return ev_ultra, ev_sorted_1, ev_sorted_2







######################################################
######################################################
#### MAIN
######################################################
######################################################
subspace_dimension = 64
LEC_num = 17
#LEC_range = 0.2
#LEC = np.ones(LEC_num)
#nucl_matt_exe = './prog_ccm.exe'
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/DNNLOgo450_20percent_64points_/'
##database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/'
#
#
##print ("ev_all="+str(ev_all))
#
#
## pick out not converge CCD results
#converge_flag = np.zeros(subspace_dimension)
#find_notconverge('./',converge_flag)
#subtract = converge_flag.nonzero()
#print("converge_flag"+str(converge_flag))
#print(converge_flag.nonzero())
#
## start validation 
#
##seed = 6
#validation_count = 10
#for loop1 in range(validation_count):
#    file_path = "ccm_in_DNNLO450"
#    LEC = inoutput.read_LEC_1(file_path)
#    LEC_random = generate_random_LEC(LEC, LEC_range)
#    print ("LEC="+str(LEC_random))
#    #LEC_random = LEC
#    ccd_cal = nuclear_matter(LEC_random)
#    #ccd_cal = 0
#    emulator_cal, ev_all_1, ev_all_2 = emulator(database_dir,LEC_random,subtract)
#    file_path = "validation_different_subspace.txt"
#    with open(file_path,'a') as f_1:
#        f_1.write('ccd = %.12f     emulator = %.12f \n' % (ccd_cal, emulator_cal))
#    file_path = "validation_detail_test_different_subspace.txt"
#    with open(file_path,'a') as f_2:
#        f_2.write('ccd = %.12f     emulator = %.12f   all =' % (ccd_cal, emulator_cal))
#        f_2.write(str(ev_all_1))
#        f_2.write(str(ev_all_2))
#        f_2.write('\n')


