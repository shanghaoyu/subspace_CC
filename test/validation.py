import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
from scipy import interpolate

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
def emulator(LEC_target,subtract):
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
     
#    H = np.delete(H,subtract_1,axis = 0)
#    H = np.delete(H,subtract_1,axis = 1) 
#    N = np.delete(N,subtract_1,axis = 0)
#    N = np.delete(N,subtract_1,axis = 1)


 
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

#    print(eigvals[0])
#    print((eigvec_R[0])) 
#    print(abs(eigvec_R[0])**2) 
#
#    print("wtf \n")
#  
#    print(eigvals_new[0])
#    print((eigvec_R_new[0])) 
#
#
#
    print(eigvals)
#    print(eigvals_new)
#    sum_1 = 0
#    for loop1 in range(np.size(eigvec_R[0],1)):
#        print("%d : %.5f %%" % (loop1,eigvec_R_new[0,0,loop1]**2*100))
#        sum_1 = sum_1 + eigvec_R_new[0,0,loop1]**2
#    print(eigvec_R_new[0])
#    print(eigvec_R_new.shape)
#    print(np.dot(eigvec_R_new[0],np.conjugate(eigvec_R_new[0].T)))

    with open("emulator.wf",'w') as f_1:
        #f_2.write('ccd = %.12f     emulator = %.12f   all =' % (ccd_cal, emulator_cal))
        f_1.write('################################\n')
        f_1.write('#### emulator wave function ####\n')
        f_1.write('################################\n')
        f_1.write('all eigvals: \n')
        for loop in range(len(eigvals)):
            loop2 = len(eigvals)-loop-1
            f_1.write("(%.4f + %.4fi)\n" % (float(eigvals[loop2].real/66), float(eigvals[loop2].imag/66)))
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


##### with subtract
#    H = np.delete(H,subtract,axis = 0)
#    H = np.delete(H,subtract,axis = 1) 
#    N = np.delete(N,subtract,axis = 0)
#    N = np.delete(N,subtract,axis = 1) 
#
#    #np.savetxt('H.test',H,fmt='%.9f')
#    #np.savetxt('N.test',N,fmt='%.9f')
#    #H = np.loadtxt('H.test')
#    #N = np.loadtxt('N.test')
#    eigvals,eigvec_L, eigvec_R = spla.eig(H,N,left =True,right=True)
#
#    loop2 = 0
#    for loop1 in range(np.size(H,1)):
#        ev = eigvals[loop1]
#        if ev.imag > 0.01:
#            continue
#    #    if ev.real < 0:
#    #        continue
#        loop2 = loop2+1
#
#    ev_all = np.zeros(loop2)
#    loop2 = 0
#    for loop1 in range(np.size(H,1)):
#        ev = eigvals[loop1]
#        if ev.imag >0.01 :
#            continue
#    #    if ev.real < 0:
#    #        continue
#        ev_all[loop2] = ev.real
#        loop2 = loop2+1
#
#    ev_sorted_2 = sorted(ev_all)
#    #print('eigvals='+str (ev_sorted))
#    #print ("ccd energy from emulator:"+str(ev_sorted[0]))
#  
#    ev_ultra = 0
#    for loop1 in range(len(ev_sorted_2)):
#        if ev_ultra != 0:
#            break
#        for loop2 in range(len(ev_sorted_1)):
#            if ( np.abs((ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1]) < 0.01 ):
#                print(np.abs(ev_sorted_2[loop1] - ev_sorted_1[loop2] )/ev_sorted_2[loop1])
#                ev_ultra = ev_sorted_1[loop2]
#                break
#
    #return eigvals_new , eigvals_R_new
    return eigvals_new , eigvec_R_new
 

def emulator_test(LEC_target,subtract):
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
    print("len of H ="+str(len(H)))
    print("rank of N ="+str(np.linalg.matrix_rank(N)))

    #generate symmetric matrix
    
#    for loop1 in range(len(H)-1):
#        for loop2 in range(loop1+1):
#            H[loop1 + 1,loop2] =  H[loop2, loop1+1]
#
#    for loop1 in range(len(N)-1):
#        for loop2 in range(loop1+1):
#            N[loop1 + 1,loop2] =  N[loop2, loop1+1]


    print(H)

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
#database_dir = '/home/slime/subspace_CC/test/emulator/DNNLO450/snm_132_0.16_DNNLOgo_20percent_64points/'
#database_dir = '/home/slime/subspace_CC/test/emulator/DNNLO450/pnm_66_0.20_DNNLOgo_20percent_64points/'
database_dir = '/home/slime/subspace_CC/test/emulator/DNNLO394/pnm_66_0.16_DNNLOgo_christian_64points/'
#database_dir = '/home/slime/subspace_CC/test/emulator/DNNLO394/snm_132_0.16_DNNLOgo_christian_64points/'

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
subtract = range(48)
# start validation 


#seed = 6
validation_count = 1
for loop1 in range(validation_count):
    file_path = "ccm_in_DNNLO394"
    LEC = read_LEC(file_path)
    #file_path = "2.txt"
    #LEC = read_LEC_2(file_path)
    #LEC_random = generate_random_LEC(LEC, LEC_range)
    LEC_random = LEC
    print ("LEC="+str(LEC_random))
    #LEC_random = LEC
    #ccd_cal = nuclear_matter(LEC_random)
    #ccd_cal = 0
    eigvalue, eigvec = emulator(LEC_random,subtract)
#    gs = eigvals[x[0]]
#    gs_vec = eigvec_R[x[0]]
#    file_path = "validation_different_subspace.txt"
#    with open(file_path,'w') as f_1:
#        f_1.write('ccd = %.12f     emulator = %.12f \n' % (ccd_cal, emulator_cal))
    file_path = "validation_detail_test_different_subspace.txt"
    with open(file_path,'w') as f_2:
        #f_2.write('ccd = %.12f     emulator = %.12f   all =' % (ccd_cal, emulator_cal))
        f_2.write(str(eigvalue))
        f_2.write('\n')
        f_2.write(str(eigvec))
#        for loop2 in range(len(eigvec)):
#            f_2.write('eigvalue = %.12f \n' % (eigvalue[loop2]) )
#            f_2.write('eigvec   =  ' + str(eigvec[loop2]) )


### plot
##file_path = "ccm_in_DNNLO450"
##LEC = read_LEC(file_path)
##
##LEC_new = np.zeros(LEC_num)
##
##LEC_new = LEC.copy() 
##sm_count   = 20
##sm_cal_new = np.zeros(sm_count)
##LEC_new_shift = np.zeros(sm_count)
##
##count = 0 
##which_LEC_1 = 10
##which_LEC_2 = 7
##for loop1 in np.arange(0,1,1./sm_count):
##    LEC_range = 0.6 
##    LEC_max = LEC * ( 1 + LEC_range)
##    LEC_min = LEC * ( 1 - LEC_range)
##    LEC_new[which_LEC_1] = LEC_min[which_LEC_1] + loop1 * (LEC_max[which_LEC_1] - LEC_min[which_LEC_1])
##    #LEC_new[which_LEC_2] = LEC_min[which_LEC_2] + loop1 * (LEC_max[which_LEC_2] - LEC_min[which_LEC_2])
##    LEC_new_shift[count] = LEC_new[which_LEC_1]
##    #sm_cal_new[count],temp__    = emulator(LEC_new)
##    sm_cal_new[count]    = emulator(LEC_new)
##    print(LEC_new)
##    print(sm_cal_new[count])
##    count  = count + 1 
##
###print(sm_cal_new)
##
##fig1 = plt.figure('fig1')
##
##matplotlib.rcParams['xtick.direction'] = 'in'
##matplotlib.rcParams['ytick.direction'] = 'in'
##ax1 = plt.subplot(111)
##plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
##ax1.spines['bottom'].set_linewidth(2)
##ax1.spines['top'].set_linewidth(2)
##ax1.spines['left'].set_linewidth(2)
##ax1.spines['right'].set_linewidth(2)
##
##y_list_1 =  sm_cal_new
###x_list_1 =  LEC_new_shift
##x_list_1 =  np.arange(0, 1.0, 0.05)
##
##print (x_list_1)
##print (y_list_1)
###l0 = plt.scatter (x_list_0,y_list_0,color = 'k', marker = 's',s = 200 ,zorder = 4, label=r'$\Delta$NNLO$_{\rm{go}}$(450)')
##l1 = plt.scatter (x_list_1, y_list_1,color = 'k' ,zorder=2,label = 'emulator')
###l2 = plt.plot([-10, 40], [-10, 40], ls="-",color = 'k', lw = 3, zorder = 3)
###plt.xlim((-10,40))
###plt.ylim((-10,40))
###plt.xticks(np.arange(-10,41,10),fontsize = 15)
###plt.yticks(np.arange(-10,41,10),fontsize = 15)
##
##
##plt.legend(loc='upper left',fontsize = 15)
##plt.xlabel(r"$\rm{LEC} \ [\rm{MeV}]$",fontsize=20)
##plt.ylabel(r"$\rm{E} \ [\rm{MeV}]$",fontsize=20)
##
##plot_path = 'emulator_1point_test.pdf'
##plt.savefig(plot_path,bbox_inches='tight')
##plt.close('all')




