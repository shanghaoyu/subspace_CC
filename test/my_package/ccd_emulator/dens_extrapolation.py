import os
import numpy as np
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

######################################################
######################################################
### generate nuclear matter infile
######################################################
######################################################
def output_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax,option):
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
        f_1.write('%-20s \n' % (option))
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
def nuclear_matter(vec_input,dens):
    neutron_num  = 14
    particle_num = 28
    density      = dens
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2
    #snm_dens    = np.zeros(5)
    #snm_energy_per_nucleon = np.zeros(5)
    #snm_dens_new = np.zeros(interpolation_count)
    #snm_energy_per_nucleon_new = np.zeros(interpolation_count)

    nucl_matt_in_dir   = './ccm_in_pnm_%.2f' % (density)
    nucl_matt_out_dir  = './pnm_rho_%.2f.out' % (density)
    option = 'PBC'
    output_ccm_in_file(nucl_matt_in_dir,vec_input,neutron_num,'pnm',density,nmax,option)
    os.system('./'+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir) 
    ccd = read_nucl_matt_out(nucl_matt_out_dir)
    print ("ccd energy from real CC calculation: "+str(ccd))
    return ccd


######################################################
######################################################
### Emulator!!!
######################################################
######################################################
def emulator(LEC_target,dens):
    neutron_num  = 14
    particle_num = 28
    density      = dens
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2
    nucl_matt_in_dir   = './ccm_in_pnm_%.2f' % (dens)
    nucl_matt_out_dir  = './pnm_rho_%.2f.out' % (dens)
    option = 'solve_general_EV' 
    output_ccm_in_file(nucl_matt_in_dir,LEC_target,neutron_num,'pnm',density,nmax,option)
    os.system('./'+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir) 
 
    H = np.zeros((subspace_dimension,subspace_dimension))
    N = np.zeros((subspace_dimension,subspace_dimension))
    K = np.zeros((subspace_dimension,subspace_dimension))

    in_dir = "./H_matrix.txt"
    H = np.loadtxt(in_dir)
    in_dir = "./N_matrix.txt"
    N = np.loadtxt(in_dir)
    in_dir = "./K_matrix.txt"
    K = np.loadtxt(in_dir)
    H = H + K 
 #   print("H="+str(H))
 #   print("rank of N ="+str(np.linalg.matrix_rank(N)))
        
#    eigvals,eigvec_L, eigvec_0 = spla.eig(H,N,left =True,right=True)
    eigvals = H
#    loop2 = 0
#    for loop1 in range(subspace_dimension):
#        ev = eigvals[loop1]
#        if ev.imag > 0.01:
#            continue
#    #    if ev.real < 0:
#    #        continue
#        loop2 = loop2+1
#
#    ev_all = np.zeros(loop2)
#    loop2 = 0
#    for loop1 in range(subspace_dimension):
#        ev = eigvals[loop1]
#        if ev.imag >0.01 :
#            continue
#    #    if ev.real < 0:
#    #        continue
#        ev_all[loop2] = ev.real
#        loop2 = loop2+1
#
#    ev_sorted = sorted(ev_all)
    #print('eigvals='+str (ev_sorted))
    #print('eigvec_L='+str (eigvec_L))
    #print('eigvec_0='+str (eigvec_0))

    #print('eigvals_gs='+str (ev_sorted[1]))
#    print ("ccd energy from emulator:"+str(ev_sorted[0]))
    return eigvals,eigvals







######################################################
######################################################
#### MAIN density extrapolation (validation)
######################################################
######################################################
subspace_dimension = 1 
LEC_num = 17
LEC_range = 0.2
LEC = np.ones(LEC_num)
nucl_matt_exe = './prog_ccm.exe'
database_dir  = '/work/Eigenvector_continuation/CCM_kspace_deltafull/test/backup/DNNLOgo450_dens_5points' 


#print ("ev_all="+str(ev_all))

# start validation 

dens_min = 0.14
dens_max = 0.30
dens_gap = 0.02
dens_count = int((dens_max - dens_min) / dens_gap + 2)
print (dens_count)
dens        = np.zeros(dens_count)
ccd_cal      = np.zeros(dens_count) 
emulator_cal = np.zeros(dens_count) 

for loop1 in range(dens_count):
    dens[loop1] = dens_min + ( dens_gap * loop1)
    print("densty = "+str(dens))
    file_path = "ccm_in_DNNLO450"
    LEC = read_LEC(file_path)
    #ccd_cal[loop1] = nuclear_matter(LEC,dens[loop1])
    ccd_cal[loop1] = 0
    emulator_cal[loop1], ev_all = emulator(LEC,dens[loop1])
    file_path = "density_extrapolation.txt"
    with open(file_path,'a') as f_1:
        f_1.write('dens=%.4f   ccd = %.12f     emulator = %.12f \n' % (dens[loop1],ccd_cal[loop1], emulator_cal[loop1]))
    file_path = "density_extrapolation_detail.txt"
    with open(file_path,'a') as f_2:
        f_2.write('dens=%.4f   ccd = %.12f     emulator = %.12f   all =' % (dens[loop1], ccd_cal[loop1], emulator_cal[loop1]))
        f_2.write(str(ev_all))
        f_2.write('\n')


# plot
x_list_1 = dens  # dens
y_list_1 = ccd_cal  # ccd
y_list_2 = emulator_cal  # extrapolation

print(x_list_1)
print(y_list_1)


file_path = "subspace_sample_5point.txt"
data_num = input_raw_data_count(file_path)
raw_data_2 = np.zeros((data_num,2),dtype = np.float)
input_file_2(file_path,raw_data_2)

x_list_3 = raw_data_2[:,0] # subspace data
y_list_3 = raw_data_2[:,1] #

print(x_list_3)
print(y_list_3)

interpol_count = 100 

dens = x_list_1
spl_pnm = interpolate.UnivariateSpline(dens,y_list_2,k = 5)
spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)

interp_pnm = spl_pnm(spldens)

x_list_2_new = spldens
y_list_2_new = interp_pnm 

# start plotting
fig1 = plt.figure('fig1')
l1 = plt.scatter(x_list_1,y_list_1,color = 'r', marker = 'd',zorder=1,label="CCD calculation")
l2 = plt.plot(x_list_2_new,y_list_2_new,color = 'k',linestyle=':',linewidth=1.5,alpha=0.9,  label="SP-CC(5)",zorder=1)
l3 = plt.plot(x_list_3,y_list_3,color = 'k', marker = 'o',markersize = 10,markerfacecolor='none',linestyle='',zorder=3, label="subspace samples")

#plt.yticks(np.arange(8,24,2),fontsize = 13) 
#plt.xticks(np.arange(0.12,0.205,0.01),fontsize = 13) 
plt.legend(loc='lower right',fontsize = 13)

plt.ylabel('$E/A$ [MeV]',fontsize=18)
plt.xlabel(r"$\rho$ [fm$^{-3}$]",fontsize=18)

plot_path = 'density_extrapolation.pdf'
plt.savefig(plot_path,bbox_inches='tight')









