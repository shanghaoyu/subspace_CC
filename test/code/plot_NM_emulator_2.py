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
from validation import io_1
from validation import NM_emulator


def validation(matter_type,particle_num):
    my_path     = "/home/slime/subspace_CC/test/"
    ccd_data = np.zeros((validation_count,density_count))
    for loop in range(5):
        density = round(density_min + loop*density_gap,2)
        database_dir = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/" % (matter_type,particle_num,density)
        #print(database_dir)
        file_path    = my_path + "LEC_read2.txt"
        input_dir    = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
        print(input_dir)
        validation_dir=my_path + "validation/DNNLO394_validation/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
    
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
        ccd_data[:,loop] = ccd_1

        #with open(file_path,'r') as f_2:
        #    count = len(open(file_path,'rU').readlines())
        #    data = f_2.readlines()
        #    wtf = re.match('#', 'abc',flags=0)
        #    for loop1 in range(1,count):
        #        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
        #        LEC[0]    = temp_1[16]
        #        LEC[1]    = temp_1[15]
        #        LEC[2:6]  = temp_1[11:15]
        #        LEC[6:10] = temp_1[0:4]
        #        LEC[10:17]= temp_1[4:11]
        #        ccd_data.append(ccd_1[loop1-1])
        #        #eigvalue, eigvec = emulator1(database_dir,LEC,subtract)
        #        #gs = eigvalue[0]
        #        #emulator_data.append(gs)
        #        density_data.append(density)
    return ccd_data 

def evaluate_NM(domain_point):
    emulator_switch=5
    interpolation_choice = 'GP'
    
    density_count = 5
    density_batch =  [0.12,0.14,0.16,0.18,0.20]
    pnm_data      = np.zeros(density_count)
    snm_data      = np.zeros(density_count)
    vote_pnm      = np.zeros(density_count)
    vote_snm      = np.zeros(density_count)
    pnm_hf_data   = np.zeros(density_count)
    snm_hf_data   = np.zeros(density_count)
    LEC_target    = np.zeros(domain_dimension)
    LEC_target    = domain_point[1:18]
    hyperparameter = np.zeros(5)
    #LEC_target    = domain_point[1:18]
    # test
    #LEC_target    =  read_LEC("/home/slime/history_matching/O28_prediction/history_matching/emulators/ccm_in_DNNLO394")
    
    ###########################################
    ##### pnm calculation for different density
    ###########################################
    H_matrix = np.zeros((domain_dimension,subspace_dimension,subspace_dimension))
    LEC_all_matrix = np.zeros(domain_dimension)
    
    
    for loop in range(density_count):
        for loop1 in range(domain_dimension):
            H_matrix[loop1,:,:] = subspace_matrices_NM_pnm[loop*18+ loop1+1][0]
        C  = subspace_matrices_NM_pnm[loop*18][0]
        N  = subspace_norm_matrix_NM_pnm[loop][0]
        pnm_data[loop],eigvec_temp,vote_pnm[loop] = NM_emulator.emulator(emulator_switch,"pnm",H_matrix, C, N, [],LEC_target,hyperparameter)
        pnm_data[loop]    = pnm_data[loop]/66
    
    ###########################################
    ##### snm calculation for different density
    ###########################################
    for loop in range(density_count):
        for loop1 in range(domain_dimension):
    
            H_matrix[loop1,:,:] = subspace_matrices_NM_snm[loop*18+ loop1+1][0]
        C  = subspace_matrices_NM_snm[loop*18][0]
        N  = subspace_norm_matrix_NM_snm[loop][0]
        snm_data[loop],eigvec_temp,vote_snm[loop] = NM_emulator.emulator(emulator_switch,"snm",H_matrix, C, N, [],LEC_target,hyperparameter)
        snm_data[loop] = snm_data[loop]/132

    return pnm_data, snm_data, vote_pnm, vote_snm 

#def hyperparameter_optimization():





# main
density_min   = 0.12
density_max   = 0.20
density_gap   = 0.02
density_count = 5
validation_count  = 50

LEC_batch     = io_1.read_LEC_batch("LEC_read2.txt")
LEC_set_num   = np.arange(len(LEC_batch))
LEC_set_num   = LEC_set_num.reshape(-1,1)
LEC_batch_new = np.concatenate((LEC_set_num,LEC_batch),axis=1)

# load emulator matrix
path = "/home/slime/subspace_CC/test/emulator/DNNLO394/"
domain_dimension =  17
subspace_dimension = 64
subspace_matrices_NM_snm = []
subspace_matrices_NM_pnm = []
subspace_norm_matrix_NM_snm = []
subspace_norm_matrix_NM_pnm = []
subtract_snm = []
subtract_pnm = []
hf_matrices_snm  = []
hf_matrices_pnm  = []

for loop in range(density_count):
    matter_type = 'pnm'
    particle_num = 66
    density = density_min + loop*density_gap
    database_dir = path + '%s_%d_%.2f_DNNLOgo_christian_64points/' % (matter_type,particle_num,density)
    if (matter_type == "snm") :
        expectation = [-1831,-1921,-1955,-1932,-1852]
    else:
        expectation = [763,896,1044,1206,1378]
    input_dir = path + "%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
    #S = find_subtract(input_dir,expectation[loop],matter_type)
    #m.subtract_pnm.append([S])
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    subspace_norm_matrix_NM_pnm.append([N])
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    subspace_matrices_NM_pnm.append([C])
    for loop1 in range(domain_dimension):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        mtx = np.loadtxt(in_dir)
        subspace_matrices_NM_pnm.append([mtx])

for loop in range(density_count):
    matter_type = 'snm'
    particle_num = 132
    density = density_min + loop*density_gap
    database_dir = path + '%s_%d_%.2f_DNNLOgo_christian_64points/' % (matter_type,particle_num,density)
    if (matter_type == "snm") :
        expectation = [-1831,-1921,-1955,-1932,-1852]
    else:
        expectation = [763,896,1044,1206,1378]
    input_dir = path + "%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % (matter_type,particle_num,density)
    #S = find_subtract(input_dir,expectation[loop],matter_type)
    #subtract_snm.append([S])
    in_dir = database_dir+"N_matrix.txt"
    N = np.loadtxt(in_dir)
    subspace_norm_matrix_NM_snm.append([N])
    in_dir = database_dir+"C_matrix.txt"
    C = np.loadtxt(in_dir)
    subspace_matrices_NM_snm.append([C])
    for loop1 in range(domain_dimension):
        in_dir = database_dir+"LEC_"+str(loop1+1)+"_matrix"
        mtx = np.loadtxt(in_dir)
        subspace_matrices_NM_snm.append([mtx])


# read 50 sets of validation data
ccd_pnm = validation('pnm',66)/66
ccd_snm = validation('snm',132)/132


emulator_pnm = np.zeros((validation_count,density_count))
emulator_snm = np.zeros((validation_count,density_count))
emulator_pnm_vote = np.zeros((validation_count,density_count))
emulator_snm_vote = np.zeros((validation_count,density_count))


for loop in range(validation_count):
    temp_pnm,temp_snm,temp_pnm_vote,temp_snm_vote = evaluate_NM(LEC_batch_new[loop])
    emulator_pnm[loop,:]= temp_pnm
    emulator_snm[loop,:]= temp_snm
    emulator_pnm_vote[loop,:]= temp_pnm_vote
    emulator_snm_vote[loop,:]= temp_snm_vote


print(ccd_pnm)
print(ccd_snm)
print(emulator_pnm)
print(emulator_snm)

#ccd_pnm =           ccd_pnm[:,]
#ccd_snm =           ccd_snm[:,]
#emulator_pnm = emulator_pnm[:,]
#emulator_snm = emulator_snm[:,]

# plot
x_list_1_pnm = ccd_pnm 
x_list_1_pnm_vote = emulator_pnm_vote
y_list_1_pnm = (emulator_pnm - ccd_pnm)/abs(ccd_pnm)
y_list_11_pnm = emulator_pnm 

x_list_1_snm = ccd_snm
x_list_1_snm_vote = emulator_snm_vote
y_list_1_snm = (emulator_snm - ccd_snm)/abs(ccd_snm)
y_list_11_snm = emulator_snm

sns.set_style("white")
fig1 = plt.figure('fig1')
plt.figure(figsize=(5,9))
plt.subplots_adjust(wspace =0.3, hspace =0.2)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax = plt.subplot(211)
ax.grid(False)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tick_params(top=True,bottom=True,left=True,right=True,length = 4,width=1.5,color="k")

plt.hlines(0,5 , 35,linestyle=':',alpha = 0.6)
l1 = plt.plot(x_list_1_pnm,y_list_1_pnm,color = 'b', linestyle="",marker = 's',markersize=5,zorder=1,label="CCD calculation")
#plt.xlim(5,25)
#plt.ylim(5,25)
plt.xlim(5,35)
#plt.ylim(-7.5,0.5)
#plt.ylabel(r'$\rm{CCD}-\rm{emulator}$  [MeV]',fontsize=18)
#plt.ylabel(r'$(\rm{CCD}-\rm{emulator})/|CCD|$  ',fontsize=18)
plt.ylabel('Relative Change',fontsize=18)
plt.xlabel('CCD $E/N$ [MeV]',fontsize=18)
#plt.xlabel('Votes',fontsize=18)


ax = plt.subplot(212)
ax.grid(False)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tick_params(top=True,bottom=True,left=True,right=True,length = 4,width=1.5,color="k")


plt.hlines(0,-20 , -5,linestyle=':',alpha = 0.6)
l1 = plt.plot(x_list_1_snm,y_list_1_snm,color = 'b', linestyle="",marker = 's',markersize=5,zorder=1,label="CCD calculation")
#plt.xlim(-22.5,-11.5)
#plt.ylim(-22.5,-11.5)
plt.xlim(-20,-5)
#plt.ylim(-2.5,0.5)
#plt.ylabel(r'$\rm{CCD}-\rm{emulator}$  [MeV]',fontsize=18)
#plt.ylabel(r'$(\rm{CCD}-\rm{emulator})/|CCD|$  ',fontsize=18)
plt.ylabel('Relative Change',fontsize=18)
plt.xlabel('CCD $E/A$ [MeV]',fontsize=18)
#plt.xlabel('Votes',fontsize=18)


#plot_path = 'emulator_vs_ccd_without_small_batch_voting_2.pdf'
plot_path = 'emulator_vs_ccd_with_small_batch_voting_51.pdf'
plt.savefig(plot_path,bbox_inches='tight')

