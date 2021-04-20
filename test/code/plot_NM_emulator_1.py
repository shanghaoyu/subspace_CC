import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import interpolate
from validation import io_1
from validation import NM_emulator

######################################################
######################################################
### generate nuclear matter infile
######################################################
######################################################
def output_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax,option):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write('3, 394\n')
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
        f_1.write('CCD'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 3'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('394, 4'+'\n')

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

######################################################
######################################################
### call CCM_nuclear_matter
######################################################
######################################################
def nuclear_matter(vec_input,dens,matter_type):
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
    if matter_type == "pnm":
        nucl_matt_in_dir   = './ccm_in_pnm_%.2f' % (density)
        nucl_matt_out_dir  = './pnm_rho_%.2f.out' % (density)
        option = 'PBC'
        output_ccm_in_file(nucl_matt_in_dir,vec_input,neutron_num,matter_type,density,nmax,option)
        os.system('./'+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir)
        ccd = read_nucl_matt_out(nucl_matt_out_dir)

    if matter_type == "snm":
        nucl_matt_in_dir   = './ccm_in_snm_%.2f' % (density)
        nucl_matt_out_dir  = './snm_rho_%.2f.out' % (density)
        option = 'PBC'
        output_ccm_in_file(nucl_matt_in_dir,vec_input,particle_num,matter_type,density,nmax,option)
        os.system('./'+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir)
        ccd = read_nucl_matt_out(nucl_matt_out_dir)



    print ("ccd energy from real CC calculation: "+str(ccd))
    return ccd

######################################################
######################################################
### main
######################################################
######################################################
# print LECs sets (only one parameter is changed)
target_LEC_count = 10 
nucl_matt_exe = './prog_ccm.exe'
#parameter_min = LEC_DNNLO394[target_LEC_count]*0.5
#parameter_max = LEC_DNNLO394[target_LEC_count]*1.5
#parameter_gap =  abs(LEC_DNNLO394[target_LEC_count]*0.1)
parameter_min = 2 
parameter_max = 6
parameter_gap = 0.2
parameter_count = abs(int((parameter_max-parameter_min)/parameter_gap))+1 
ccd_cal_pnm      = np.zeros(parameter_count)
ccd_cal_snm      = np.zeros(parameter_count)
emulator_cal_pnm_1 = np.zeros(parameter_count)
emulator_cal_snm_1 = np.zeros(parameter_count)
emulator_cal_pnm_2 = np.zeros(parameter_count)
emulator_cal_snm_2 = np.zeros(parameter_count)
ccd_precal_flag = True

LEC_DNNLO394 = io_1.read_LEC("ccm_in_DNNLO394")
LEC_batch = []

my_LEC_label = ['cE','cD','c1','c2','c3','c4','Ct1S0pp','Ct1S0np','Ct1S0nn','Ct3S1','C1S0','C3P0','C1P1','C3P1','C3S1','CE1','C3P2']
file_path = "LEC_read.txt"
with open(file_path,'r') as f:
    count = len(open(file_path,'rU').readlines())
    data = f.readlines()
    wtf = re.match('#', 'abc',flags=0)
    LEC_label = data[0].split()
    LEC_label = LEC_label[1::]

#    print("LEC_label"+str(LEC_label))
    x = []
#    print(len(my_LEC_label))
#    print(len(LEC_label))
    for loop1 in range(len(my_LEC_label)):
        for loop2 in range(len(my_LEC_label)):
            #print(str(LEC_label[loop1])+" "+str(my_LEC_label[loop2]))
            if LEC_label[loop1]==my_LEC_label[loop2]:
                x.append(loop2)
                break
    LEC_DNNLO394_new = LEC_DNNLO394[x].copy()
    # y is the index of target_LEC in other format
    for loop1 in range(len(my_LEC_label)):
        if x[loop1] == target_LEC_count:
            y = loop1 

with open("LEC_read_temp.txt",'w') as f:
    f.write("# \n")
    for loop1 in range(5):
        LEC_DNNLO394_new[y] = 3 +  parameter_gap * loop1  
        for loop2 in range(17):
            f.write("%.12f " % (LEC_DNNLO394_new[loop2]))
        f.write("\n")

print("parameter_origin ="+str(LEC_DNNLO394[target_LEC_count]))
print("parameter_count ="+str(parameter_count))

def ccd_calculation(matter_type):
    ccd_cal = []
    parameter_new = []
    if ccd_precal_flag == False:
        for loop1 in range(parameter_count):
            parameter_new.append ( parameter_min + (parameter_gap * loop1))
            # update one parameter
            LEC = LEC_DNNLO394
            LEC[target_LEC_count] = parameter_new[loop1]
            print(LEC)
            ccd_cal.append (nuclear_matter(LEC,0.16,matter_type))
        with open ("ccd_%s.txt"%(matter_type),"w") as f:
            for loop2 in range(parameter_count):
                f.write("%.6f   %.4f \n" %(parameter_new[loop2],ccd_cal[loop2]))
    
    elif ccd_precal_flag == True:
        with open ("ccd_%s.txt"%(matter_type),"r") as f_1:
            count = len(open("ccd_%s.txt"%(matter_type),'rU').readlines())
            data =  f_1.readlines()
            wtf = re.match('#', 'abc',flags=0)
            for loop1 in range(0,count):
                    temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                    parameter_new.append(float(temp_1[0]))
                    ccd_cal.append(float(temp_1[1]))

    return parameter_new,np.array(ccd_cal)
parameter_new , ccd_cal_pnm = ccd_calculation("pnm")
parameter_new , ccd_cal_snm = ccd_calculation("snm")
#ccd_cal_snm = ccd_calculation("snm")
print(ccd_cal_pnm)
print(ccd_cal_snm)

# load emulator matrix
path = "/home/slime/subspace_CC/test/emulator/DNNLOgo394_one_parameter_5point/"
domain_dimension =  17
subspace_dimension = 5
subspace_matrices_NM_snm = []
subspace_matrices_NM_pnm = []
subspace_norm_matrix_NM_snm = []
subspace_norm_matrix_NM_pnm = []
subtract_snm = []
subtract_pnm = []
hf_matrices_snm  = []
hf_matrices_pnm  = []

# load pnm emulator matrix
matter_type = 'pnm'
particle_num = 14
density = 0.16
database_dir = path + '%s_%d_%.2f_DNNLOgo394_one_parameter_5points/' % (matter_type,particle_num,density)
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

# load snm emulator matrix
matter_type = 'snm'
particle_num = 28
density = 0.16
database_dir = path + '%s_%d_%.2f_DNNLOgo394_one_parameter_5points/' % (matter_type,particle_num,density)
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



def evaluate_NM_5points(LEC_target,matter_type):
    emulator_switch = 1
    interpolation_choice = 'GP'
    H_matrix = np.zeros((domain_dimension,subspace_dimension,subspace_dimension))
    LEC_all_matrix = np.zeros(domain_dimension)
    if matter_type == "pnm":
        for loop in range(1):
            for loop1 in range(domain_dimension):
                H_matrix[loop1,:,:] = subspace_matrices_NM_pnm[loop*18+ loop1+1][0]
            C  = subspace_matrices_NM_pnm[loop*18][0]
            N  = subspace_norm_matrix_NM_pnm[loop][0]
            emulator_cal,eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [],LEC_target)
            #pnm_data[loop]    = pnm_data[loop]/66
    elif matter_type == "snm":
        for loop in range(1):
            for loop1 in range(domain_dimension):
                H_matrix[loop1,:,:] = subspace_matrices_NM_snm[loop*18+ loop1+1][0]
            C  = subspace_matrices_NM_snm[loop*18][0]
            N  = subspace_norm_matrix_NM_snm[loop][0]
            emulator_cal,eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [],LEC_target)
    return emulator_cal[0]    

def evaluate_NM_3points(LEC_target,matter_type):
    emulator_switch = 1
    interpolation_choice = 'GP'
    H_matrix = np.zeros((domain_dimension,subspace_dimension,subspace_dimension))
    LEC_all_matrix = np.zeros(domain_dimension)
    if matter_type == "pnm":
        for loop in range(1):
            for loop1 in range(domain_dimension):
                H_matrix[loop1,:,:] = subspace_matrices_NM_pnm[loop*18+ loop1+1][0]
            C  = subspace_matrices_NM_pnm[loop*18][0]
            N  = subspace_norm_matrix_NM_pnm[loop][0]
            emulator_cal,eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [0,1],LEC_target)
            #pnm_data[loop]    = pnm_data[loop]/66
    elif matter_type == "snm":
        for loop in range(1):
            for loop1 in range(domain_dimension):
                H_matrix[loop1,:,:] = subspace_matrices_NM_snm[loop*18+ loop1+1][0]
            C  = subspace_matrices_NM_snm[loop*18][0]
            N  = subspace_norm_matrix_NM_snm[loop][0]
            emulator_cal,eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [0,1],LEC_target)
    return emulator_cal[0]    


for loop1 in range(parameter_count):
    # update one parameter
    LEC = LEC_DNNLO394
    LEC[target_LEC_count] = parameter_new[loop1]
    emulator_cal_pnm_1[loop1]=evaluate_NM_5points(LEC_DNNLO394,"pnm")
    emulator_cal_snm_1[loop1]=evaluate_NM_5points(LEC_DNNLO394,"snm")

for loop1 in range(parameter_count):
    # update one parameter
    LEC = LEC_DNNLO394
    LEC[target_LEC_count] = parameter_new[loop1]
    emulator_cal_pnm_2[loop1]=evaluate_NM_3points(LEC_DNNLO394,"pnm")
    emulator_cal_snm_2[loop1]=evaluate_NM_3points(LEC_DNNLO394,"snm")


######################################################
######################################################
### plot
######################################################
######################################################
x_list_1 = parameter_new  
y_list_1_pnm = ccd_cal_pnm/14
y_list_1_snm = ccd_cal_snm/28

y_list_21_pnm = emulator_cal_pnm_1/14
y_list_21_snm = emulator_cal_snm_1/28

y_list_22_pnm = emulator_cal_pnm_2/14
y_list_22_snm = emulator_cal_snm_2/28

x_list_3 = np.arange(3,4,0.2)
y_list_3_pnm = ccd_cal_pnm[5:10]/14
y_list_3_snm = ccd_cal_snm[5:10]/28

dens = x_list_1
spl_pnm   = interpolate.UnivariateSpline(dens,y_list_21_pnm,k=4)
spl_snm   = interpolate.UnivariateSpline(dens,y_list_21_snm,k=4)
spl_pnm_2 = interpolate.UnivariateSpline(dens,y_list_22_pnm,k=4)
spl_snm_2 = interpolate.UnivariateSpline(dens,y_list_22_snm,k=4)
spldens = np.linspace(dens[0],dens[len(dens)-1],num = 100)
interp_pnm = spl_pnm(spldens)
interp_snm = spl_snm(spldens)
interp_pnm_2 = spl_pnm_2(spldens)
interp_snm_2 = spl_snm_2(spldens)

x_list_1_new = spldens
y_list_21_new_pnm = interp_pnm
y_list_21_new_snm = interp_snm
y_list_22_new_pnm = interp_pnm_2
y_list_22_new_snm = interp_snm_2


sns.set_style("white")
fig1 = plt.figure('fig1')
plt.figure(figsize=(5,8))
plt.subplots_adjust(wspace =0.3, hspace =0)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax = plt.subplot(211)
ax.grid(False)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tick_params(top=True,bottom=True,left=True,right=True,length = 4,width=1.5,color="k")

plt.vlines(2.505389, 3.5, 40,linestyle=':',alpha = 0.6)
plt.text(2.24,20,r"$ \Delta\rm{NNLO_{GO}(394)}$", rotation=90,fontsize= 15)

plt.text(3.775,24,"1", fontsize= 15)
plt.text(3.575,22.9,"2", fontsize= 15)
plt.text(3.375,21.4,"3", fontsize= 15)
plt.text(3.175,19.9,"4", fontsize= 15)
plt.text(2.975,17.8,"5", fontsize= 15)

l1 = plt.scatter(x_list_1,y_list_1_pnm,color = 'r', marker = 'd',zorder=1,label="CCD calculation")
l2 = plt.plot(x_list_1_new,y_list_21_new_pnm,color = 'k',linestyle='-',linewidth=2,alpha=0.9,  label="SP-CC(5)",zorder=1)
l2 = plt.plot(x_list_1_new,y_list_22_new_pnm,color = 'k',linestyle='--',linewidth=2,alpha=0.9,  label="SP-CC(3)",zorder=1)
l3 = plt.plot(x_list_3,y_list_3_pnm,color = 'k', marker = 'o',markersize = 12 ,markeredgewidth = 2,markerfacecolor='none',linestyle='',zorder=3, label="subspace samples")

#plt.yticks(np.arange(8,24,2),fontsize = 13)
#plt.xticks(np.arange(0.12,0.205,0.01),fontsize = 13)
#plt.legend(loc='lower right',fontsize = 13)
plt.ylim((3.5,40))
plt.ylabel('$E/N$ [MeV]',fontsize=18)
#plt.xlabel(r"Low-energy constant $C_{^1S_0}$ [$10^4 \rm{GeV^{-4}}$]",fontsize=18)

ax = plt.subplot(212)
ax.grid(False)
ax.spines['top'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
plt.tick_params(top=True,bottom=True,left=True,right=True,length = 4,width=1.5,color="k")

plt.vlines(2.505389, -22, -10,linestyle=':',alpha = 0.6)
plt.text(3.775,-12.2,"1", fontsize= 15)
plt.text(3.575,-12.5,"2", fontsize= 15)
plt.text(3.375,-13,"3", fontsize= 15)
plt.text(3.175,-13.5,"4", fontsize= 15)
plt.text(2.975,-14.25,"5", fontsize= 15)

l1 = plt.scatter(x_list_1,y_list_1_snm,color = 'r', marker = 'd',zorder=1,label="CCD calculation")
l2 = plt.plot(x_list_1_new,y_list_21_new_snm,color = 'k',linestyle='-',linewidth=2,alpha=0.9,  label="SP-CC(5) (points 1-5)",zorder=1)
l2 = plt.plot(x_list_1_new,y_list_22_new_snm,color = 'k',linestyle='--',linewidth=2,alpha=0.9,  label="SP-CC(3) (points 1-3)",zorder=1)
l3 = plt.plot(x_list_3,y_list_3_snm,color = 'k', marker = 'o',markersize = 12,markeredgewidth=2,markerfacecolor='none',linestyle='',zorder=3, label="subspace samples")

#plt.yticks(np.arange(8,24,2),fontsize = 13)
#plt.xticks(np.arange(0.12,0.205,0.01),fontsize = 13)
plt.legend(loc='lower right',fontsize = 13)

plt.ylim((-22,-10.1))
plt.ylabel('$E/A$ [MeV]',fontsize=18)
plt.xlabel(r"$C_{^1S_0}$ [$10^4 \rm{GeV^{-4}}$]",fontsize=18)



plot_path = 'emulator_vs_ccd_one_parameter.pdf'
plt.savefig(plot_path,bbox_inches='tight')


