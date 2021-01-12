import numpy as np
import re
import os
from validation import io_1
from validation import NM_emulator
import sympy
import time

# multiprocessing modules
import multiprocessing as mp
import psutil
from functools import partial

def evaluate_NM_batch(domain_points):
    observable_batch = []
    for loop in range(len(domain_points)):
        observable_batch.append(evaluate_NM(domain_points[loop]))
    return observable_batch


def evaluate_NM(domain_point):
    """
    Evaluate emulator at a single domain point

    Returns:
        observables (array of floats): eigvals[0], obs_vals
    """
    emulator_switch= 5
    interpolation_choice = 'GP'

    density_count = 5
    density_batch =  [0.12,0.14,0.16,0.18,0.20]
    pnm_data      = np.zeros(density_count)
    snm_data      = np.zeros(density_count)
    pnm_hf_data   = np.zeros(density_count)
    snm_hf_data   = np.zeros(density_count)
    LEC_target    = np.zeros(domain_dimension)
    LEC_target    = domain_point[1:18]
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
        pnm_data[loop],eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [],LEC_target)   
        pnm_data[loop]    = pnm_data[loop]/66


        C = hf_matrices_pnm[loop*18]
        LEC_all_matrix = hf_matrices_pnm[loop*18+1:loop*18+19]
        pnm_hf_data[loop] = NM_emulator.hf_emulator(LEC_all_matrix,C,LEC_target)


    ###########################################
    ##### snm calculation for different density
    ###########################################
    for loop in range(density_count):
        for loop1 in range(domain_dimension):

            H_matrix[loop1,:,:] = subspace_matrices_NM_snm[loop*18+ loop1+1][0]
        C  = subspace_matrices_NM_snm[loop*18][0]
        N  = subspace_norm_matrix_NM_snm[loop][0]
        snm_data[loop],eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [],LEC_target)   
        snm_data[loop] = snm_data[loop]/132


        C = hf_matrices_snm[loop*18]
        LEC_all_matrix = hf_matrices_snm[loop*18+1:loop*18+19]
        snm_hf_data[loop] = NM_emulator.hf_emulator(LEC_all_matrix,C,LEC_target)

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



################################
## main
################################
time_start = time.time()
#ccd_batch = io_1.read_ccd_data(input_dir="/home/slime/subspace_CC/test/emulator/DNNLO394/pnm_66_0.12_DNNLOgo_christian_64points/ccd.out",data_count = 64)

#cc_data_path  = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/%s_%d_%.2f_DNNLO_christian_34points/%s"

density_min   = 0.12
density_max   = 0.20
density_gap   = 0.02
density_count = 5
#validation_count  = 34
matter_type   = "pnm"
particle_num  = 66

#ccd_pnm_batch_all = np.zeros((validation_count,density_count))
#ccd_snm_batch_all = np.zeros((validation_count,density_count))
#density_batch_all = np.zeros((validation_count,density_count))
LEC_batch     = io_1.read_LEC_batch("LEC_read5.txt")
LEC_set_num   = np.arange(len(LEC_batch))
LEC_set_num   = LEC_set_num.reshape(-1,1)
LEC_batch_new = np.concatenate((LEC_set_num,LEC_batch),axis=1)
print(len(LEC_batch))
print(LEC_batch_new)

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

    # load hf matrix  
    database_dir = path + 'hf_emulator/%s_%d_%.2f_hf/' % (matter_type,particle_num,density)
    in_dir = database_dir+"LEC_hf_C_matrix"
    hf_matrices_pnm.append(np.loadtxt(in_dir))
    for loop1 in range(domain_dimension):
        in_dir = database_dir+"LEC_hf_"+str(loop1+1)+"_matrix"
        hf_matrices_pnm.append(np.loadtxt(in_dir))

    print("loading " + database_dir + " pnm data")


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

    # load hf matrix  
    database_dir = path + 'hf_emulator/%s_%d_%.2f_hf/' % (matter_type,particle_num,density)
    in_dir = database_dir+"LEC_hf_C_matrix"
    hf_matrices_snm.append(np.loadtxt(in_dir))
    for loop1 in range(domain_dimension):
        in_dir = database_dir+"LEC_hf_"+str(loop1+1)+"_matrix"
        hf_matrices_snm.append(np.loadtxt(in_dir))

    print("loading " + database_dir + " snm data")


print("done loading NM ")

### check for single point
#LEC_test = io_1.read_LEC("ccm_in_DNNLO394")
#observables = evaluate_NM(LEC_test)
#io_1.plot_9(raw_data[0],raw_data[1],raw_data[2],raw_data[3],raw_data[4],raw_data[10])
#
#LEC_test  = LEC_batch[18]
#observables, eigvec_temp,raw_data = evaluate_NM(LEC_test)
#io_1.plot_9(raw_data[0],raw_data[1],raw_data[2],raw_data[3],raw_data[4],raw_data[10])

#print(raw_data[4])
#print(raw_data[10])


### start to calculate the whole batch 
#with open("NM_ccd_589_samples.txt","a") as f:
#    f.write("##  saturation density[fm^-3]  saturation energy[MeV]  symmetry energy[MeV]      L     K\n")

#print(LEC_batch[0:10])
#print("wtf")
#print(np.array_split(LEC_batch[0:10],3,axis=0))

nprocs = psutil.cpu_count(logical=False)
os.environ["OMP_NUM_THREADS"] = "1"
print("nprocs = "+str(nprocs))

#observables_batch = evaluate_NM_batch(LEC_batch[0:5])
#print(observables_batch)

with mp.Pool(processes= nprocs) as p:
    observables_batch = p.map(evaluate_NM_batch,np.array_split(LEC_batch_new[0:8],nprocs,axis=0))
observables_batch = np.vstack(observables_batch)
#for loop in range(len(observables_batch)):
#    print(observables_batch[loop])
#
#print(observables_batch)


time_end = time.time()
print('time cost',time_end-time_start,'s')

for loop in range(8):
    with open("NM_ccd_800k_samples.txt","a") as f:
        f.write("%6d   %.4f   %.6f   %.6f   %.2f   %.1f       " % ( observables_batch[loop][0] ,observables_batch[loop][1] , observables_batch[loop][2] ,observables_batch[loop][3] ,observables_batch[loop][4], observables_batch[loop][5]))
        f.write("%.4f  %.4f  %.4f  %.4f  %.4f  "          % ( observables_batch[loop][6] ,observables_batch[loop][7] , observables_batch[loop][8] ,observables_batch[loop][9] ,observables_batch[loop][10]))
        f.write("%.4f  %.4f  %.4f  %.4f  %.4f  "          % ( observables_batch[loop][11],observables_batch[loop][12], observables_batch[loop][13],observables_batch[loop][14],observables_batch[loop][15]))
        f.write("%.4f  %.4f  %.4f  %.4f  %.4f  "          % ( observables_batch[loop][16],observables_batch[loop][17], observables_batch[loop][18],observables_batch[loop][19],observables_batch[loop][20]))
        f.write("%.4f  %.4f  %.4f  %.4f  %.4f \n "        % ( observables_batch[loop][21],observables_batch[loop][22], observables_batch[loop][23],observables_batch[loop][24],observables_batch[loop][25]))


#sample_count             = 589
#samples_num              = np.zeros(sample_count)
#saturation_density_batch = np.zeros(sample_count)
#saturation_energy_batch  = np.zeros(sample_count) 
#symmetry_energy_batch    = np.zeros(sample_count)
#L_batch                  = np.zeros(sample_count)
#K_batch                  = np.zeros(sample_count)
#
#file_path = "NM_ccd_589_samples.txt"
#with open(file_path,"r") as f_1:
#    count = len(open(file_path,"rU").readlines())
#    data  = f_1.readlines()
#    wtf = re.match('#','abc',flags=0)
#    for loop1 in range(0,count-1):
#        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
#        samples_num[loop1]              = float(temp_1[0])
#        saturation_density_batch[loop1] = float(temp_1[1])
#        saturation_energy_batch[loop1]  = float(temp_1[2]) 
#        symmetry_energy_batch[loop1]    = float(temp_1[3])
#        L_batch[loop1]                  = float(temp_1[4])
#        K_batch[loop1]                  = float(temp_1[5])

       
#file_path = "NM_ccd_589_samples_new.txt"
#with open(file_path,"w") as f_2:
#    f_2.write("num, saturation density[fm^-3], saturation energy[MeV], symmetry energy[MeV],       L,       K\n")
#
#for loop in range(len(LEC_batch)):
#    with open(file_path,"a") as f_2:
#        if ( float(saturation_density_batch[loop])==0.12 or float(saturation_density_batch[loop])==0.20 ):
#            f_2.write("%2d,            ,                            ,                     ,                  ,        \n" % (loop))
#        else:
#            f_2.write("%2d,          %.4f,                  %.6f,             %.6f,          %.2f,   %.1f \n" % (int(samples_num[loop]),float(saturation_density_batch[loop]),float(saturation_energy_batch[loop]),float(symmetry_energy_batch[loop]),float(L_batch[loop]),float(K_batch[loop])))


#samples_num_cut              = samples_num[np.where(saturation_density_batch!=0.12)]
#saturation_density_batch_cut = saturation_density_batch[np.where(saturation_density_batch!=0.12)]
#saturation_energy_batch_cut  = saturation_energy_batch [np.where(saturation_density_batch!=0.12)]
#symmetry_energy_batch_cut    = symmetry_energy_batch [np.where(saturation_density_batch!=0.12)]
#L_batch_cut                  = L_batch [np.where(saturation_density_batch!=0.12)]
#K_batch_cut                  = K_batch [np.where(saturation_density_batch!=0.12)]


#samples_num_cut              = samples_num_cut             [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#saturation_density_batch_cut = saturation_density_batch_cut[np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#saturation_energy_batch_cut  = saturation_energy_batch_cut [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#symmetry_energy_batch_cut    = symmetry_energy_batch_cut   [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#L_batch_cut                  = L_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
#K_batch_cut                  = K_batch_cut                 [np.where((K_batch_cut < 800) & (K_batch_cut > 0))]
                  
#print("samples_num_cut:"+str(samples_num_cut)) 



#for loop in range(density_count):
#    density = round(density_min + loop*density_gap,2)
#    matter_type   = "pnm"
#    particle_num  = 66
#    ccd_pnm_batch_all[:,loop]=io_1.read_ccd_data(input_dir = cc_data_path % (matter_type,particle_num,density,"ccdt.out"),data_count=validation_count)/particle_num
#    density_batch_all[:,loop]= density
#    matter_type   = "snm"
#    particle_num  = 132
#    ccd_snm_batch_all[:,loop]=io_1.read_ccd_data(input_dir = cc_data_path % (matter_type,particle_num,density,"ccdt_n3.out"),data_count=validation_count)/particle_num


#for loop in range(density_count):
#    density = round(density_min + loop*density_gap,2)
#    matter_type   = "pnm"
#    particle_num  = 66
#    ccd_pnm_batch_all[:,loop]=emulator.read_ccd_data(input_dir = cc_data_path % (matter_type,particle_num,density,"ccdt.out"),data_count=validation_count)/particle_num
#    density_batch_all[:,loop]= density
#    matter_type   = "snm"
#    particle_num  = 132
#    ccd_snm_batch_all[:,loop]=emulator.read_ccd_data(input_dir = cc_data_path % (matter_type,particle_num,density,"ccdt_n3.out"),data_count=validation_count)/particle_num
#


#print(density_batch_all)
#print(len(ccd_pnm_batch_all[:]))
#print(len(ccd_snm_batch_all[:]))
#saturation_density_batch, saturation_energy_batch, symmetry_energy_batch,L_batch,K_batch = io_1.generate_NM_observable_batch(ccd_pnm_batch_all,ccd_snm_batch_all,density_batch_all,"interpolate")
#with open("pb208_NM_ccdt.txt","w") as f:
#    f.write("##  saturation density[fm^-3]  saturation energy[MeV]  symmetry energy[MeV]      L     K\n")
#    for loop in range(validation_count):
#        f.write("%2d        %.4f                   %.6f              %.6f           %.2f       %.1f \n" % (loop, saturation_density_batch[loop],saturation_energy_batch[loop], symmetry_energy_batch[loop],L_batch[loop],K_batch[loop]))
#     


#io_1.plot_3(saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,K_batch_cut,L_batch_cut)


#rskin_batch = io_1.read_rskin_data("./pb208_rskin.txt",validation_count)
#
#io_1.plot_4(rskin_batch,symmetry_energy_batch,L_batch,K_batch)
#print(rskin_batch)


#ccd_pnm_batch_1 =  [11.50824259, 13.52328119,15.77696707,18.23395481,20.85337956]
#ccd_snm_batch_1 =  [-14.43847422,-15.11123482,-15.38290713,-15.23134748,-14.65166185]
#density_batch_1 =  [0.12,0.14,0.16,0.18,0.20]
##
#saturation_density_batch, saturation_energy_batch, symmetry_energy_batch,L_batch,K_batch,raw_data = io_1.generate_NM_observable(ccd_pnm_batch_1,ccd_snm_batch_1,density_batch_1,"GP")
#train_x   = raw_data[0]
#train_y_1 = raw_data[1]
#train_y_2 = raw_data[2]
#dens_list = raw_data[3].T[0]
#pnm       = raw_data[4] 
#pnm_cov   = raw_data[5] 
#d_pnm     = raw_data[6] 
#d_pnm_cov = raw_data[7] 
#dd_pnm    = raw_data[8] 
#dd_pnm_cov= raw_data[9] 
#snm       = raw_data[10] 
#snm_cov   = raw_data[11] 
#d_snm     = raw_data[12] 
#d_snm_cov = raw_data[13]
#dd_snm    = raw_data[14] 
#dd_snm_cov= raw_data[15] 
#
#
#saturation_density_batch, saturation_energy_batch, symmetry_energy_batch,L_batch,K_batch,raw_data = io_1.generate_NM_observable(ccd_pnm_batch_1,ccd_snm_batch_1,density_batch_1,"interpolation")
#
#dens_list_2 = raw_data[0]
#pnm_2     = raw_data[1] 
#d_pnm_2   = raw_data[3] 
#dd_pnm_2  = raw_data[5] 
#snm_2     = raw_data[7] 
#d_snm_2   = raw_data[9] 
#dd_snm_2  = raw_data[11] 
#
#
##io_1.plot_6(train_x, train_y_1,train_y_2,dens_list,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm, d_snm_cov, dd_snm, dd_snm_cov,dens_list_2,pnm_2,d_pnm_2,dd_pnm_2,snm_2,d_snm_2, dd_snm_2 )
#
#print(saturation_density_batch)
#print(saturation_energy_batch)
#print(symmetry_energy_batch)
#print(L_batch)
#print(K_batch)

#sympy.E
#x1 = sympy.Symbol('x1')
#x2 = sympy.Symbol('x2')
#a = sympy.Symbol('a')
#l = sympy.Symbol('l')
#f = sympy.Function('f')(x1,x2,a,l)
#
#f = a**2* sympy.E**(-(x1-x2)**2/2/l**2)
#print(sympy.diff(f,x1,2,x2,2).subs(l,2))

