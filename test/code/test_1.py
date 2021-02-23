import numpy as np
import re
from validation import io_1
from validation import NM_emulator
import sympy


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
    LEC_target    = np.zeros(domain_dimension)
    LEC_target    = domain_point
    # test
    #LEC_target    =  read_LEC("/home/slime/history_matching/O28_prediction/history_matching/emulators/ccm_in_DNNLO394")

    ###########################################
    ##### pnm calculation for different density
    ###########################################
    H_matrix = np.zeros((domain_dimension,subspace_dimension,subspace_dimension))
    for loop in range(density_count):
        for loop1 in range(domain_dimension):
            H_matrix[loop1,:,:] = subspace_matrices_NM_pnm[loop*18+ loop1+1][0]
        C  = subspace_matrices_NM_pnm[loop*18][0]
        N  = subspace_norm_matrix_NM_pnm[loop][0]
        pnm_data[loop],eigvec_temp,vote_temp = NM_emulator.emulator(emulator_switch,H_matrix, C, N, [],LEC_target)   
        pnm_data[loop] = pnm_data[loop]/66

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

    print("pnm："+str(pnm_data))
    print("snm: "+str(snm_data))
    ###########################################
    ##### calculate NM observables
    ###########################################
    saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = io_1.generate_NM_observable(pnm_data,snm_data,density_batch,"GP")
    observables = np.array([saturation_density]+ [saturation_energy]+ [symmetry_energy]+[L]+[K])
    print("saturation_density="+str(observables[0]))
    print("saturation_energy="+str(observables[1]))
    print("symmetry_energy="+str(observables[2]))
    print("L="+str(observables[3]))
    print("K="+str(observables[4]))
    return observables, eigvec_temp, raw_data



################################
## main
################################

#ccd_batch = io_1.read_ccd_data(input_dir="/home/slime/subspace_CC/test/emulator/DNNLO394/pnm_66_0.12_DNNLOgo_christian_64points/ccd.out",data_count = 64)

cc_data_path  = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/%s_%d_%.2f_DNNLO_christian_34points/%s"

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

LEC_batch = io_1.read_LEC_batch("LEC_read4.txt")

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
print("loading " + database_dir + " snm data")
print("done loading NM ")

### check for single point
#LEC_test = io_1.read_LEC("ccm_in_DNNLO394")
#observables, eigvec_temp,raw_data = evaluate_NM(LEC_test)
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
#
#for loop in range(len(LEC_batch)):
#    observables, eigvec_temp,raw_data = evaluate_NM(LEC_batch[loop])
#    with open("NM_ccd_589_samples.txt","a") as f:
#        f.write("%3d        %.4f                   %.6f              %.6f           %.2f     %.1f \n" % (loop, observables[0],observables[1], observables[2],observables[3],observables[4]))
#    with open("NM_ccd_589_samples_detail.txt","a") as f:
#        f.write(str(loop)+"   pnm: "+str(raw_data[1])+ "   snm:"+str(raw_data[2])+"\n")
#
#
sample_count             = 34 
samples_num              = np.zeros(sample_count)
saturation_density_batch = np.zeros(sample_count)
saturation_energy_batch  = np.zeros(sample_count) 
symmetry_energy_batch    = np.zeros(sample_count)
L_batch                  = np.zeros(sample_count)
K_batch                  = np.zeros(sample_count)

density_count = 8                     
validation_count = 34
ccd_pnm_batch_all = np.zeros((validation_count,density_count)) 
ccd_snm_batch_all = np.zeros((validation_count,density_count))                  
density_batch_all = np.zeros((validation_count,density_count))

database_dir = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/"
for loop1 in range(density_count):
    dens = 0.06 + loop1 * 0.02
    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt.out" % ('pnm',66,dens)
    ccd_pnm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/66

    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt_n3.out" % ('snm',132,dens)
    ccd_snm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/132
    density_batch_all[:,loop1] = dens

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
for loop in range(validation_count):
    saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = io_1.generate_NM_observable(ccd_pnm_batch_all[loop],ccd_snm_batch_all[loop],density_batch_all[loop],switch="GP")
    #sub_saturation_iX = ((saturation_density * 0.7 - density_batch_all.min())/(density_batch_all.max()-density_batch_all.min()) * len(raw_data[16])).astype(int)
    sub_saturation_iX  = round((0.11-0.06)/(0.20-0.06)*1400)
    print(sub_saturation_iX)
    print(raw_data[3][sub_saturation_iX])
    K_batch[loop]               = raw_data[16][sub_saturation_iX]
    symmetry_energy_batch[loop] = raw_data[17][sub_saturation_iX]
    L_batch[loop]               = raw_data[18][sub_saturation_iX]

#with open("pb208_NM_ccdt.txt","w") as f:
#    f.write("##  saturation density[fm^-3]  saturation energy[MeV]  symmetry energy[MeV]      L     K\n")
#    for loop in range(validation_count):
#        f.write("%2d        %.4f                   %.6f              %.6f           %.2f       %.1f \n" % (loop, saturation_density_batch[loop],saturation_energy_batch[loop], symmetry_energy_batch[loop],L_batch[loop],K_batch[loop]))
#     

#io_1.plot_3(saturation_density_batch_cut,saturation_energy_batch_cut,symmetry_energy_batch_cut,K_batch_cut,L_batch_cut)


rskin_batch = io_1.read_rskin_data("./pb208_rskin.txt",validation_count,1,3)
#
io_1.plot_4(rskin_batch,symmetry_energy_batch,L_batch,K_batch)
#print(rskin_batch)


