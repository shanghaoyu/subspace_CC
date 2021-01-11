import numpy as np
from validation import io_1
from validation import NM_emulator
import sympy


################################
## main
################################

#ccd_batch = io_1.read_ccd_data(input_dir="/home/slime/subspace_CC/test/emulator/DNNLO394/pnm_66_0.12_DNNLOgo_christian_64points/ccd.out",data_count = 64)

cc_data_path  = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/%s_%d_%.2f_DNNLO_christian_34points/%s"

density_min   = 0.12
density_max   = 0.20
density_gap   = 0.02
density_count = 5
validation_count  = 34
matter_type   = "pnm"
particle_num  = 66

ccd_pnm_batch_all = np.zeros((validation_count,density_count))
ccd_snm_batch_all = np.zeros((validation_count,density_count))
density_batch_all = np.zeros((validation_count,density_count))

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
#io_1.plot_3(saturation_density_batch,saturation_energy_batch,symmetry_energy_batch,K_batch)
#rskin_batch = io_1.read_rskin_data("./pb208_rskin.txt",validation_count)
#
#io_1.plot_4(rskin_batch,symmetry_energy_batch,L_batch,K_batch)
#print(rskin_batch)


ccd_pnm_batch_1 =  [11.50824259, 13.52328119,15.77696707,18.23395481,20.85337956]
ccd_snm_batch_1 =  [-14.43847422,-15.11123482,-15.38290713,-15.23134748,-14.65166185]
density_batch_1 =  [0.12,0.14,0.16,0.18,0.20]
#
saturation_density_batch, saturation_energy_batch, symmetry_energy_batch,L_batch,K_batch,raw_data = io_1.generate_NM_observable(ccd_pnm_batch_1,ccd_snm_batch_1,density_batch_1,"GP")
train_x   = raw_data[0]
train_y_1 = raw_data[1]
train_y_2 = raw_data[2]
dens_list = raw_data[3].T[0]
pnm       = raw_data[4] 
pnm_cov   = raw_data[5] 
d_pnm     = raw_data[6] 
d_pnm_cov = raw_data[7] 
dd_pnm    = raw_data[8] 
dd_pnm_cov= raw_data[9] 
snm       = raw_data[10] 
snm_cov   = raw_data[11] 
d_snm     = raw_data[12] 
d_snm_cov = raw_data[13]
dd_snm    = raw_data[14] 
dd_snm_cov= raw_data[15] 


saturation_density_batch, saturation_energy_batch, symmetry_energy_batch,L_batch,K_batch,raw_data = io_1.generate_NM_observable(ccd_pnm_batch_1,ccd_snm_batch_1,density_batch_1,"interpolation")

dens_list_2 = raw_data[0]
pnm_2     = raw_data[1] 
d_pnm_2   = raw_data[3] 
dd_pnm_2  = raw_data[5] 
snm_2     = raw_data[7] 
d_snm_2   = raw_data[9] 
dd_snm_2  = raw_data[11] 


#io_1.plot_6(train_x, train_y_1,train_y_2,dens_list,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm, d_snm_cov, dd_snm, dd_snm_cov,dens_list_2,pnm_2,d_pnm_2,dd_pnm_2,snm_2,d_snm_2, dd_snm_2 )

print(saturation_density_batch)
print(saturation_energy_batch)
print(symmetry_energy_batch)
print(L_batch)
print(K_batch)

#sympy.E
#x1 = sympy.Symbol('x1')
#x2 = sympy.Symbol('x2')
#a = sympy.Symbol('a')
#l = sympy.Symbol('l')
#f = sympy.Function('f')(x1,x2,a,l)
#
#f = a**2* sympy.E**(-(x1-x2)**2/2/l**2)
#print(sympy.diff(f,x1,2,x2,2).subs(l,2))

