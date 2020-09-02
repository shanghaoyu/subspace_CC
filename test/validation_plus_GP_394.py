import os
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
import scipy.linalg as spla
import time
from scipy import interpolate

######################################################
######################################################
### GP tool
######################################################
######################################################
class GP_test:

    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_x, self.train_y = None, None
        self.sigma  = 100
        self.length = 0.25
        self.optimize = optimize
        self.gaussian_noise = 1

    def fit_data(self, x, y, gaussian_noise):
        # store train data
        self.train_x = np.asarray(x)
        self.train_y = np.asarray(y)
        self.gaussian_noise = gaussian_noise

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.l, self.sigma = params[0], params[1]
            Kyy = self.kernel(self.train_x, self.train_x) + 1e-8 * np.eye(len(self.train_x))
            return 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_x) * np.log(2 * np.pi)

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.length, self.sigma],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.length, self.sigma = res.x[0], res.x[1]
            
        self.is_fit = True

    def predict(self, x):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        x = np.asarray(x)
        # gaussian_noise**2 here is the variance of the gaussian like of noise in y ( y = f(x) + noise)  (noise = N (0, gaussian_noise**2))
        Kff = self.kernel(self.train_x, self.train_x) + self.gaussian_noise**2 * np.eye(len(self.train_x))  # (N, N)
        Kyy = self.kernel(x, x)  # (k, k)
        Kfy = self.kernel(self.train_x, x)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_x)))  # (N, N)
        
        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma ** 2 * np.exp(-0.5 / self.length ** 2 * dist_matrix)

#def yy(y, noise_sigma=1):
#    y = y + np.random.normal(0, noise_sigma, size=y.shape)
#    return y.tolist()

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
#    print ("ccd energy from real CC calculation: "+str(ccd))
    return ccd

######################################################
######################################################
### convergence test
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
    t5 = time.time()
    for loop1 in range(LEC_num):
        H = H + LEC_target[loop1] * H_matrix[loop1,:,:]
    H = H + C 

#   print("H="+str(H))
#    eigvals,eigvec = spla.eig(N)
#    print ("N eigvals = "+str(sorted(eigvals)))

   # print("rank of N ="+str(np.linalg.matrix_rank(N)))
    #print("N= "+str(N))

##### without subtract 
    subtract_1 = subtract
    print("subtract_1="+str(subtract_1))
    H = np.delete(H,subtract_1,axis = 0)
    H = np.delete(H,subtract_1,axis = 1) 
    N = np.delete(N,subtract_1,axis = 0)
    N = np.delete(N,subtract_1,axis = 1) 
    #np.savetxt('H.test',H,fmt='%.10f')
    #np.savetxt('N.test',N,fmt='%.10f')
    #H = np.loadtxt('H.test')
    #N = np.loadtxt('N.test')

### solve the general eigval problem
    eigvals,eigvec_L, eigvec_R = spla.eig(H,N,left =True,right=True)

### drop states with imaginary part
    eigvals_new   = eigvals[np.where(abs(eigvals.imag) < 0.01)]
    eigvals_R_new = eigvec_R[:,np.where(abs(eigvals.imag) < 0.01)]
    eigvals_R_new = eigvals_R_new.T

### sort with eigval
    x = np.argsort(eigvals_new)
    eigvals_new   = eigvals_new[x]
    eigvals_R_new = eigvals_R_new[x]
#    print(eigvals_new[0])
#    print(eigvals_R_new[0])
#    print(np.dot(eigvals_R_new[11],np.conjugate(eigvals_R_new[11].T)))

###### with subtract
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
#                ev_ultra = ev_sorted_1[loop2]
#                break
#    #print(ev_ultra)
#    #print(ev_sorted_1)
#    #print(ev_sorted_2)
#
    t6 = time.time()
    return eigvals_new[0], eigvals_R_new[0]

######################################################
######################################################
#### subtract
######################################################
######################################################
def find_subtract(matter_type,input_dir,expectation):
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
    #            print(file_count)
    #            print(ccd_1)
    return subtract



######################################################
######################################################
###  plot 
######################################################
######################################################
def plot_1():
    plt.figure()
    plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
#    plt.fill_between(test_x.ravel(), test_y_1 + confidence_1, test_y_1 - confidence_1, alpha=0.1)
#    plt.plot(test_x, test_y_1, label="predict")
    plt.scatter(train_x, train_y_1, label="train", c="red", marker="x")
    ccd_data_x = np.array([0.12,0.14,0.16,0.18,0.20])
    ccd_data_y = np.array([-1831,-1921,-1955,-1932,-1852])
    plt.scatter(ccd_data_x, ccd_data_y/132, label="ccd", c="black", marker="o")

    plt.legend()
    plot_path = 'snm_gp_test.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')
    
    plt.figure()
    plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
#    plt.fill_between(test_x.ravel(), test_y_2 + confidence_2, test_y_2 - confidence_2, alpha=0.1)
#    plt.plot(test_x, test_y_2, label="predict")
    plt.scatter(train_x, train_y_2, label="train", c="red", marker="x")

    ccd_data_x = np.array([0.12,0.14,0.16,0.18,0.20])
    ccd_data_y = np.array([763,896,1044,1206,1378])
 
    plt.scatter(ccd_data_x, ccd_data_y/66, label="ccd", c="black", marker="o")
    plt.legend()
    plot_path = 'pnm_gp_test.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')

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
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/DNNLOgo450_20percent_64points_/'
#database_dir = '/home/slime/work/Eigenvector_continuation/CCM_kspace_deltafull/test/emulator/'
#database_dir = '/home/slime/subspace_CC/test/emulator/'
#database_dir = '/home/slime/subspace_CC/test/emulator/snm_132_0.16_DNNLOgo_20percent_64points/'



#### pick out not converge CCD results
#converge_flag = np.zeros(subspace_dimension)
#find_notconverge('./',converge_flag)
#subtract = converge_flag.nonzero()
#print("converge_flag"+str(converge_flag))
#print(converge_flag.nonzero())
#subtract = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#subtract = [2,3,5,33,41,55,59]
#subtract = range(30,45)
#subtract_snm = [4,6,9,12,16,18,19,20,21,22,23,24,25,26,28,30,34,35,37,38,39,40,42,43,45,46,47,48,49,51,52,54,57,62]
#subtract_pnm = [0,2,3,9,10,16,17,18,22,26,27,28,30,33,37,39,40,42,43,44,45,47,48,52,54,55]
#subtract_pnm = [0,2,3,5,9,10,11,16,17,18,22,23,26,27,28,29,30,32,33,34,36,37,39,40,42,43,44,45,47,48,51,52,54,55,61,62]
subtract = []
dens_count = 5
pnm_data = np.zeros(dens_count)
snm_data = np.zeros(dens_count)
my_path      = "./"
error_sample = 0.05

t3 = time.time()
validation_count = 1
for loop1 in range(validation_count):
    file_path = "ccm_in_DNNLO394"
    LEC = read_LEC(file_path)
    #LEC_random = generate_random_LEC(LEC, LEC_range)
    LEC_random = LEC

    for loop2 in range(dens_count):
        dens = 0.12 + loop2 * 0.02
        dens = np.around(dens, 2 )
        database_dir = "./emulator/DNNLO394/pnm_66_"+str(("%.2f" % dens))+"_DNNLOgo_christian_64points/"
        input_dir = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % ('pnm',66,dens)
        expectation = [763,896,1044,1206,1378]
        subtract = find_subtract("pnm",input_dir,expectation[loop2])
        print(64-len(subtract))
        print(subtract)
        emulator_cal, emulator_vec = emulator(database_dir,LEC_random,subtract)
        pnm_data[loop2] = emulator_cal.real/66

    for loop2 in range(dens_count):
        dens = 0.12 + loop2 * 0.02
        dens = np.around(dens, 2 )
        database_dir = "./emulator/DNNLO394/snm_132_"+str(("%.2f" % dens))+"_DNNLOgo_christian_64points/"
        input_dir = my_path + "emulator/DNNLO394/%s_%d_%.2f_DNNLOgo_christian_64points/ccd.out" % ('snm',132,dens)
        expectation =[-1831,-1921,-1955,-1932,-1852] 
        subtract = find_subtract("snm",input_dir,expectation[loop2])
        print(64-len(subtract))
        print(subtract)

        emulator_cal, emulator_vec = emulator(database_dir,LEC_random,subtract)
        snm_data[loop2] = emulator_cal.real/132
#    print("snm: "+str(snm_data*132))

t4 = time.time()

print("time for snm+pnm : "+ str(t4-t3))
######################################################
######################################################
###  use GP to find the saturation point  
######################################################
######################################################
t1 = time.time()
train_x = np.arange(0.12,0.12+dens_count*0.02,0.02)
train_x = train_x.reshape(-1,1)
train_y_1 = snm_data
test_x  = np.arange(0.12,0.22,0.001).reshape(-1,1)

gpr = GP_test()
gaussian_noise = 0.02

gpr.fit_data(train_x, train_y_1, gaussian_noise)

snm, snm_cov = gpr.predict(test_x)

iX=np.argmin(snm)
test_y_1 = snm.ravel()
confidence_1 = 1.96 * np.sqrt(np.diag(snm_cov))

density_range = test_x[np.where((snm[:]<(snm[iX]+confidence_1[iX]))&(snm[:]>(snm[iX]-confidence_1[iX])))]


print("saturation density: %.3f +/- %.3f" % (test_x[iX], 0.5*(np.max(density_range)-np.min(density_range))))
print("saturation energy:  %.3f +/- %.3f" % (snm[iX] , confidence_1[iX]))


train_y_2 = pnm_data
gpr = GP_test()
gpr.fit_data(train_x, train_y_2, gaussian_noise)

pnm, pnm_cov = gpr.predict(test_x)

test_y_2 = pnm.ravel()
confidence_2 = 1.96 * np.sqrt(np.diag(pnm_cov))

print("pnm energy:  %.3f +/- %.3f" % ( pnm[iX], confidence_2[iX]))
print("symmetry energy:  %.3f +/- %.3f" % (pnm[iX]-snm[iX],(confidence_1[iX]+confidence_2[iX])))

t2 = time.time()
print("time for GP : "+ str(t2-t1))
plot_1()
