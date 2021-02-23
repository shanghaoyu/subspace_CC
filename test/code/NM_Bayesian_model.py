import numpy as np 
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import math
import re
#from numpy import polyfit, poly1d
from scipy import interpolate
from validation import io_1
from validation import models as gm
from scipy.special import gamma
from scipy.stats import pearsonr
from scipy.optimize import minimize, fmin
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,WhiteKernel
import pandas as pd


def input_file_1(file_path,raw_data,matter_type):
    count = len(open(file_path,'r').readlines())
    with open(file_path,'r') as f_1:
        data = f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])  #density
                raw_data[loop2][1] = float(temp_1[1])  #ccd
                raw_data[loop2][2] = float(temp_1[2])  #ccd(t)
                if matter_type == "pnm":
                    #raw_data[loop2][3] = (3 * math.pi**2 * raw_data[loop2][0])**(1./3)#pf
                    raw_data[loop2][3] = io_1.density_kf(raw_data[loop2][0],2) 
                elif matter_type == "snm":
                    raw_data[loop2][3] = io_1.density_kf(raw_data[loop2][0],4)  # fm -3  , pnm: g=2
                else:
                    print("matter_type error")

                loop2 = loop2 + 1
            loop1 = loop1 + 1

def input_file_count(file_path):
    count = len(open(file_path,'r').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                loop2 = loop2 + 1
            loop1 = loop1 + 1
       # print ('data_num='+str(loop2))
        return loop2

### test gsum
def test_gsum(kf_series_new, y_ref_new, Q_series_new,y_all_new):
    print("test gm")
   
    # Hyperparameters
    center0 = 0
    disp0 = 0
    df0 = 10
    scale0 = pow((df0-2) /df0 , 0.5)
    n_orders = 3 
    orders = np.arange(0, n_orders)
    
    ls = 0.5
    sd = 1
    center = 0
    ratio = 0.7
    nugget = 1e-10
    seed = 3
    
    kernel = ConstantKernel(1.0, constant_value_bounds='fixed') * RBF(1.0, length_scale_bounds='fixed')
    gp = gm.ConjugateGaussianProcess(kernel=kernel, center=center, df=np.inf, scale=sd, nugget=0)
    
    kernel_fit = RBF(length_scale=ls) + WhiteKernel(noise_level=nugget, noise_level_bounds='fixed')
    
    matter_type = "pnm"
    def reff(X):
        if matter_type == "pnm":
            return 16 * pow((X/1.680),2)
        elif matter_type == "snm":
            return 16 * pow((X/1.333),2)
    
    ref   = y_ref_new
    ratio = Q_series_new
    gp_trunc = gm.TruncationGP(kernel=kernel_fit, ref=ref, ratio=ratio, center=center0, disp=disp0, df=df0, scale=scale0)
    
    x =  kf_series_new 
    X = x[:, None]
    #print(X)
    y = np.array(y_all_new)
    y = y.T
    #print(y.shape)
    
    orders = np.array([0,2,3])
    gp_trunc.fit(X=X, y=y, orders=orders)
    
    ls_ = 0.7
    ratio_val = Q_series_new
    def ratio_val__(X):
        return 0.7*X
    
    #def ratio_val_(X):
    #    return Q_series
    print("########################################")
    print("########################################")
    print("########################################")
    ls_ratio_loglike = gp_trunc.log_marginal_likelihood(theta=[ls_,])
    #ls_vals = np.linspace(1e-3, 0.5, 100)
    #ratio_vals = np.linspace(0.3, 0.7, 80)
    #ls_ratio_loglike = np.array([[gp_trunc.log_marginal_likelihood(theta=[ls_,], ratio=ratio_val) for ls_ in np.log(ls_vals)] for ratio_val in ratio_vals])
    print("They： loglike="+str(ls_ratio_loglike))




def plot_NM(DLO450_data, DNLO450_data, DNNLO450_data,matter_type):
    rn = 1

    x        = DNNLO450_data[:,0]
    x_new    = np.linspace(x.min(),x.max(),1000)
    
    f_smooth = interpolate.splrep(DNNLO450_data[:,0],DNNLO450_data[:,2])
    y1       = interpolate.splev(x_new,f_smooth)
    
    f_smooth = interpolate.splrep(DNNLO450_data[:,0],DNNLO450_data[:,4])
    y1_error = rn * interpolate.splev(x_new,f_smooth)
    
    
    f_smooth = interpolate.splrep(DNLO450_data[:,0],DNLO450_data[:,2])
    y2       = interpolate.splev(x_new,f_smooth)
    f_smooth = interpolate.splrep(DNLO450_data[:,0],DNLO450_data[:,4])
    y2_error = rn * interpolate.splev(x_new,f_smooth)
    
    
    f_smooth = interpolate.splrep(DLO450_data[:,0],DLO450_data[:,2])
    y3       = interpolate.splev(x_new,f_smooth)
    f_smooth = interpolate.splrep(DLO450_data[:,0],DLO450_data[:,4])
    y3_error = rn * interpolate.splev(x_new,f_smooth)
    
    
    fig1  = plt.figure('fig1',figsize= (10,9))
    plt.plot(x_new,y1,color = 'b',linestyle = '-.',linewidth=2, alpha=0.9, label=r'$\Delta$NNLO(450)',zorder=4)
    plt.fill_between(x_new, y1 + y1_error, y1 - y1_error,color='b',alpha=0.3,zorder=1)
    
    plt.plot(x_new,y2,color = 'g',linestyle = '--',linewidth=2, alpha=0.9, label=r'$\Delta$NLO(450)',zorder=5)
    plt.fill_between(x_new, y2 + y2_error, y2 - y2_error,color='g',alpha=0.3,zorder=2)
    
    plt.plot(x_new,y3,color = 'r',linestyle = ':',linewidth=2, alpha=0.9, label=r'$\Delta$LO(450)',zorder=6)
    plt.fill_between(x_new, y3 + y3_error, y3 - y3_error,color='r',alpha=0.3,zorder=3)
    
   
 #   if matter_type == "pnm":
 #       plt.yticks(np.arange(0,60,5),fontsize = 13)
 #       plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
 #       plt.xlim((0.05,0.20))
 #       plt.ylim((0,30))
 #   elif matter_type == "snm":
 #       #plt.yticks(np.arange(-20,5,2),fontsize = 13)
 #       plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
 #       plt.xlim((0.05,0.20))
 #       plt.ylim((-20,-5))
 #   else:
 #       print("matter_type error")

    #plt.yticks(np.arange(0,60,5),fontsize = 13)
    #plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
    #plt.xlim((0.05,0.20))
    #plt.ylim((0,30))
    
    plt.legend(loc='upper left',fontsize = 13)
    plt.ylabel('$E/A$ [MeV]',fontsize=18)
    plt.xlabel(r"$\rho$ [fm$^{-3}$]",fontsize=18)
    
    plot_path = 'EOS_Delta450_%s.pdf' % (matter_type)
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')

### plot EOS with error
def plot_observable_coefficients(train_x,ck,matter_type):
    c0  = ck[0]
    c2  = ck[1]
    c3  = ck[2]

#    X0 = DLO450_data[:,2]
#    #print(X0)
#    for loop1 in range(len(DNNLO450_data)):
#        DLO450_data[loop1,4]   = X0[loop1] * Q[loop1]**2 * max(abs(c0[loop1]),abs(c1[loop1]))
#        #print(max(abs(a0[loop1]),abs(a1[loop1]),abs(a2[loop1])))
#        DNLO450_data[loop1,4]  = X0[loop1] * Q[loop1]**3 * max(abs(c0[loop1]),abs(c1[loop1]),abs(c2[loop1]))
#        DNNLO450_data[loop1,4] = X0[loop1] * Q[loop1]**4 * max(abs(c0[loop1]),abs(c1[loop1]),abs(c2[loop1]),abs(c3[loop1])) 
    
    #print(DLO450_data[:,4])
    #print(DNLO450_data[:,4])
    #print(DNNLO450_data[:,4])
    
    train_x = train_x.reshape(-1,1)
    train_y_1 =  c0 
    train_y_2 =  c2
    train_y_3 =  c3
    test_x  = np.arange(train_x.min(),train_x.max(),0.001).reshape(-1,1)
    
    gpr = io_1.GP_test()
     
    gpr.fit_data(train_x,train_y_1,gaussian_noise=0,sigma=1,length=1)
    a0, a0_cov, d_a0, d_a0_cov,dd_a0,dd_snm_a0 = gpr.predict(test_x)
    
    gpr = io_1.GP_test()
    gpr.fit_data(train_x,train_y_2,gaussian_noise=0,sigma=1,length=1)
    a2, a2_cov, d_a2, d_a2_cov,dd_a2,dd_snm_a2 = gpr.predict(test_x)
    
    gpr = io_1.GP_test()
    gpr.fit_data(train_x,train_y_3,gaussian_noise=0,sigma=1,length=1)
    a3, a3_cov, d_a3, d_a3_cov,dd_a3,dd_snm_a3 = gpr.predict(test_x)
    
    
    # plot
    io_1.plot_12(test_x,a0,a2,a3,a0_cov,a2_cov,a3_cov,train_x,c0, c2, c3,matter_type)



### read EOS data
def read_EOS_data(cutoff,matter_type):
    cutoff = 450
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNNLO%d_%s/EOS_different_density.txt"  %(cutoff, matter_type)
    data_num  = input_file_count(file_path)
    DNNLO_data = np.zeros((data_num,5),dtype = np.float)
    input_file_1(file_path,DNNLO_data,matter_type)
    
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNLO%d_%s/EOS_different_density.txt"  %(cutoff, matter_type)
    data_num  = input_file_count(file_path)
    DNLO_data = np.zeros((data_num,5),dtype = np.float)
    input_file_1(file_path,DNLO_data,matter_type)
    
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DLO%d_%s/EOS_different_density.txt"  %(cutoff, matter_type) 
    data_num  = input_file_count(file_path)
    DLO_data = np.zeros((data_num,5),dtype = np.float)
    input_file_1(file_path,DLO_data,matter_type)
    return DLO_data, DNLO_data, DNNLO_data

def DLO450_snm_data(file_path):
    density  = []
    DLO450   = []
    count = len(open(file_path,'r').readlines())
    with open(file_path,'r') as f_1:
        data = f_1.readlines()
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(count):
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1  = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                density.append(float(temp_1[3]))  #density
                DLO450.append(float(temp_1[7]))  #ccd
    density = [0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.165, 0.17, 0.18, 0.19, 0.2,0.22,0.30,0.38,0.40]
    DLO450   = [-5.46352, -7.28969, -8.48419, -8.8608, -9.19679, -9.49923, -9.77336, -10.02316, -10.25176, -10.35892, -10.46167, -10.65495, -10.8333, -10.9982,-11.29259,-12.13149,-12.66103,-12.76]


    f = np.poly1d(np.polyfit(density,DLO450,9))
    x = np.arange(0.06,0.42,0.02)
    y = f(x)

    y[0] = -6.30153
    y[4] = -9.77336
    y[8] = -11.29259
    y[12]= -12.13149
    y[16]= -12.66103
    #print(density)
    #print(DLO450)

    # check by plot
    fig1     = plt.figure('fig1')
    x_list_1 = density
    y_list_1 = DLO450

    x_list_2 = x
    y_list_2 = y

    #l1 = plt.scatter(x_list_1,y_list_1,s=5, marker = 's')
    l1 = plt.scatter(x_list_1,y_list_1,s=5, marker = 's')
    l2 = plt.plot(x_list_2,y_list_2)

    plot_path = 'test_DLO450_snm.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')
 
    return y

def Q_(kf):
    return kf * hbarc/ lambda_b

def y_ref_(kf,matter_type):
    if matter_type == "pnm":
        kf_0 = 1.680
    if matter_type == "snm":
        kf_0 = 1.333
    return 16 * pow(kf/kf_0,2)

def prepare_training_data(matter_type):
    # read in EOS data 
    DLO450_data, DNLO450_data, DNNLO450_data = read_EOS_data(450,matter_type)
    if matter_type == "snm":
        temp = DLO450_snm_data("/home/slime/subspace_CC/test/DNNLO_old/Andreas_paper_origin_data/LO_nmax4_snm_450.dat")
        DLO450_data[:,2] = temp
    
    plot_NM(DLO450_data, DNLO450_data, DNNLO450_data,matter_type=matter_type)
    
    # set up reference and ratio 
    Q = Q_(kf = DNNLO450_data[:,3])

    density_series = DNNLO450_data[:,0]

    # choice of y_ref
    if y_ref_switch == 0:
        y_ref = y_ref_(kf=DNNLO450_data[:,3],matter_type=matter_type)
    elif y_ref_switch == 1:
        if matter_type == "pnm":
            y_ref = DLO450_data[:,2]
        elif matter_type == "snm":
            y_ref = -1 * DLO450_data[:,2]

    # choice of kf_series
    #kf_series      = DNNLO450_data[:,3] # use own kf 
    kf_series      = io_1.density_kf(density_series,2) # use pnm kf
    print("kf="+str(kf_series)) 

    c0 = (DLO450_data[:,2])/y_ref
    c1 = np.zeros(len(DNNLO450_data)) 
    c2 = (DNLO450_data[:,2] - DLO450_data[:,2])/y_ref / (Q**2)
    c3 = (DNNLO450_data[:,2] - DNLO450_data[:,2])/y_ref / (Q**3)
    
    ck    = np.stack((c0,c2,c3))

    y_0   = y_ref*c0
    y_2   = y_ref*(c0 + c1 * Q  + c2 * pow(Q,2))
    y_3   = y_ref*(c0 + c1 * Q  + c2 * pow(Q,2) + c3*pow(Q,3))
    y_all = np.vstack((y_0,y_2,y_3))

#   check if everything is in line    
#    print(y_0)
#    print(DLO450_data[:,2])
#    print(DNLO450_data[:,2])
#    print(DNNLO450_data[:,2])
#    print("density_sereis="+str(density_series))
#    print("kf_sereis="+str(kf_series))
#    print("y_ref="+str(y_ref))
#    print("Q="+str(Q))
#    print(c0)
#    print(c2)
#    print(c3)
    
    #  use 5 density points for the truncation error study
    y_ref_new     = np.zeros(5)
    kf_series_new = np.zeros(5)
    density_series_new = np.zeros(5)
    ck_new        = np.zeros((len(ck),5))
    y_all_new     = np.zeros((len(ck),5))
    Q_series_new  = np.zeros(5)
    for loop in range(5):
        count = 0.06 + loop * 0.08
        kf_series_new[loop]      = kf_series[0+loop*4]
        density_series_new[loop] = density_series[0+loop*4]
        y_ref_new[loop]          = y_ref[0+loop*4]
        Q_series_new[loop]       = Q[0+loop*4]    
        for loop2 in range(len(ck)):
            ck_new[loop2][loop]   = ck[loop2,0+loop*4]
            y_all_new[loop2][loop]= y_all[loop2,0+loop*4]
          
    print("density_series_new=" + str(density_series_new))
    print("kf_series_new=" + str(kf_series_new))
    print("y_ref_new" + str(y_ref_new))
    print("Q_new ="+str(Q_series_new))
    print("ck_new=" + str(ck_new))
    #print(y_all_new)

    plot_observable_coefficients(density_series_new,ck_new,matter_type)

    return kf_series_new, y_ref_new, Q_series_new, ck_new , y_all_new , ck, y_ref

###############################################
###############################################
###############################################
#####     Main                    #############
###############################################
###############################################
###############################################
lambda_b = 600
hbarc    = 197.326960277
# find the best hyperparameters with training data
gpr = io_1.GP_test()
eta_0        = 0
V_0          = 0
nu_0         = 10
tau_square_0 = (nu_0 - 2)/nu_0 
y_ref_switch = 1


def optimize_hyperparameters(matter_type):
    kf_series_new, y_ref_new, Q_series_new, ck_new, y_all_new ,ck, y_ref= prepare_training_data(matter_type)
    gpr.fit_(eta_0=eta_0, V_0=V_0,nu_0=nu_0 ,tau_square_0=tau_square_0, ck_matrix= ck_new, x_series = kf_series_new, y_ref = y_ref_new, Q_series= Q_series_new)
    # plot the Log-likelihood for l
    #gpr.plot_l_Q()
    
    print("###################################################")
    print("###find the best hyperparameters with training data")
    print("###################################################")
    #loglike_l_Q = gpr.log_likelihood_l_Q(0.7)
    #print("Mine: loglike="+str(loglike_l_Q))
    
    # optimization
    fun_1 = lambda x: -1 * gpr.log_likelihood_l_Q(x[0])
    #res = minimize(gpr.log_likelihood_l_Q, 0.5, bounds=(0,2),method='L-BFGS-B')
    #res = minimize(fun_1, 0.5, bounds=(0,2),method='L-BFGS-B')
    res = fmin(func=fun_1,x0 = 0.5)
    l_best = res[0] 
    print("For "+matter_type+" the best length_scale: "+str(res))
    c_square_best  = gpr.update_c_square_bar(l_best)
    print("For "+matter_type+" the best c_bar(standard deviation): "+str(pow(c_square_best,0.5)))
    return l_best, c_square_best, kf_series_new, y_ref_new, Q_series_new, ck_new, y_all_new, ck, y_ref

l_pnm,variance_pnm,kf_series_pnm, y_ref_pnm, Q_series_pnm, ck_pnm, y_all_pnm ,ck_pnm_raw, y_ref_pnm_raw\
= optimize_hyperparameters(matter_type = "pnm")
l_snm,variance_snm,kf_series_snm, y_ref_snm, Q_series_snm, ck_snm, y_all_snm ,ck_snm_raw, y_ref_snm_raw\
= optimize_hyperparameters(matter_type = "snm")

####################
####test student_t
####################
#gpr.student_t_test_log_like()
#plot_EOS_with_error()
test_gsum(kf_series_snm, y_ref_snm, Q_series_snm,y_all_snm)

#################################
####setup cross covariance matrix
#################################
assert (np.all((kf_series_pnm - kf_series_snm) == 0))

# only for matrix_visualization
#kf_series_pnm  = np.arange(1.2 ,3,0.01)
#density_series = io_1.kf_density(kf_series_pnm,2)
#kf_series_snm  = io_1.density_kf(density_series,4)
#y_ref_pnm      = y_ref_(kf_series_pnm,"pnm") 
#y_ref_snm      = y_ref_(kf_series_snm,"snm") 
#Q_series_pnm   = Q_(kf_series_pnm)
#Q_series_snm   = Q_(kf_series_snm)

# For our calculation
print("#######################################")
print("###### Pb208 project")
print("#######################################")
density_series = np.arange(0.12,0.21,0.02)
#density_series_raw = np.array([0.06, 0.08,0.10, 0.12,0.14,0.16,0.18,0.20,0.22,0.24,0.26,0.28,0,30,0.32,0.34,0.36,0.38,0.40])
#density_series_raw = np.arange(0.06,0.41,0.02)
#density_series_raw = np.array(density_series_raw) 
print("density = "+str(density_series))
#print("y_ref_snm_raw = "+str(y_ref_snm_raw))
#print(density_series[0])
#print(np.where( density_series_raw == 0.12))

kf_series_pnm  = io_1.density_kf(density_series,2)
kf_series_snm  = io_1.density_kf(density_series,4)
print(y_ref_pnm_raw)

if y_ref_switch == 0:
    y_ref_pnm      = y_ref_(kf_series_pnm,"pnm") 
    y_ref_snm      = y_ref_(kf_series_snm,"snm") 
elif y_ref_switch == 1:
    for loop in range(len(density_series)):
        y_ref_pnm[loop]  =  y_ref_pnm_raw[3+loop]
        y_ref_snm[loop]  =  y_ref_snm_raw[3+loop]

print(y_ref_pnm)
Q_series_pnm   = Q_(kf_series_pnm)
Q_series_snm   = Q_(kf_series_snm)

# output ck_raw
print("ck_snm_raw="+str(len(ck_pnm_raw[0])))
c_k_raw_data = np.zeros((len(ck_snm_raw[0]),9))
for loop in range(len(ck_snm_raw[0])):
    c_k_raw_data[loop,0] = 0.06+0.02 * loop
    c_k_raw_data[loop,1] = io_1.density_kf(0.06+0.02 * loop,2)
    c_k_raw_data[loop,2] = io_1.density_kf(0.06+0.02 * loop,4)
    
    c_k_raw_data[loop,3] = ck_pnm_raw[0,loop]
    c_k_raw_data[loop,4] = ck_pnm_raw[1,loop]
    c_k_raw_data[loop,5] = ck_pnm_raw[2,loop]
    c_k_raw_data[loop,6] = ck_snm_raw[0,loop]
    c_k_raw_data[loop,7] = ck_snm_raw[1,loop]
    c_k_raw_data[loop,8] = ck_snm_raw[2,loop]

#print(c_k_raw_data)


colums_name =['density','kf_pnm','kf_snm','c0_pnm','c2_pnm','c3_pnm',\
              'c0_snm', 'c2_snm', 'c3_snm']

df = pd.DataFrame(c_k_raw_data,columns=colums_name)
df.to_pickle('c_k_data.pickle')
#print(df)


# empirical Pearson correlation coefficient
def empirical_Pearson_correlation():
    x      = ck_snm_raw.flatten() 
    y      = ck_pnm_raw.flatten() 
    #x      = np.array([ 1 , 1.01 , 1.02 , 1.03, 1, 2, 3,4] )
    #y      = np.array([ 1 , 0.99 , 0.99,  0.98, 1, 2 , 3, 4])
    density_count_per_order = 18
    x      = x[density_count_per_order*1 : density_count_per_order *3]
    y      = y[density_count_per_order*1 : density_count_per_order *3]
    #print(x)
    #print(y)
    
    x1    = ck_snm_raw.flatten()[density_count_per_order:density_count_per_order *2]
    y1    = ck_pnm_raw.flatten()[density_count_per_order:density_count_per_order *2]
    x2    = ck_snm_raw.flatten()[density_count_per_order *2:density_count_per_order *3]
    y2    = ck_pnm_raw.flatten()[density_count_per_order *2:density_count_per_order *3]
    print(x1)
    print(y1)
    def plot_Pearson_correlation(x1,y1,x2,y2):
        plt.figure(figsize=(5,5))
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
        plt.plot(x1,y1,label = "c2")
        plt.plot(x2,y2,label = "c3")
    
        plt.xlabel("E/A",fontsize=15)
        plt.ylabel("E/N",fontsize=15)
        plt.xlim((-4,4))
        plt.ylim((-2,2))
       
        plot_path = "Pearson_correlation.pdf"
        plt.savefig(plot_path,bbox_inches = 'tight')
        plt.close('all')
    
    plot_Pearson_correlation(x1,y1,x2,y2)
        
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    #a =(x-x_mean)* (y - y_mean)
    #print(np.sum(a))
    
    #x_mean = 0 
    #y_mean = 0
    r_xy   =  np.sum ((x-x_mean) * (y-y_mean))\
    / pow(np.sum((x-x_mean)**2) * np.sum((y-y_mean)**2),0.5)
    print("Pearson correlation coefficient = "+str(r_xy))
    print("ck_snm="+str(ck_snm))
    print(ck_snm.flatten())
    #print(pearsonr(x[0:5],-1*y[0:5]))
    return r_xy

rho_empirical = empirical_Pearson_correlation()
rho_empirical = 1 

cross_cov_matrix   = gpr.setup_cross_cov_matrix(3,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,rho_empirical,1)
print(cross_cov_matrix[0:5,0:5] + cross_cov_matrix[5:10,5:10])
cross_cov_matrix_2 = gpr.setup_cross_cov_matrix(3,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,rho_empirical,2)
print(cross_cov_matrix_2[0:5,0:5] + cross_cov_matrix_2[5:10,5:10])
cross_cov_matrix_3 = gpr.setup_cross_cov_matrix(3,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,rho_empirical,3)
print(cross_cov_matrix_3[0:5,0:5] + cross_cov_matrix_3[5:10,5:10])


# 95% confidence region 
confidence_pnm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_2)[0:5])
confidence_snm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_2)[5:10])
print("95% confidence:")
print("pnm(+/-): "+str(confidence_pnm))
print("snm(+/-): "+str(confidence_snm))

# check kernel invariance (checked)
#kernel_yy = RBF(length_scale=0.51591797)# + WhiteKernel(noise_level=1e-10)
#K_yy     =  2.8127822**2 * kernel_yy(kf_series_snm.reshape(-1,1))
#print(1.96* np.sqrt(y_ref_snm ** 2 * Q_series_snm ** (2 * 4)/(1-Q_series_snm**2)*np.diag(K_yy)))
#print("test for snm diag_kernel="+str(np.diag(K_yy)))
#print("test for snm diag_kernel="+str(K_yy))
#print("test for snm y_ref2="+str(y_ref_snm))
#print("test for snm Q_series2="+str(Q_series_snm))


################################################################
####sample from the GP distribution with cross covariance matrix
################################################################
L = np.linalg.cholesky(cross_cov_matrix_3)
def sample_from_cov_matrix(L):
    standard_normal_distribution_series = np.random.randn(10)
    truncation_error_sample = L @ standard_normal_distribution_series 
    #u,s,v = np.linalg.svd(cross_cov_matrix)
    #print(u.shape)
    #print(pow(s,0.5).shape)
    #print(u @ pow(s,0.5))
    #print(u @ pow(s,0.5)*standard_normal_distribution_series)
    return truncation_error_sample

sample_count = 10000
truncation_error_sample_batch = np.zeros((sample_count,len(L)))

for loop in range(sample_count):
    truncation_error_sample_batch[loop] = sample_from_cov_matrix(L)
    #print("error for pnm: "+str(truncation_error_sample[0:5]))
    #print("error for snm: "+str(truncation_error_sample[5:10]))

### test the mean value and covariance of our samples
#cov_sample = np.cov(truncation_error_sample_batch.T)
#
#print("cov_matrix for the multi GP:")
#print(cross_cov_matrix)
#print("cov_matrix for our samples:")
#print(cov_sample)
#print("mean value:")
#for loop in range(10):
#    print(np.mean(truncation_error_sample_batch[:,loop]))

################################################################
#### read 34 sets of LECs
################################################################
lec_count                = 34
samples_num              = np.zeros((lec_count,sample_count))
saturation_density_batch = np.zeros((lec_count,sample_count))
saturation_energy_batch  = np.zeros((lec_count,sample_count))
symmetry_energy_batch    = np.zeros((lec_count,sample_count))
L_batch                  = np.zeros((lec_count,sample_count))
K_batch                  = np.zeros((lec_count,sample_count))

density_count = 5
validation_count = 34                                                                               
ccdt_pnm_batch_all = np.zeros((validation_count,density_count))
ccdt_snm_batch_all = np.zeros((validation_count,density_count))
density_batch_all = np.zeros((validation_count,density_count))                                      
                                                                                                    
database_dir = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/"
for loop1 in range(density_count):
    dens = 0.12 + loop1 * 0.02
    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt.out" % ('pnm',66,dens)
    ccdt_pnm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/66 
    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt_n3.out" % ('snm',132,dens)
    ccdt_snm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/132 
    density_batch_all[:,loop1] = dens  

set_num = 20
print("pnm ccdt"+str(ccdt_pnm_batch_all[set_num]))
print("snm ccdt"+str(ccdt_snm_batch_all[set_num]))

for loop1 in range(sample_count):
    method_error_pnm = ccdt_pnm_batch_all[set_num] * 0.01 * np.random.randn(5)
    method_error_snm = ccdt_snm_batch_all[set_num] * 0.01 * np.random.randn(5)

    # y_ccdt + EFT truncation error + method error
    pnm_ = ccdt_pnm_batch_all[set_num] + truncation_error_sample_batch[loop1][0:5] \
           + method_error_pnm 
    snm_ = ccdt_snm_batch_all[set_num] + truncation_error_sample_batch[loop1][5:10] \
           + method_error_snm
#    pnm_ = ccdt_pnm_batch_all[set_num]  \
#           + method_error_pnm 
#    snm_ = ccdt_snm_batch_all[set_num]  \
#           + method_error_snm
#    pnm_ = ccdt_pnm_batch_all[set_num] + truncation_error_sample_batch[loop1][0:5]
#    snm_ = ccdt_snm_batch_all[set_num] + truncation_error_sample_batch[loop1][5:10] 

    saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = \
    io_1.generate_NM_observable(pnm_,snm_,density_batch_all[set_num],switch="interpolation")
    saturation_density_batch[set_num,loop1]      = saturation_density
    saturation_energy_batch[set_num,loop1]       = saturation_energy
    symmetry_energy_batch[set_num,loop1]         = symmetry_energy
    L_batch[set_num,loop1]                       = L
    K_batch[set_num,loop1]                       = K

######################################
### save sample results in pickle file
######################################
raw_data = np.vstack((saturation_density_batch[set_num,:],saturation_energy_batch[set_num,:],symmetry_energy_batch[set_num,:],L_batch[set_num,:],K_batch[set_num,:]))


observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K']
df_nm = pd.DataFrame(raw_data.T,columns=observables)
df_nm.to_pickle('NM_ccdt_sampling_from_one_of_34.pickle')

# draw corner plot
io_1.plot_corner_plot(df_nm)


#saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = \
#    io_1.generate_NM_observable(ccdt_pnm_batch_all[set_num],ccdt_snm_batch_all[set_num],density_batch_all[set_num],switch="interpolation")

#io_1.plot_13(saturation_density,saturation_energy,saturation_density_batch[set_num],saturation_energy_batch[set_num])

