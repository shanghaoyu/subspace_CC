import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import re
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
                raw_data[loop2][4] = float(temp_1[3])  #mbpt
                raw_data[loop2][5] = float(temp_1[4])  #HF
                if matter_type == "pnm":
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

    plt.close('all')
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
    #plt.fill_between(x_new, y1 + y1_error, y1 - y1_error,color='b',alpha=0.3,zorder=1)
    
    plt.plot(x_new,y2,color = 'g',linestyle = '--',linewidth=2, alpha=0.9, label=r'$\Delta$NLO(450)',zorder=5)
    #plt.fill_between(x_new, y2 + y2_error, y2 - y2_error,color='g',alpha=0.3,zorder=2)
    
    plt.plot(x_new,y3,color = 'r',linestyle = ':',linewidth=2, alpha=0.9, label=r'$\Delta$LO(450)',zorder=6)
    #plt.fill_between(x_new, y3 + y3_error, y3 - y3_error,color='r',alpha=0.3,zorder=3)
    
   
    if matter_type == "pnm":
        #plt.yticks(np.arange(0,60,5),fontsize = 13)
        plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
        #plt.xlim((0.05,0.20))
        #plt.ylim((0,30))
    elif matter_type == "snm":
        #plt.yticks(np.arange(-20,5,2),fontsize = 13)
        plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
        #plt.xlim((0.05,0.20))
        #plt.ylim((-20,-5))
    else:
        print("matter_type error")

    #plt.yticks(np.arange(0,60,5),fontsize = 13)
    #plt.xticks(np.arange(0.06,0.41,0.02),fontsize = 13)
    #plt.xlim((0.05,0.20))
    #plt.ylim((0,30))
    
    plt.legend(loc='upper left',fontsize = 13)
    plt.ylabel('$E/A$ [MeV]',fontsize=18)
    plt.xlabel(r"$\rho$ [fm$^{-3}$]",fontsize=18)
    
    plot_path = 'EOS_Delta394_%s.pdf' % (matter_type)
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')

### plot EOS with error
def plot_observable_coefficients(train_x,ck,matter_type,observable_type):
    c0  = ck[0]
    c2  = ck[1]
    c3  = ck[2]
    
    train_x = train_x.reshape(-1,1)
    train_y_1 =  c0 
    train_y_2 =  c2
    train_y_3 =  c3
    test_x  = np.arange(train_x.min(),train_x.max(),0.001).reshape(-1,1)
    
    spl_c0  = interpolate.UnivariateSpline(train_x,train_y_1,k=4,s=0) 
    spl_c2  = interpolate.UnivariateSpline(train_x,train_y_2,k=4,s=0) 
    spl_c3  = interpolate.UnivariateSpline(train_x,train_y_3,k=4,s=0) 
   
    a0      = spl_c0(test_x)
    a2      = spl_c2(test_x)
    a3      = spl_c3(test_x)
    a0_cov  = 0
    a2_cov  = 0
    a3_cov  = 0
    #gpr = io_1.GP_test()
    # 
    #gpr.fit_data(train_x,train_y_1,gaussian_noise=0,sigma=1,length=1)
    #a0, a0_cov, d_a0, d_a0_cov,dd_a0,dd_snm_a0 = gpr.predict(test_x)
    #
    #gpr = io_1.GP_test()
    #gpr.fit_data(train_x,train_y_2,gaussian_noise=0,sigma=1,length=1)
    #a2, a2_cov, d_a2, d_a2_cov,dd_a2,dd_snm_a2 = gpr.predict(test_x)
    #
    #gpr = io_1.GP_test()
    #gpr.fit_data(train_x,train_y_3,gaussian_noise=0,sigma=1,length=1)
    #a3, a3_cov, d_a3, d_a3_cov,dd_a3,dd_snm_a3 = gpr.predict(test_x)
    
    # plot
    io_1.plot_12(test_x,a0,a2,a3,a0_cov,a2_cov,a3_cov,train_x,c0, c2, c3,matter_type,observable_type)


### read EOS data
def read_EOS_data(cutoff,matter_type):
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNNLO%d_%s/EOS_different_density_.txt"  %(cutoff, matter_type)
    data_num  = input_file_count(file_path)
    DNNLO_data = np.zeros((data_num,6),dtype = np.float)
    input_file_1(file_path,DNNLO_data,matter_type)
    
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DNLO%d_%s/EOS_different_density_.txt"  %(cutoff, matter_type)
    data_num  = input_file_count(file_path)
    DNLO_data = np.zeros((data_num,6),dtype = np.float)
    input_file_1(file_path,DNLO_data,matter_type)
    
    file_path = "/home/slime/subspace_CC/test/DNNLO_old/DLO%d_%s/EOS_different_density_.txt"  %(cutoff, matter_type) 
    data_num  = input_file_count(file_path)
    DLO_data = np.zeros((data_num,6),dtype = np.float)
    input_file_1(file_path,DLO_data,matter_type)
    return DLO_data, DNLO_data, DNNLO_data

#def DLO450_snm_data(file_path):
#    density  = []
#    DLO450   = []
#    count = len(open(file_path,'r').readlines())
#    with open(file_path,'r') as f_1:
#        data = f_1.readlines()
#        loop1 = 0
#        wtf = re.match('#', 'abc',flags=0)
#        for loop1 in range(count):
#            if ( re.match('#', data[loop1],flags=0) == wtf):
#                temp_1  = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
#                density.append(float(temp_1[3]))  #density
#                DLO450.append(float(temp_1[7]))  #ccd
#    density = [0.05, 0.075, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.165, 0.17, 0.18, 0.19, 0.2,0.22,0.30,0.38,0.40]
#    DLO450   = [-5.46352, -7.28969, -8.48419, -8.8608, -9.19679, -9.49923, -9.77336, -10.02316, -10.25176, -10.35892, -10.46167, -10.65495, -10.8333, -10.9982,-11.29259,-12.13149,-12.66103,-12.76]
#
#
#    f = np.poly1d(np.polyfit(density,DLO450,9))
#    x = np.arange(0.06,0.42,0.02)
#    y = f(x)
#
#    y[0] = -6.30153
#    y[4] = -9.77336
#    y[8] = -11.29259
#    y[12]= -12.13149
#    y[16]= -12.
#    #print(density)
#    #print(DLO450)
#
#    # check by plot
#    fig1     = plt.figure('fig1')
#    x_list_1 = density
#    y_list_1 = DLO450
#
#    x_list_2 = x
#    y_list_2 = y
#
#    #l1 = plt.scatter(x_list_1,y_list_1,s=5, marker = 's')
#    l1 = plt.scatter(x_list_1,y_list_1,s=5, marker = 's')
#    l2 = plt.plot(x_list_2,y_list_2)
#
#    plot_path = 'test_DLO450_snm.pdf'
#    plt.savefig(plot_path,bbox_inches='tight')
#    plt.close('all')
#    return y

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
    DLO450_data, DNLO450_data, DNNLO450_data = read_EOS_data(394,matter_type)
    #if matter_type == "snm":
    #    temp = DLO450_snm_data("/home/slime/subspace_CC/test/DNNLO_old/Andreas_paper_origin_data/LO_nmax4_snm_450.dat")
    #    DLO450_data[:,2] = temp
    
    plot_NM(DLO450_data, DNLO450_data, DNNLO450_data,matter_type=matter_type)
    
    density_series = DNNLO450_data[:,0]
    # set up reference and ratio 
    Q = Q_(kf = DNNLO450_data[:,3])

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

    #####################
    ### For method error 
    #####################
    global ccdt_pnm_correlation_energy, ccdt_snm_correlation_energy
    if matter_type == "pnm":
        ccdt_pnm_correlation_energy = (DNNLO450_data[:,2] - DNNLO450_data[:,5])
    elif matter_type == "snm":
        ccdt_snm_correlation_energy = (DNNLO450_data[:,2] - DNNLO450_data[:,5])

   # method error / ccdt correction energy
    #m0 = (DLO450_data[:,4]   - DLO450_data[:,2])   / (DLO450_data[:,2]   - DLO450_data[:,5])  
    #m1 = (DNLO450_data[:,4]  - DNLO450_data[:,2])  / (DNLO450_data[:,2]  - DNLO450_data[:,5])
    #m2 = (DNNLO450_data[:,4] - DNNLO450_data[:,2]) / (DNNLO450_data[:,2] - DNNLO450_data[:,5])

    # (ccdt -ccd)  / (ccdt - HF)
    m0 = (DLO450_data[:,2]   - DLO450_data[:,1])   / (DLO450_data[:,2]   - DLO450_data[:,5])  
    m1 = (DNLO450_data[:,2]  - DNLO450_data[:,1])  / (DNLO450_data[:,2]  - DNLO450_data[:,5])
    m2 = (DNNLO450_data[:,2] - DNNLO450_data[:,1]) / (DNNLO450_data[:,2] - DNNLO450_data[:,5])

   # # (mbpt -ccd)  / (ccd - HF)
   # m0 = (DLO450_data[:,4]   - DLO450_data[:,1])   / (DLO450_data[:,1]   - DLO450_data[:,5])  
   # m1 = (DNLO450_data[:,4]  - DNLO450_data[:,1])  / (DNLO450_data[:,1]  - DNLO450_data[:,5])
   # m2 = (DNNLO450_data[:,4] - DNNLO450_data[:,1]) / (DNNLO450_data[:,1] - DNNLO450_data[:,5])
   # method error / ccdt correction energy
   # m0 = (DLO450_data[:,4]   - DLO450_data[:,2])   / (DLO450_data[:,2] )  
   # m1 = (DNLO450_data[:,4]  - DNLO450_data[:,2])  / (DNLO450_data[:,2])
   # m2 = (DNNLO450_data[:,4] - DNNLO450_data[:,2]) / (DNNLO450_data[:,2] )



    mk    = np.stack((m0,m1,m2)) 


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
    
    #  use n density points for the truncation error study
    n = 5
    gap = 0.08
    y_ref_new     = np.zeros(n)
    kf_series_new = np.zeros(n)
    density_series_new = np.zeros(n)
    ck_new        = np.zeros((len(ck),n))
    y_all_new     = np.zeros((len(ck),n))
    Q_series_new  = np.zeros(n)
    mk_new        = np.zeros((len(ck),n))# for method error

    for loop in range(n):
        count = int(gap / 0.02)
        kf_series_new[loop]      = kf_series[0+loop*count]
        density_series_new[loop] = density_series[0+loop*count]
        y_ref_new[loop]          = y_ref[0+loop*count]
        Q_series_new[loop]       = Q[0+loop*count]    

        for loop2 in range(len(ck)):
            ck_new[loop2][loop]   = ck[loop2,0+loop*count]
            mk_new[loop2][loop]   = mk[loop2,0+loop*count]
            y_all_new[loop2][loop]= y_all[loop2,0+loop*count]
          
    print("density_series_new=" + str(density_series_new))
    print("kf_series_new=" + str(kf_series_new))
    print("y_ref_new" + str(y_ref_new))
    print("Q_new ="+str(Q_series_new))
    print("ck_new=" + str(ck_new))
    print("mk_new=" + str(mk_new))
    #print(y_all_new)



    plot_observable_coefficients(density_series_new,ck_new,matter_type,'c')

    #print("wtf:"+str(DNNLO450_data))

    plot_observable_coefficients(density_series_new,mk_new,matter_type,'m')


    mk_scal = 50 
    #mk    = np.stack((m0,m1,m2))  *  mk_scal
    mk     = mk * mk_scal
    mk_new = mk_new * mk_scal

    return kf_series_new, y_ref_new, Q_series_new, ck_new , y_all_new , ck, y_ref,density_series, mk,mk_new




###############################################
###############################################
###############################################
#####     Main                    #############
###############################################
###############################################
###############################################
lambda_b = 600
hbarc    = 197.326960277

####################################################
### find the best hyperparameters with training data
####################################################
gpr = io_1.GP_test()
eta_0        = 0
V_0          = 0
nu_0         = 10
tau_square_0 = (nu_0 - 2)/nu_0 
y_ref_switch = 1   # 1: y_ref = LO , 0: y_ref = f(kf)


def optimize_hyperparameters(matter_type):
    kf_series_new, y_ref_new, Q_series_new, ck_new, y_all_new ,ck, y_ref,density_series,mk,mk_new = prepare_training_data(matter_type)
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

    print("###################################################")
    print("###find the best hyperparameters with method error")
    print("###################################################")
    print(mk_new)
    mk_new_ = mk_new[1:,:]
    gpr.fit_(eta_0=eta_0, V_0=V_0,nu_0=nu_0 ,tau_square_0=tau_square_0, ck_matrix= mk_new_, x_series = kf_series_new, y_ref = np.ones(mk_new.shape), Q_series= np.ones(len(kf_series_new)))

    # optimization
    fun_2 = lambda x: -1 * gpr.log_likelihood_l_Q_method_error(x[0])
    res = fmin(func=fun_2,x0 = 0.5)
    l_mk_best = res[0] 
    print("For "+matter_type+" the best method error length_scale: "+str(res))
    mk_scal = 50
    m_square_best  = gpr.update_c_square_bar(l_mk_best) / mk_scal/mk_scal
    print("For "+matter_type+" the best m_bar(standard deviation): "+str(pow(m_square_best,0.5)))



    return l_best, c_square_best, kf_series_new, y_ref_new, Q_series_new, ck_new, y_all_new, ck, y_ref, density_series , mk,mk_new, l_mk_best, m_square_best



ccdt_pnm_correlation_energy = np.zeros(18)
ccdt_snm_correlation_energy = np.zeros(18)

l_pnm,variance_pnm,kf_series_pnm, y_ref_pnm, Q_series_pnm, ck_pnm, y_all_pnm ,ck_pnm_raw, y_ref_pnm_raw, density_series_raw, mk_pnm_raw,mk_pnm, l_mk_pnm, variance_mk_pnm \
= optimize_hyperparameters(matter_type = "pnm")
l_snm,variance_snm,kf_series_snm, y_ref_snm, Q_series_snm, ck_snm, y_all_snm ,ck_snm_raw, y_ref_snm_raw, density_series_raw, mk_snm_raw,mk_snm, l_mk_snm, variance_mk_snm\
= optimize_hyperparameters(matter_type = "snm")

########################################################
####test gsum  (note that ".clone_with_theta" is wrong )
########################################################
#gpr.student_t_test_log_like()
#plot_EOS_with_error()
test_gsum(kf_series_snm, y_ref_snm, Q_series_snm,y_all_snm)
#test_gsum(kf_series_snm, np.ones(y_ref_snm.shape), np.ones(Q_series_snm.shape), mk_snm)

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
#temp!!! 
#density_series = np.arange(0.1382,0.2283,0.02)
#density_series = np.arange(0.16,0.241,0.02)
density_count  = len(density_series)
print("density = "+str(density_series))

def y_ref_2(y_ref_raw,density_raw,density_new):
    spl = interpolate.UnivariateSpline(density_raw,y_ref_raw,k=4,s=0) 
    y_ref_2 = spl(density_new)
    return y_ref_2


kf_series_pnm  = io_1.density_kf(density_series,2)
kf_series_snm  = io_1.density_kf(density_series,4)

if y_ref_switch == 0:
    y_ref_pnm      = y_ref_(kf_series_pnm,"pnm") 
    y_ref_snm      = y_ref_(kf_series_snm,"snm") 
elif y_ref_switch == 1:
    
    y_ref_pnm = y_ref_2(y_ref_pnm_raw, density_series_raw,density_series)
    y_ref_snm = y_ref_2(y_ref_snm_raw, density_series_raw,density_series)
#    for loop in range(len(density_series)):
#        y_ref_pnm[loop]  =  y_ref_pnm_raw[3+loop]
#        y_ref_snm[loop]  =  y_ref_snm_raw[3+loop]

print(y_ref_pnm)

Q_series_pnm   = Q_(kf_series_pnm)
Q_series_snm   = Q_(kf_series_snm)


#columns_name =['density','kf_pnm','kf_snm','c0_pnm','c2_pnm','c3_pnm',\
#              'c0_snm', 'c2_snm', 'c3_snm']
#
#c_k_raw_data = np.zeros((len(ck_snm_raw[0]),len(columns_name)))
#
#for loop in range(len(ck_snm_raw[0])):
#    c_k_raw_data[loop,0] = 0.06+0.02 * loop
#    c_k_raw_data[loop,1] = io_1.density_kf(0.06+0.02 * loop,2)
#    c_k_raw_data[loop,2] = io_1.density_kf(0.06+0.02 * loop,4)
#    
#    c_k_raw_data[loop,3] = ck_pnm_raw[0,loop]
#    c_k_raw_data[loop,4] = ck_pnm_raw[1,loop]
#    c_k_raw_data[loop,5] = ck_pnm_raw[2,loop]
#    c_k_raw_data[loop,6] = ck_snm_raw[0,loop]
#    c_k_raw_data[loop,7] = ck_snm_raw[1,loop]
#    c_k_raw_data[loop,8] = ck_snm_raw[2,loop]
#
#df = pd.DataFrame(c_k_raw_data,columns=columns_name)
#df.to_pickle('c_k_data.pickle')
#print(df)

#################################################
# empirical Pearson correlation coefficient
#################################################
def empirical_Pearson_correlation(x, y, observable_type):
    density_count_per_order = 18
    x      = x[density_count_per_order*1 : density_count_per_order *3]
    y      = y[density_count_per_order*1 : density_count_per_order *3]
    #print(x)
    #print(y)
    
    x1    = x[density_count_per_order *0 :density_count_per_order *1]
    y1    = y[density_count_per_order *0 :density_count_per_order *1]
    x2    = x[density_count_per_order *1 :density_count_per_order *2]
    y2    = y[density_count_per_order *1 :density_count_per_order *2]
    print(x1)
    print(y1)
    def plot_Pearson_correlation(x1,y1,x2,y2):
        plt.figure(figsize=(5,5))
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
        plt.plot(x1,y1,label = "%s2"%(observable_type))
        plt.plot(x2,y2,label = "%s3"%(observable_type))
    
        plt.xlabel("E/A",fontsize=15)
        plt.ylabel("E/N",fontsize=15)
        #plt.xlim((-4,4))
        #plt.ylim((-2,2))
        plt.legend() 
        plot_path = "Pearson_correlation_%s.pdf" %(observable_type)
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
   #print("ck_snm="+str(ck_snm))
   #print(ck_snm.flatten())
    #print(pearsonr(x[0:5],-1*y[0:5]))
    return r_xy
mk_scale = 50
rho_empirical    = empirical_Pearson_correlation(ck_snm_raw.flatten(),ck_pnm_raw.flatten(),"c")
rho_empirical_mk = empirical_Pearson_correlation(mk_snm_raw.flatten()/50,mk_pnm_raw.flatten()/50,"m")
print("mk_snm"+str(mk_snm_raw.flatten()))


#################################
####setup cross covariance matrix
#################################
truncation_order = 3
cov_method_switch = 1 
# 1 and 2 as in Drischler's paper: PHYSICAL REVIEW C 102, 054315 (2020),||| 1:C16,17 ||| 2:C18 ||| 
# 3:l1 and l2 for diagonal component, (l1+l2)/2 for off-diagonal component, c_bar_square = c_snm * c_pnm * rho(empirical correlation coefficient)
rho = [rho_empirical,0,0.5]

###################################################
# show results for three different choice of rho
###################################################
cross_cov_matrix_1 = gpr.setup_cross_cov_matrix(truncation_order,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,rho_empirical,cov_method_switch)
#print(cross_cov_matrix_1[0:5,0:5] + cross_cov_matrix_1[5:10,5:10])
np.savetxt("cov_matrix_snm.txt",cross_cov_matrix_1[density_count::,density_count::],fmt='%.4f')

cross_cov_matrix_2 = gpr.setup_cross_cov_matrix(truncation_order,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,0,cov_method_switch)
#print(cross_cov_matrix_2[0:5,0:5] + cross_cov_matrix_2[5:10,5:10])

cross_cov_matrix_3 = gpr.setup_cross_cov_matrix(truncation_order,variance_pnm,l_pnm,variance_snm,l_snm,kf_series_pnm,y_ref_pnm,y_ref_snm,Q_series_pnm,Q_series_snm,0.5,cov_method_switch)
#print(cross_cov_matrix_3[0:5,0:5] + cross_cov_matrix_3[5:10,5:10])

###############################
### for correlated method error
###############################
#sigma1_mk   = pow(variance_mk_pnm,0.5)
#sigma2_mk   = pow(variance_mk_snm,0.5)
sigma1_mk   = 0.05
sigma2_mk   = 0.05

kernel_xx = RBF(length_scale=(l_mk_pnm)) + WhiteKernel(noise_level=1e-10)
K_xx      = sigma1_mk**2 * kernel_xx(kf_series_pnm.reshape(-1,1))
kernel_yy = RBF(length_scale=(l_mk_snm)) + WhiteKernel(noise_level=1e-10)
K_yy      = sigma2_mk**2 * kernel_yy(kf_series_pnm.reshape(-1,1))
kernel_xy = RBF(length_scale=(pow((l_mk_pnm**2+l_mk_snm**2)/2,0.5))) + WhiteKernel(noise_level=1e-10)
print("empirical mk rho:"+str(rho_empirical_mk))
print("our mk rho:"+str(pow((2*l_mk_pnm*l_mk_pnm/(l_mk_pnm**2+l_mk_snm**2)),0.5)))
K_xy      = sigma1_mk  * sigma2_mk * rho_empirical_mk * kernel_xy(kf_series_pnm.reshape(-1,1))
cross_cov_matrix_0 = np.hstack((K_xx,K_xy))
temp_              = np.hstack((K_xy.T,K_yy))
cross_cov_matrix_0 = np.vstack((cross_cov_matrix_0,temp_))
print(cross_cov_matrix_0.shape)

temp_length = round(len(cross_cov_matrix_1)/2)
# 95% confidence region 
confidence_pnm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_1)[0:temp_length])
confidence_snm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_1)[temp_length:temp_length*2])
print("95% confidence for truncation error:")
print("pnm(+/-): "+str(confidence_pnm))
print("snm(+/-): "+str(confidence_snm))
print("standard deviation:")
print("pnm: "+str(np.sqrt(np.diag(cross_cov_matrix_1)[0:temp_length])))
print("snm: "+str(np.sqrt(np.diag(cross_cov_matrix_1)[temp_length:temp_length*2])))
# 95% confidence region for method error 
temp_length = round(len(cross_cov_matrix_0)/2)
confidence_pnm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_0)[0:temp_length])
confidence_snm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_0)[temp_length:temp_length*2])
print("95% confidence for method error:")
print("pnm(+/-): "+str(confidence_pnm))
print("snm(+/-): "+str(confidence_snm))
print("standard deviation:")
print("pnm: "+str(np.sqrt(np.diag(cross_cov_matrix_0)[0:temp_length])))
print("snm: "+str(np.sqrt(np.diag(cross_cov_matrix_0)[temp_length:temp_length*2])))


#confidence_pnm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_0)[0:5])
#confidence_snm = 1.96 * np.sqrt(np.diag(cross_cov_matrix_0)[5:10])
#print("95% confidence for method error:")
#print("pnm(+/-): "+str(confidence_pnm))
#print("snm(+/-): "+str(confidence_snm))
  
# check kernel invariance (checked)
#kernel_yy = RBF(length_scale=0.51591797)# + WhiteKernel(noise_level=1e-10)
#K_yy     =  2.8127822**2 * kernel_yy(kf_series_snm.reshape(-1,1))
#print(1.96* np.sqrt(y_ref_snm ** 2 * Q_series_snm ** (2 * 4)/(1-Q_series_snm**2)*np.diag(K_yy)))
#print("test for snm diag_kernel="+str(np.diag(K_yy)))
#print("test for snm diag_kernel="+str(K_yy))
#print("test for snm y_ref2="+str(y_ref_snm))
#print("test for snm Q_series2="+str(Q_series_snm))

################################################################
#### read 34 sets of LECs
################################################################
validation_count = 34
density_count    = len(density_series)
sample_count     = 1 
database_dir = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/"

samples_num              = np.zeros((validation_count,sample_count))
saturation_density_batch = np.zeros((validation_count,sample_count))
saturation_energy_batch  = np.zeros((validation_count,sample_count))
symmetry_energy_batch    = np.zeros((validation_count,sample_count))
L_batch                  = np.zeros((validation_count,sample_count))
K_batch                  = np.zeros((validation_count,sample_count))
rho_batch                = np.zeros((validation_count,sample_count))
set_num_                 = np.zeros((validation_count,sample_count))
likelihood_              = np.zeros((validation_count,sample_count))
pnm_with_error           = []
snm_with_error           = []
density_                 = []

ccdt_pnm_batch_all = np.zeros((validation_count,density_count))
ccdt_snm_batch_all = np.zeros((validation_count,density_count))
density_batch_all  = np.zeros((validation_count,density_count))           
ccdt_pnm_correlation_energy_new = np.zeros(density_count) 
ccdt_snm_correlation_energy_new = np.zeros(density_count)

for loop1 in range(density_count):
    #temp!!!
    #dens = 0.12 + loop1 * 0.02
    dens = density_series[loop1]
    ccdt_pnm_correlation_energy_new[loop1] = ccdt_pnm_correlation_energy[loop1+3] 
    ccdt_snm_correlation_energy_new[loop1] = ccdt_snm_correlation_energy[loop1+3]
    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt.out" % ('pnm',66,dens)
    ccdt_pnm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/66 
    input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt_n3.out" % ('snm',132,dens)
    ccdt_snm_batch_all[:,loop1] = io_1.read_ccd_data(input_dir = input_dir, data_count = validation_count )/132 
    density_batch_all[:,loop1] = dens  

print("ccdt_pnm_correlation_energy_new")
print(ccdt_pnm_correlation_energy_new)
print("ccdt_snm_correlation_energy_new")
print(ccdt_snm_correlation_energy_new)

#print(ccdt_snm_batch_all[20])
#spl  = interpolate.UnivariateSpline(np.arange(0.06,0.201,0.02),ccdt_snm_batch_all[20],k=4,s=0)
#print(spl(np.arange(0.06,0.161,0.01)))
#os._exit(0)
################################################################
####sample from the GP distribution with cross covariance matrix
################################################################
def sample_obs_from_cross_cov_matrix(cross_cov_matrix, rho, set_num,sample_count,method_error_flag = True, method_error_switch= 0 ):  
# cross_cov_matrix: covariance matrix between pnm and snm 
# rho: the correlation coefficient of the off-diagonal matrix between pnm and snm
# set_num: determine which set of LECs are used to generate the EOS (without error) 

    L    = np.linalg.cholesky(cross_cov_matrix)
    L_mk = np.linalg.cholesky(cross_cov_matrix_0)
    def sample_from_cov_matrix(L):
        standard_normal_distribution_series = np.random.randn(10)
        truncation_error_sample = L @ standard_normal_distribution_series 
        return truncation_error_sample
    
    truncation_error_sample_batch = np.zeros((sample_count,len(L)))
    method_error_sample_batch = np.zeros((sample_count,len(L_mk)))



    
    for loop in range(sample_count):
        truncation_error_sample_batch[loop] = sample_from_cov_matrix(L)
        method_error_sample_batch[loop]     = sample_from_cov_matrix(L_mk)

    ##################################################### 
    ### test the mean value and covariance of our samples
    ##################################################### 
    #cov_sample = np.cov(truncation_error_sample_batch.T)
    #
    #print("cov_matrix for the multi GP:")
    #print(cross_cov_matrix)
    #print("cov_matrix for our samples:")
    #print(cov_sample)
    #print("mean value:")
    #for loop in range(10):
    #    print(np.mean(truncation_error_sample_batch[:,loop]))
    pnm_batch = []    
    snm_batch = []
    def generate_observable_data(truncation_error_sample_batch,rho):
        for loop1 in range(sample_count):
        
            # y_ccdt + EFT truncation error + method error
            if(method_error_flag == True):
                if method_error_switch == 0:
                    method_error_pnm = ccdt_pnm_batch_all[set_num] * method_error_sample_batch[loop1][0:5]
                    method_error_snm = ccdt_snm_batch_all[set_num] * method_error_sample_batch[loop1][5:10]
                elif method_error_switch == 1:
                    # correlated method error
                    method_error_pnm = ccdt_pnm_correlation_energy_new * method_error_sample_batch[loop1][0:5]
                    method_error_snm = ccdt_snm_correlation_energy_new * method_error_sample_batch[loop1][5:10]
                    #print("method_error_pnm"+str(method_error_pnm))                  
                    #print("method_error_snm"+str(method_error_snm))                  

                pnm_ = ccdt_pnm_batch_all[set_num] + truncation_error_sample_batch[loop1][0:5] \
                       + method_error_pnm 
                snm_ = ccdt_snm_batch_all[set_num] + truncation_error_sample_batch[loop1][5:10] \
                       + method_error_snm
            elif(method_error_flag == False):
                pnm_ = ccdt_pnm_batch_all[set_num] + truncation_error_sample_batch[loop1][0:5]
                snm_ = ccdt_snm_batch_all[set_num] + truncation_error_sample_batch[loop1][5:10] 
            pnm_batch.append(pnm_) 
            snm_batch.append(snm_) 
        #    pnm_ = ccdt_pnm_batch_all[set_num]  \
        #           + method_error_pnm 
        #    snm_ = ccdt_snm_batch_all[set_num]  \
        #           + method_error_snm
       
            saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = \
            io_1.generate_NM_observable(pnm_,snm_,density_batch_all[set_num],switch="interpolation")
            saturation_density_batch[set_num,loop1]      = saturation_density
            saturation_energy_batch[set_num,loop1]       = saturation_energy
            symmetry_energy_batch[set_num,loop1]         = symmetry_energy
            L_batch[set_num,loop1]                       = L
            K_batch[set_num,loop1]                       = K
            rho_batch[set_num,loop1]                     = round(rho,2)
            set_num_[set_num,loop1]                      = set_num
            
            density_.append(raw_data[3])
            pnm_with_error.append(raw_data[4])
            snm_with_error.append(raw_data[10])
#            likelihood_[set_num,loop1]                   = likelihood

            raw_data = np.vstack((saturation_density_batch[set_num,:],saturation_energy_batch[set_num,:]\
           ,symmetry_energy_batch[set_num,:],L_batch[set_num,:],K_batch[set_num,:],rho_batch[set_num,:],set_num_[set_num,:]))


        return raw_data.T
     
    # for different rho 
    #for loop2 in range(3):
    raw_data = generate_observable_data(truncation_error_sample_batch,rho)

####################################
#### plot 68% EOS for an interaction
####################################
#    pnm_batch = np.array(pnm_batch)
#    snm_batch = np.array(snm_batch)
#    io_1.plot_confidence_EOS(density_batch_all[0,:],ccdt_pnm_batch_all[set_num],ccdt_snm_batch_all[set_num],pnm_batch,snm_batch,q=0.68)
    #snm_batch = np.array(snm_batch)
    #print(ccdt_snm_batch_all[set_num])
    #print(snm_batch[:,1])
    #print(np.mean(snm_batch[:,1]))
    #print(1.96* np.sqrt(np.var(snm_batch[:,1])))
 

    return raw_data 

#############################
### test for one set of LEC
#############################
#set_num =20 
#sample_count =1000
#raw_data_1 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_1, rho[0], set_num,sample_count,method_error_flag = False)
#
#os._exit(0)
#
#raw_data_2 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_2, rho[1], set_num,sample_count,method_error_flag = False)
#raw_data_3 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_3, rho[2], set_num,sample_count,method_error_flag = False)
#raw_data = np.vstack((raw_data_1,raw_data_2,raw_data_3))
#######################################
#### save sample results in pickle file
#######################################
#observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K','rho','set_num']
#df_nm = pd.DataFrame(raw_data,columns=observables)
#df_nm.to_pickle('NM_ccdt_sampling_from_one_of_34.pickle')
#print(df_nm)
#########################
## draw corner plot
#########################
#df_nm= df_nm.loc[(df_nm['rho']==0)]
##io_1.plot_corner_plot(df_nm.loc[:,['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K','rho']])
##########################################
#### print 5 set of EOS with error
##########################################
##raw_data_temp = sample_obs_from_cross_cov_matrix(cross_cov_matrix_1, rho[0], 20 ,sample_count,method_error_flag = True,method_error_switch = 1)
##io_1.plot_14(density_batch_all[20],ccdt_pnm_batch_all[20],ccdt_snm_batch_all[20],density_,pnm_with_error,snm_with_error)                                
#os._exit(0)


##########################################################
### sample for 34 sets of interactions with likelihood
##########################################################
# read likelihood pickle file
sample_count = 10000
likelihood_switch = 3 
method_error_flag = True # decide whether to include method error or not
method_error_switch = 1

df_likelihood = pd.read_pickle('NIsamples_with_likelihood_A48_208.pickle') 

print(df_likelihood.columns)
print(df_likelihood)

#os._exit(0)

df_likeli     = df_likelihood.values
normal_likeli = df_likeli[:,7+likelihood_switch] / sum(df_likeli[:,7+likelihood_switch])

if likelihood_switch == 0:
    normal_likeli =  1./34 * np.ones(34)
#print(np.where(normal_likeli== np.max(normal_likeli)))

# pick 'sample_count' sets of LEC accroding to their likelihood
set_num_batch = np.random.choice(validation_count, sample_count, p=normal_likeli)
for loop in range(sample_count):
    if loop == 0:
        raw_data_1 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_1, rho[0], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data_2 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_2, rho[1], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data_3 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_3, rho[2], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data = np.vstack((raw_data_1,raw_data_2,raw_data_3))
        #raw_data = raw_data_1 
    elif loop > 0:
        raw_data_1 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_1, rho[0], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data_2 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_2, rho[1], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data_3 = sample_obs_from_cross_cov_matrix(cross_cov_matrix_3, rho[2], set_num_batch[loop],1,method_error_flag = method_error_flag, method_error_switch =method_error_switch)
        raw_data = np.vstack((raw_data,raw_data_1,raw_data_2,raw_data_3))
        #raw_data = np.vstack((raw_data,raw_data_1))
    else:
        print("Looping error!")

print(raw_data)
#####################################
## save sample results in pickle file
#####################################
observables = ['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K','rho','set_num']
raw_data = raw_data[np.where(raw_data[:,5]==0.9)]
df_nm = pd.DataFrame(raw_data,columns=observables)
df_nm.to_pickle('NM_ccdt_sampling_from_34_interaction_likelihood_%d.pickle' % (likelihood_switch))
#pd.options.display.float_format = '{:.2f}'.format
print(df_nm)
#######################
# draw corner plot
#######################
io_1.plot_corner_plot(df_nm.loc[:,['saturation_density', 'saturation_energy','symmetry_energy', 'L', 'K','rho','set_num']])
#io_1.plot_corner_plot(df_nm.loc[:,['saturation_density', 'saturation_energy','symmetry_energy','rho','set_num']])

#######################
# 2D credible interval
#######################
df_  = df_nm.dropna(axis=0,how='any')
df_  = df_.loc[(df_['rho']==np.round(rho_empirical,2)) ]
df_2 = df_.loc[:,'saturation_energy']
print("df_=" +str(df_2.to_numpy()))
print("df_=" +str(df_['saturation_density'].to_numpy()))
io_1.plot_confidence_ellipse(df_['saturation_density'].to_numpy(),df_['saturation_energy'].to_numpy())

########################################
### quantile 68% 90% for each observable
########################################
def quantile_(df,q,rho):
    significant_digits = 3


    df_BCI = pd.DataFrame(columns=df.columns[0:5])
#    print(df_BCI)
#    print('############################################################')
#    print("%.0f%% Bayesian credible interval, rho = %.2f " % (q*100,rho))
    df_ = df.dropna(axis=0,how='any')
    df_ = df_.loc[(df_['rho']==rho) ]

    header = ''
    for row_obs in df_BCI.columns:
        header += f' {row_obs:>10}    '
    header += 'Likelihood, Correlation'
    print(header+'\n '+'-'*112)
    
    new = [io_1.hdi(df_[col_obs], hdi_prob=q) for icol, col_obs in enumerate(df_.columns[0:5]) ]
    #df_BCI=df_BCI.append(new,ignore_index=True)
    df_BCI.loc[0] = new
    #pd.options.display.float_format = '{:.2f}'.format
    #print(df_BCI)

    rowstr = ''
    for icol, col_obs in enumerate(df_.columns[0:5]): 
        (x_lo,x_hi) = io_1.hdi(df_[col_obs], hdi_prob=q)
        # Round to three significant digits
        x_lo_round = round(x_lo, significant_digits - int(math.floor(math.log10(abs(x_lo)))) - 1)
        x_hi_round = round(x_hi, significant_digits - int(math.floor(math.log10(abs(x_hi)))) - 1)
        #if row_obs in ['EA_NM','S_NM','L_NM']: 
        if col_obs in ['saturation_energy','symmetry_energy','L']: 
            rowstr += f' [{x_lo_round:5.1f}, {x_hi_round:5.1f}]'
        #elif row_obs in ['rho_NM','Rskin208Pb']: 
        elif col_obs in ['saturation_density','Rskin208Pb']: 
            rowstr += f' [{x_lo_round:5.3f}, {x_hi_round:5.3f}]'
        elif col_obs in ['K']: 
            rowstr += f' [{x_lo_round:3.0f}, {x_hi_round:3.0f}]'
    #rowstr += f' {likelihood_spec[0]:>13}, {likelihood_spec[1]}'
    #rowster + = f'df_likeli.columns[:,7+likelihood_switch]'
    print(rowstr)

    #x_lo_round = round(x_lo, significant_digits - int(math.floor(math.log10(abs(x_lo)))
    #x_hi_round = round(x_hi, significant_digits - int(math.floor(math.log10(abs(x_hi)))

    print('\n')


#quantile_(df_nm,0.68,0)
#quantile_(df_nm,0.9,0)
#quantile_(df_nm,0.68,0.5)
#quantile_(df_nm,0.9,0.5)
quantile_(df_nm,0.68,np.round(rho_empirical,2))
#quantile_(df_nm,0.9,np.round(rho_empirical,2))

significant_digits = 3
#header = ''
#for row_obs in df_plot.columns:
#    header += f' {row_obs:>10}    '
#header += 'Likelihood, Correlation'
#print(header+'\n '+'-'*112)
#for likelihood_spec in likelihood_specs:
#    rowstr = ''
#    for row_obs in df_plot.columns:
#        cred_list = d_cred[(row_obs,likelihood_spec)]
#        x_med = cred_list[0]
#        (x_lo,x_hi) = cred_list[1]
#        # Round to three significant digits
#        x_lo_round = round(x_lo, significant_digits - int(math.floor(math.log10(abs(x_lo)))) - 1)
#        x_hi_round = round(x_hi, significant_digits - int(math.floor(math.log10(abs(x_hi)))) - 1)
#        if row_obs in ['EA_NM','S_NM','L_NM']: 
#            rowstr += f' [{x_lo_round:5.1f}, {x_hi_round:5.1f}]'
#        elif row_obs in ['rho_NM','Rskin208Pb']: 
#            rowstr += f' [{x_lo_round:5.3f}, {x_hi_round:5.3f}]'
#        elif row_obs in ['K_NM']: 
#            rowstr += f' [{x_lo_round:3.0f}, {x_hi_round:3.0f}]'
#    rowstr += f' {likelihood_spec[0]:>13}, {likelihood_spec[1]}'
#    print(rowstr)
#
#



