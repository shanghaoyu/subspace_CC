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
sns.set('talk')
sns.set_style("whitegrid")
import matplotlib.patches as mpatches
import pandas as pd
from scipy.special import loggamma,gamma
from sklearn.gaussian_process.kernels import RBF, ConstantKernel,  WhiteKernel
from numpy.linalg import solve, cholesky
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.transforms as transforms
######################################################
######################################################
### read LECs set from file
######################################################
######################################################
def read_LEC(file_path):
    LEC_num = 17
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


def read_ccd_data(input_dir,data_count):
    ccd_batch   = np.zeros(data_count)
    file_count  = np.zeros(data_count)
    with open(input_dir,'r') as f:   
        count = len(open(input_dir,'rU').readlines())    
        data  = f.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(count):
            temp_1           = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count[loop1]=round(float(temp_1[0]))-1
            ccd_batch[loop1]=float(temp_1[1])
    assert data_count == count
    xx = np.argsort(file_count)
    ccd_batch = ccd_batch[xx]
    return ccd_batch

def read_correlation_energy_data(input_dir,data_count):
    ccd_batch   = np.zeros(data_count)
    file_count  = np.zeros(data_count)
    with open(input_dir,'r') as f:   
        count = len(open(input_dir,'rU').readlines())    
        data  = f.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(count):
            temp_1           = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            file_count[loop1]=round(float(temp_1[0]))-1
            ccd_batch[loop1]=float(temp_1[1])
    assert data_count == count
    xx = np.argsort(file_count)
    ccd_batch = ccd_batch[xx]
    return ccd_batch



def read_rskin_data(input_dir,data_count,start_line,position):
    rskin_batch   = np.zeros(data_count)
    file_count  = np.zeros(data_count)
    with open(input_dir,'r') as f:   
        count = len(open(input_dir,'rU').readlines())    
        data  = f.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(data_count):
            temp_1           = re.findall(r"[-+]?\d+\.?\d*",data[loop1+start_line])
            rskin_batch[loop1]=float(temp_1[position])
    return rskin_batch

def read_LEC_batch(file_path):
    LEC_batch = []
    my_LEC_label = ['cE','cD','c1','c2','c3','c4','Ct1S0pp','Ct1S0np','Ct1S0nn','Ct3S1','C1S0','C3P0','C1P1','C3P1','C3S1','CE1','C3P2']
    with open(file_path,'r') as f:
        count = len(open(file_path,'rU').readlines())
        data = f.readlines()
        wtf = re.match('#', 'abc',flags=0)
        LEC_label = data[0].split()
        LEC_label = LEC_label[1::]

        #print("LEC_label"+str(LEC_label))
        x = []
        for loop1 in range(len(my_LEC_label)):
            for loop2 in range(len(my_LEC_label)):
                if LEC_label[loop2]==my_LEC_label[loop1]:
                    x.append(loop2)
        #print("x=",x)
        for loop1 in range(1,count):
            myarray = np.fromstring(data[loop1],dtype=float, sep=' ')
            myarray = myarray[x]
            LEC_batch.append(myarray)
            #temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
            #LEC_batch[loop1-1][0:16]    = temp_1[0:16]
        #print("LEC_batch"+str(LEC_batch))
    return LEC_batch



#quadratic_curve
def f_2(x, A, B, C):
    return A*x*x + B*x + C

#cubic_curve
def f_3(x, A, B, C, D):
    return A*x*x*x + B*x*x + C*x + D

# kf / density converter:
def kf_density(kf,g):  # pnm:g=2 ; snm:g=4
    density = g* pow(kf,3)/6/pow(math.pi,2)
    return density
def density_kf(density,g):  # pnm:g=2 ; snm:g=4
    kf = pow(density*6*pow(math.pi,2)/g,1/3)
    return kf

# ---------------------------
# Statistics helper functions (From Chrisitian)
# ---
def hdi(samples, hdi_prob=0.68):
    '''
    Extract a Highest Denisty Interval (HDI). Only works for unimodal distributions.

    The HDI is the minimum width Bayesian credible interval (BCI). This method is inspired by Arviz
    https://arviz-devs.github.io/arviz/_modules/arviz/stats/stats.html#hdi

    Args:
        samples: obj
               object containing posterior samples.
        hdi_prob: float, optional
                Prob for which the highest density interval will be computed. Defaults to 0.68.

    Returns:
        np.ndarray. lower and upper values of the interval.
    '''
    samples = np.array(samples)
    nsamples = len(samples)

    samples = np.sort(samples)

    interval_idx_inc = int(np.floor(hdi_prob * nsamples)) # Number of samples in the HDI
    n_intervals = nsamples - interval_idx_inc             # Number of possible intervals
    # Differences between the first and last blocks
    interval_width = np.subtract(samples[interval_idx_inc:], samples[:n_intervals], dtype=np    .float_)

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width) # Finding the shortest interval
    hdi_min = samples[min_idx]
    hdi_max = samples[min_idx + interval_idx_inc]

    return np.array([hdi_min, hdi_max])



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


        self.ck_matrix    = None
        self.x_series     = None 
        self.eta_0        = None
        self.V_0          = None
        self.nu_0         = None
        self.tau_square_0 = None
        self.n_c          = None
        self.N            = None
        self.R_l          = None
        self.R_l_I        = None
        self.mean_0       = None
        self.nu           = None
        self.nu_tau_square= None
        self.nu_tau_square_0 = None 
        self.eta          = None
        self.V            = None
        self.c_square     = None
        self.Q_series     = None

    def fit_data(self, x, y, gaussian_noise,sigma,length):
        # store train data
        self.train_x = np.asarray(x)
        self.train_y = np.asarray(y)
        self.gaussian_noise = gaussian_noise
        self.sigma   = sigma
        self.length  = length


         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.l, self.sigma = params[0], params[1]
            Kyy = self.kernel(self.train_x, self.train_x)\
                  + 1e-8 * np.eye(len(self.train_x))
            return 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_x) * np.log(2 * np.pi)

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.length, self.sigma],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.length, self.sigma = res.x[0], res.x[1]

        self.is_fit = True

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma ** 2 * np.exp(-0.5 / self.length ** 2 * dist_matrix)

    def kernel_11(self, x1,x2): #first derivative with respect to both variables
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T) 
        return self.sigma**2/self.length**4*(self.length**2-dist_matrix)*\
               np.exp(-0.5 / self.length ** 2 * dist_matrix)

    def kernel_01(self, x1,x2): #first derivative with respect to the second variable
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        x1x2_matrix = x1 - x2.T
        return self.sigma**2/self.length**2*(x1x2_matrix)*\
               np.exp(-0.5 / self.length ** 2 * dist_matrix)
    def kernel_10(self, x1,x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        x2x1_matrix = -(x1 -x2.T)
        return self.sigma**2/self.length**2*(x2x1_matrix)*\
               np.exp(-0.5 / self.length ** 2 * dist_matrix)

    def kernel_22(self, x1,x2): #second derivative with respect to the first variable
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma**2/self.length**4*\
               (dist_matrix*dist_matrix/self.length**4 - 6*dist_matrix/self.length**2 + 3  )*\
                np.exp(-0.5 / self.length ** 2 * dist_matrix)


    def kernel_20(self, x1,x2): #second derivative with respect to the first variable
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma**2/self.length**4*(dist_matrix-self.length**2)*\
               np.exp(-0.5 / self.length ** 2 * dist_matrix)

    def kernel_02(self, x1,x2): #second derivative with respect to the first variable
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.sigma**2/self.length**4*(dist_matrix-self.length**2)*\
               np.exp(-0.5 / self.length ** 2 * dist_matrix)




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
        Kff11 = self.kernel_11(self.train_x, self.train_x) #+ self.gaussian_noise**2 * np.eye(l    en(self.train_x))  # (N, N)
        Kyy11 = self.kernel_11(x, x)  # (k, k)
        Kyf10 = self.kernel_10(x,self.train_x)  # (k,N)
        Kfy01 = self.kernel_01(self.train_x, x)  # (N,k)
     
        d_mu = Kyf10.dot(Kff_inv).dot(self.train_y)
        d_cov = Kyy11 - Kyf10.dot(Kff_inv).dot(Kfy01)

        Kyy22 = self.kernel_22(x, x)  # (k, k)
        Kyf20 = self.kernel_20(x,self.train_x)  # (k,N)
        Kfy02 = self.kernel_02(self.train_x, x)  # (N,k)
 
        dd_mu = Kyf20.dot(Kff_inv).dot(self.train_y)
        dd_cov = Kyy22 - Kyf20.dot(Kff_inv).dot(Kfy02)

        return mu, cov, d_mu, d_cov, dd_mu , dd_cov
   
    def fit_(self,eta_0,V_0,nu_0,tau_square_0, ck_matrix, x_series,y_ref,Q_series ):
        self.eta_0     = eta_0
        self.V_0       = V_0
        self.nu_0      = nu_0
        self.tau_square_0 = tau_square_0
        self.ck_matrix = ck_matrix
        self.x_series  = x_series      
        self.y_ref     = y_ref 
        self.Q_series  = Q_series 
 
    def RBF__(self, x1, x2, sigma, length):
        x1 = x1.reshape(-1,1)
        x2 = x2.reshape(-1,1)
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return sigma ** 2 * np.exp(-0.5 / length ** 2 * dist_matrix)

    def update_c_square_bar(self, l):  # ck_matrix is the observed c0 , c2 , c3 vector
        # c_square_bar has a normal-scaled-inverse-\chi_square prior
        #print("start update_c_square_bar")
        #self.eta_0        = 0
        #self.V_0          = 0
        #self.nu_0         = 10
        #self.tau_square_0 = (self.nu_0 - 2)/self.nu_0
        self.n_c          = self.ck_matrix.shape[0]    # number of order  (c0, c2, c3)
        self.N            = self.ck_matrix.shape[1]    # number of density points
        p                 = 1
        #eta_0_vec    = eta_0 * np.ones(p)
        B     = np.ones((self.N,p))
        V_0_matrix   = self.V_0 * np.ones((1,1))
        ll = np.array(l).reshape((1,))
        #print("wtf:"+str(ll.shape))
        R_l_2         = self.RBF__(self.x_series,self.x_series,1,l)
        #kernel = ConstantKernel(1.0, constant_value_bounds='fixed') * \
        #    RBF(1.0, length_scale_bounds='fixed')
        kernel             = RBF(length_scale=l) + WhiteKernel(noise_level=1e-10)
        #kernel             = RBF(length_scale=2.01) + WhiteKernel(noise_level=1e-10)
        #kernel             = kernel.clone_with_theta(ll)
        self.R_l           = kernel(self.x_series.reshape(-1,1))
        #kernel_new          = kernel.clone_with_theta(theta=[0.7])
        #print("my_sereies:"+str(self.x_series.reshape(-1,1)))
        #print("my_kernel_1:"+str(self.R_l))
        #print("my_kernel_2:"+str(R_l_2))
        #print("my_kernel:"+str(kernel_new))
        #kernel            = kernel.clone_with_theta(ll)
        #self.R_l          = kernel(self.x_series)


        #print("wtf:"+str(ll))
        self.R_l_I        = np.linalg.inv(self.R_l)    #  (R_l)^-1
        #print("B="+str(B_matrix))
        #print("n_c="+str(self.n_c))
        #print("N="+str(self.N))
        #print("R_l="+str(R_l))
        if np.all(V_0_matrix == 0):
            self.V        = np.zeros_like(V_0_matrix)
        else:
            self.V        = np.linalg.inv(np.linalg.inv(V_0_matrix) + self.n_c * B.T @ self.R_l_I @ B)
        #self.eta          = self.V

        self.c_square_0   = self.nu_0 * self.tau_square_0/(self.nu_0 - 2)
        # update c_square with observed ck
        self.nu           = self.nu_0 +self.N * self.n_c
        #V            = np.linalg.inv(V_0_matrix) + n_c*np.dot( np.dot(B,R_l_I),B)
        #print(np.dot(B,R_l_I))
        #print(np.dot(np.dot(B,R_l_I),B))
        #print("V ="+str(V))
        #s_square     =           # sample variance for correlated observations
        #nu_tau_square= nu_0 * tau_square_0 + n_c * s_square
        c_R_c_sum    = np.trace(self.ck_matrix @ self.R_l_I @ self.ck_matrix.T)

        self.nu_tau_square_0 = self.nu_0 * self.tau_square_0 
        self.nu_tau_square= self.nu_0 * self.tau_square_0 + c_R_c_sum
    
         
        self.c_square  = self.nu_tau_square/(self.nu - 2)
#        print("my_kernel: "+str(kernel))
#        print("my_length_scale: "+str(l))
#        #print("mean value of the updated c_square_bar ="+str(mean))
#        #print("mine R="+str(self.R_l))
#        #print("mine y="+str(self.ck_matrix))
#        print("mine V_0="+str(V_0_matrix))
#        print("mine V="+str(self.V))
#        print("mine df_0="+str(self.nu_0))
#        print("mine df="+str(self.nu))
#        #print("mine tau_square_0="+str(self.tau_square_0))
#        #print("mine df0* scale0**2="+str(self.nu_0 * self.tau_square_0))
#        #print("mine nu_tau_square="+str(self.nu_tau_square))
#        print("mine c_square_0="+str(self.c_square_0))
#        print("mine c_square="+str(self.c_square))
        return self.c_square
   
    
    def log_likelihood_l_Q(self,l):
        # update cn with l:
        self.update_c_square_bar(l)
        C = 1 / pow(np.linalg.det( 2*math.pi * self.R_l ) ,self.n_c) 
        #C = C * ()
        
        T1 = loggamma(self.nu/2) - self.nu/2 * np.log(self.nu_tau_square/2)
        T2 = loggamma(self.nu_0/2) - self.nu_0/2 * np.log(self.nu_tau_square_0/2)

        log_det_corr_mine = np.linalg.slogdet( 2*np.pi*self.R_l )[1] 
        #print("This is what I have:")
        #print(log_det_corr_mine)

        corr_L = np.linalg.cholesky(self.R_l)
        log_det_corr = 2 * np.sum(np.log(2 * np.pi * np.diagonal(corr_L)))
        #print("This is what they have:")
        #print(log_det_corr)


#####   there is something wrong here!!!
#        print("test_2")
#        print(self.R_l)
#        print(2 * np.pi * self.R_l)
#
#        print(np.linalg.det(self.R_l))
#        print(np.linalg.det(2*np.pi*self.R_l))
#         
#        print(np.linalg.cholesky(self.R_l))
#        print(np.linalg.cholesky(2*np.pi*self.R_l))
#
#        print(np.diagonal(np.linalg.cholesky(self.R_l)))
#        print(np.diagonal(np.linalg.cholesky(2*np.pi*self.R_l)))
#
#        print(pow(np.prod(np.diagonal(np.linalg.cholesky(self.R_l))),2))
#        print(pow(np.prod(np.diagonal(np.linalg.cholesky(2*np.pi*self.R_l))),2))
#
#        print(pow(np.prod(2 * np.pi * np.diagonal(np.linalg.cholesky(self.R_l))),2))
        ### for student-t process
        #log_pr_ck_l    = T1 - T2 - self.n_c / 2 * log_det_corr

        ### for gp 
        Kyy = self.c_square * self.R_l

        alpha = np.linalg.inv(Kyy) @ self.ck_matrix.T 

        log_pr_ck_l = -0.5 * np.einsum("ik,ik->k",self.ck_matrix.T,alpha) - 0.5 * np.linalg.slogdet(Kyy)[1] - 0.5 *self.N * np.log(2 * np.pi)
        log_pr_ck_l = log_pr_ck_l.sum(-1) 

        #print("mine : y_train="+str(self.ck_matrix))
        #print("mine : alpha="+str(np.linalg.inv(Kyy) @ self.ck_matrix.T))
        #print("mine : T2+T3="+str(- 0.5 * np.linalg.slogdet(Kyy)[1] - 0.5 *self.N * np.log(2 * np.pi)))
        #print("mine : T_1  = "+str((-0.5 * self.ck_matrix.dot(np.linalg.inv(Kyy)).dot(self.ck_matrix.T)).sum(-1)))
        #print("mine : T_all="+str(log_pr_ck_l))

        orders_fit     = [0, 2, 3]

        det_factor = np.sum(self.n_c * np.log(np.abs(self.y_ref)) + np.sum(orders_fit) * np.log(np.abs(self.Q_series)))
        log_pr_yk_l_Q  = log_pr_ck_l - det_factor

        #print("mine: "+str(Q_series))
        
        #print("mine: det_factor="+str(det_factor))

        return log_pr_yk_l_Q

    def log_likelihood_l_Q_method_error(self,l):
        # update cn with l:
        self.update_c_square_bar(l)
        C = 1 / pow(np.linalg.det( 2*math.pi * self.R_l ) ,self.n_c) 
        #C = C * ()
        
        T1 = loggamma(self.nu/2) - self.nu/2 * np.log(self.nu_tau_square/2)
        T2 = loggamma(self.nu_0/2) - self.nu_0/2 * np.log(self.nu_tau_square_0/2)

        log_det_corr_mine = np.linalg.slogdet( 2*np.pi*self.R_l )[1] 
        #print("This is what I have:")
        #print(log_det_corr_mine)

        corr_L = np.linalg.cholesky(self.R_l)
        log_det_corr = 2 * np.sum(np.log(2 * np.pi * np.diagonal(corr_L)))
        #print("This is what they have:")
        #print(log_det_corr)

        ### for gp 
        Kyy = self.c_square * self.R_l

        alpha = np.linalg.inv(Kyy) @ self.ck_matrix.T 

        log_pr_ck_l = -0.5 * np.einsum("ik,ik->k",self.ck_matrix.T,alpha) - 0.5 * np.linalg.slogdet(Kyy)[1] - 0.5 *self.N * np.log(2 * np.pi)
        log_pr_ck_l = log_pr_ck_l.sum(-1) 

        #print("mine : y_train="+str(self.ck_matrix))
        #print("mine : alpha="+str(np.linalg.inv(Kyy) @ self.ck_matrix.T))
        #print("mine : T2+T3="+str(- 0.5 * np.linalg.slogdet(Kyy)[1] - 0.5 *self.N * np.log(2 * np.pi)))
        #print("mine : T_1  = "+str((-0.5 * self.ck_matrix.dot(np.linalg.inv(Kyy)).dot(self.ck_matrix.T)).sum(-1)))
        #print("mine : T_all="+str(log_pr_ck_l))

        orders_fit     = [0, 2, 3]

        #det_factor = np.sum(self.n_c * np.log(np.abs(self.y_ref)) + np.sum(orders_fit) * np.log(np.abs(self.Q_series)))
        log_pr_yk_l_Q  = log_pr_ck_l 

        #print("mine: "+str(Q_series))
        
        #print("mine: det_factor="+str(det_factor))

        return log_pr_yk_l_Q


    def student_t_test_log_like(self):
        kernel = RBF(length_scale=0.7) + WhiteKernel(noise_level=1e-10)
        corr_  = kernel(self.x_series.reshape(-1,1))
        sqrt_R = cholesky(corr_)
        decomp ='cholesky'
        if decomp == 'cholesky':
            logdet_R = 2 * np.log(np.diag(sqrt_R)).sum()
        elif decomp == 'eig':
            eig, Q = sqrt_R
            logdet_R = np.log(eig).sum()
        else:
            raise ValueError('decomposition must be "cholesky" or "eig"')

        #log_like = (N * np.log(2*np.pi) + logdet_R)
        log_like = (self.N * np.log(2*np.pi) + logdet_R)
        log_like_old = 2 * np.sum(np.log(2 * np.pi * np.diagonal(sqrt_R)))
        log_like_mine = np.linalg.slogdet(2*np.pi*corr_)[1]
        logdet_R_mine = np.linalg.slogdet(corr_)[1]
        print("student_t_test")
        print("R="+str(corr_))
        print("logdet_R="+str(logdet_R))
        print("logdet_R_mine="+str(logdet_R_mine))
        print("log_like="+str(log_like))
        print("log_like_old="+str(log_like_old))
        print("log_like_mine="+str(log_like_mine))

    def setup_cross_cov_matrix(self,order,variance1,l1,variance2,l2,x_series,y_ref_1,y_ref_2,Q_1,Q_2,rho_empirical,method_switch):
        print("#################################") 
        print("###setup cross covariance matirx") 
        print("#################################") 
        mag = 1
        l1 = l1 * mag
        l2 = l2 * mag

        def truncation_error_cov_matrix_element(order,y_ref_x,y_ref_xp,Q_x,Q_xp,kernel_matrix_element):
            return y_ref_x * y_ref_xp * pow(Q_x * Q_xp,order+1)/(1 - Q_x * Q_xp) * kernel_matrix_element

        def plot_matrix_visualization():
            fontsize_x_label = 10
            plt.figure(figsize=(5,10))
            matplotlib.rcParams['xtick.direction'] = 'in'
            matplotlib.rcParams['ytick.direction'] = 'in'
            plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
            
            plt.matshow(K_all)
            plt.vlines(len(K_all)/2-0.5,0,len(K_all)-1,lw=1)
            plt.hlines(len(K_all)/2-0.5,0,len(K_all)-1,lw=1)

            #plt.text(50, -2, 'E/N')
            #plt.text(170, -2, 'E/A')
            #plt.text(-20, 50, 'E/N')
            #plt.text(-20, 170, 'E/A')

            plt.xticks([])
            plt.yticks([])

            plt.xlabel("E/N                                  E/A"\
                           ,fontsize=fontsize_x_label)
            plt.ylabel("E/A                                  E/N"\
                           ,fontsize=fontsize_x_label)
            plot_path = 'cross_cov_kernel_matrix.pdf'
            plt.savefig(plot_path)
            plt.close('all')
 
 
        sigma1   = pow(variance1,0.5)
        sigma2   = pow(variance2,0.5)

        #  first way:
        if method_switch == 1:
            kernel_xx = RBF(length_scale=l1)# + WhiteKernel(noise_level=1e-10)
            K_xx     = variance1 * kernel_xx(x_series.reshape(-1,1))
            kernel_yy = RBF(length_scale=l2)# + WhiteKernel(noise_level=1e-10)
            K_yy     = variance2 * kernel_yy(x_series.reshape(-1,1))
            #print("x_series="+str(x_series)) 
            #print("l1="+str(l1)) 
            #print("l2="+str(l2)) 
            #print("variance1="+str(variance1)) 
            #print("variance2="+str(variance2)) 
            #print("K_xx+K_yy="+str(K_xx+K_yy)) 
            #print("for snm sigma = "+str(variance2**(0.5)))
            #print("for snm length= "+str(l2))
            #print("for snm diag_kernel= "+str(np.diag(K_yy)))
            #print("for snm diag_kernel= "+str(K_yy))
            #print("for snm y_ref1 = "+str(y_ref_1))
            #print("for snm y_ref2 = "+str(y_ref_2))
            #print("for snm Q_series2 = "+str(Q_2))
    
            # choose cross kernel 
            kernel_xy = RBF(length_scale=(pow((l1**2+l2**2)/2,0.5)))# + WhiteKernel(noise_level=1e-10)
            rho      = pow((2*l1*l2/(l1**2+l2**2)),0.5)
            print("our rho = "+str(rho))
            rho      = rho_empirical 

            K_xy     = sigma1 * sigma2 * rho * kernel_xy(x_series.reshape(-1,1))
            cov_XX = np.zeros((len(x_series),len(x_series)))
            cov_YY = np.zeros((len(x_series),len(x_series)))
            cov_XY = np.zeros((len(x_series),len(x_series)))
            cov_all= np.zeros((2*len(x_series),2*len(x_series)))
    
            for loop1 in range(len(x_series)):
                for loop2 in range(len(x_series)):
                    cov_XX[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_1[loop2],Q_1[loop1],Q_1[loop2],K_xx[loop1,loop2])     
                    cov_YY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_2[loop1],y_ref_2[loop2],Q_2[loop1],Q_2[loop2],K_yy[loop1,loop2])     
                    cov_XY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_2[loop2],Q_1[loop1],Q_2[loop2],K_xy[loop1,loop2])     
             
            cov_all = np.hstack((cov_XX,cov_XY))
            temp_   = np.hstack((cov_XY.T,cov_YY))
            cov_all = np.vstack((cov_all,temp_))
    
            K_all = np.hstack((K_xx/variance1,  K_xy/sigma1/sigma2/rho))
            temp_ = np.hstack((K_xy.T/sigma1/sigma2/rho,K_yy/variance2))
            K_all = np.vstack((K_all ,temp_    ))
    
            #print(K_xx)
            #print(cov_XX)
            #print(cov_YY)
            #print(cov_all)
            #print(K_all)
    
            # matrix visualization
            #plot_matrix_visualization()
            return cov_all

        # second way:  
        # intrinsic coregionalization model (l1 = l2)
        elif method_switch == 2:
            kernel = RBF(length_scale=(l1+l2)/2)# + WhiteKernel(noise_level=1e-10)
            K_xx     = variance1 * kernel(x_series.reshape(-1,1))
            K_yy     = variance2 * kernel(x_series.reshape(-1,1))
            K_xy     = sigma1 * sigma2 * rho_empirical * kernel(x_series.reshape(-1,1))
            #print("x_series="+str(x_series)) 
            #print("l1="+str(l1)) 
            #print("l2="+str(l2)) 
            #print("variance1="+str(variance1)) 
            #print("variance2="+str(variance2)) 
            #print("K_xx+K_yy="+str(K_xx+K_yy)) 
            cov_XX = np.zeros((len(x_series),len(x_series)))
            cov_YY = np.zeros((len(x_series),len(x_series)))
            cov_XY = np.zeros((len(x_series),len(x_series)))
            cov_all= np.zeros((2*len(x_series),2*len(x_series)))
    
            for loop1 in range(len(x_series)):
                for loop2 in range(len(x_series)):
                    cov_XX[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_1[loop2],Q_1[loop1],Q_1[loop2],K_xx[loop1,loop2])     
                    cov_YY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_2[loop1],y_ref_2[loop2],Q_2[loop1],Q_2[loop2],K_yy[loop1,loop2])     
                    cov_XY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_2[loop2],Q_1[loop1],Q_2[loop2],K_xy[loop1,loop2])     
             
            cov_all = np.hstack((cov_XX,cov_XY))
            temp_   = np.hstack((cov_XY.T,cov_YY))
            cov_all = np.vstack((cov_all,temp_))
    
            return cov_all 

        # third way:  
        # l1 and l2 for diagnal component, (l1+l2)/2 for off-diagonal component
        elif method_switch == 3:
            kernel_xx = RBF(length_scale=l1) + WhiteKernel(noise_level=1e-10)
            K_xx      = variance1 * kernel_xx(x_series.reshape(-1,1))
            kernel_yy = RBF(length_scale=l2) + WhiteKernel(noise_level=1e-10)
            K_yy      = variance2 * kernel_yy(x_series.reshape(-1,1))
            kernel_xy = RBF(length_scale=(l1+l2)/2) + WhiteKernel(noise_level=1e-10)
            K_xy      = sigma1 * sigma2 * rho_empirical * kernel_xy(x_series.reshape(-1,1))

            cov_XX = np.zeros((len(x_series),len(x_series)))
            cov_YY = np.zeros((len(x_series),len(x_series)))
            cov_XY = np.zeros((len(x_series),len(x_series)))
            cov_all= np.zeros((2*len(x_series),2*len(x_series)))
    
            for loop1 in range(len(x_series)):
                for loop2 in range(len(x_series)):
                    cov_XX[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_1[loop2],Q_1[loop1],Q_1[loop2],K_xx[loop1,loop2])     
                    cov_YY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_2[loop1],y_ref_2[loop2],Q_2[loop1],Q_2[loop2],K_yy[loop1,loop2])     
                    cov_XY[loop1,loop2] = truncation_error_cov_matrix_element\
                      (order,y_ref_1[loop1],y_ref_2[loop2],Q_1[loop1],Q_2[loop2],K_xy[loop1,loop2])     
            print("K_xy="+str(K_xy)) 
            cov_all = np.hstack((cov_XX,cov_XY))
            temp_   = np.hstack((cov_XY.T,cov_YY))
            cov_all = np.vstack((cov_all,temp_))
    
            return cov_all 



        else:
            print("Cross covariance matrix method switch error!")



    def plot_l_Q(self):
        l_range = np.linspace(0,1.5, 100)
        #Q_range = np.linspace(0.3, 20, 30)
        # Compute the log likelihood for values on this grid.
        #l_Q_loglike = np.array([self.log_likelihood_l_Q(l_) for l_ in np.log(l_range)])
        l_Q_loglike = np.array([self.log_likelihood_l_Q(l_) for l_ in l_range])
        #print(np.log(l_range))
        #print(l_Q_loglike)
        fontsize_legend = 12
        fontsize_x_label = 15
        fontsize_y_label = 15
        fig1 = plt.figure('fig1')
    #    plt.figure(figsize=(5,10))
    #    plt.subplots_adjust(wspace =0.3, hspace =0.4)
    
    #####################################################
        matplotlib.rcParams['xtick.direction'] = 'in'
        matplotlib.rcParams['ytick.direction'] = 'in'
        plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
        #X,Y = np.meshgrid(l_range,Q_range)
        x_list = l_range
        y_list = l_Q_loglike

        l = plt.scatter(x_list,y_list,color='crimson',s = 5, marker = '.')

        #plt.contourf(X,Y,l_Q_loglike)
        #plt.contour(X,Y,l_Q_loglike)
        plot_path = 'l_Q_loglike.pdf'
        plt.savefig(plot_path)
        plt.close('all')



        # Makes sure that the values don't get too big or too small
#        ls_ratio_like = np.exp(ls_ratio_loglike - np.max(ls_ratio_loglike))
#        
#        # Now compute the marginal distributions
#        ratio_like = np.trapz(ls_ratio_like, x=ls_vals, axis=-1)
#        ls_like = np.trapz(ls_ratio_like, x=ratio_vals, axis=0)
#        
#        # Normalize them
#        ratio_like /= np.trapz(ratio_like, x=ratio_vals, axis=0)
#        ls_like /= np.trapz(ls_like, x=ls_vals, axis=0)    




#################################################
### generate NM observables with pnm and snm data
#################################################
def generate_NM_observable(pnm_data,snm_data,density_sequence,switch):
    raw_data   = []
    dens_count = len(pnm_data)
    density_accuracy = 0.0001
    if (switch == "GP"):
        dens_count = len(density_sequence)
        #######################################################
        #######################################################
        ####  use GP to find the saturation point
        #######################################################
        #######################################################
        #t1 = time.time()
        train_x = np.arange(density_sequence.min(),density_sequence.max(),0.02)
        train_x = train_x.reshape(-1,1)
        train_y_1 = pnm_data
        train_y_2 = snm_data
        test_x  = np.arange(density_sequence.min(),density_sequence.max(),density_accuracy).reshape(-1,1)

        gpr = GP_test()
        gaussian_noise = 0.02

   # def fit_data(self, x, y, gaussian_noise,sigma,length):
        gpr.fit_data(x = train_x, y=train_y_2, gaussian_noise=gaussian_noise,sigma=0.25,length=100)

        snm, snm_cov, d_snm, d_snm_cov,dd_snm,dd_snm_cov = gpr.predict(test_x)

        #test_y_1  = snm.ravel()
        #test_dy_1 = d_snm.ravel()
        #confidence_1    = 2 * np.sqrt(np.diag(snm_cov))
        #confidence_dy_1 = 2 * np.sqrt(np.diag(d_snm_cov))


        #density_range = test_x[np.where((snm[:]<(snm[iX]+confidence_1[iX]))&(snm[:]>(snm[iX]-confidence_1[iX])))]
        #print("saturation density: %.3f +/- %.3f" % (test_x[iX], 0.5*(np.max(density_range)-np.min(density_range))))
        #print("saturation energy:  %.3f +/- %.3f" % (snm[iX] , confidence_1[iX]))
        gpr = GP_test()
        gpr.fit_data(train_x, train_y_1, gaussian_noise,0.25,100)

        pnm, pnm_cov, d_pnm, d_pnm_cov, dd_pnm, dd_pnm_cov = gpr.predict(test_x)

        raw_data.append(train_x)
        raw_data.append(train_y_1)
        raw_data.append(train_y_2)
        raw_data.append(test_x)

        raw_data.append(pnm)
        raw_data.append(pnm_cov)
        raw_data.append(d_pnm)
        raw_data.append(d_pnm_cov)
        raw_data.append(dd_pnm)
        raw_data.append(dd_pnm_cov)
        raw_data.append(snm)
        raw_data.append(snm_cov)
        raw_data.append(d_snm)
        raw_data.append(d_snm_cov)
        raw_data.append(dd_snm)
        raw_data.append(dd_snm_cov)


        #test_y_2  = pnm.ravel()
        #test_dy_2 = d_pnm.ravel()
        #confidence_2    = 2 * np.sqrt(np.diag(pnm_cov))
        #confidence_dy_2 = 2 * np.sqrt(np.diag(d_pnm_cov))

        #print("pnm energy:  %.3f +/- %.3f" % ( pnm[iX], confidence_2[iX]))
        #print("symmetry energy:  %.3f +/- %.3f" % (pnm[iX]-snm[iX],(confidence_1[iX]+confidence_2[iX])))

        #t2 = time.time()
        #print("time for GP : "+ str(t2-t1))
        iX=np.argmin(snm)
        saturation_density = test_x[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]
        s  = pnm - snm
        u  = test_x / saturation_density
        L  = 3 * u[iX]*saturation_density*(d_pnm[iX]-d_snm[iX])
        K  = 9* pow(saturation_density,2)*dd_snm[iX]     ## K at saturation denstiy (K_infinite)

        K_rho = np.zeros(len(test_x))   ## K for each density point
        d_snm = d_snm.reshape(test_x.shape)
        dd_snm = dd_snm.reshape(test_x.shape)
        P =  pow(test_x,2) * d_snm
        K_rho = 9* pow(test_x,2)*dd_snm + 18 / test_x * P  
        K_rho =  18 / test_x * P  

        K_sym = 9* pow(saturation_density,2)*dd_snm
        d_pnm = d_pnm.reshape(test_x.shape)
        d_snm = d_snm.reshape(test_x.shape)
        L_rho  = 3 * test_x * (d_pnm-d_snm)
        S_rho = symmetry_energy + L*((test_x - saturation_density) /3/saturation_density) + 0.5 *K_sym * pow((test_x - saturation_density)/3/saturation_density,2)


        raw_data.append(K_rho)
        raw_data.append(S_rho)
        raw_data.append(L_rho)
        raw_data.append(iX)
        if iX == (len(snm)-1) or  iX == 0 :
        #print("iX out of range")
            saturation_density  = np.nan
            saturation_energy   = np.nan
            symmetry_energy     = np.nan
            L                   = np.nan
            K                   = np.nan


        #plot_5(train_x, train_y_1,train_y_2,test_x,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm,d_snm_cov, dd_snm, dd_snm_cov )

    elif (switch =="fit_curve_quadratic"):
        A2,B2,C2 = optimize.curve_fit(f_2,density_sequence,snm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        snm = A2*x2*x2 + B2*x2 + C2
        iX=np.argmin(snm)

        A2,B2,C2 = optimize.curve_fit(f_2,density_sequence,pnm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        pnm = A2*x2*x2 + B2*x2 + C2

        saturation_density = x2[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]

    elif (switch =="fit_curve_cubic"):
        A3,B3,C3,D3 = optimize.curve_fit(f_3,density_sequence,snm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        snm = A3*x2*x2*x2 + B3*x2*x2 + C3*x2 +D3
        iX=np.argmin(snm)

        A3,B3,C3,D3 = optimize.curve_fit(f_3,density_sequence,pnm_data)[0]
        x2  = np.arange(0.12,0.20,density_accuracy)
        pnm = A3*x2*x2*x2 + B3*x2*x2 + C3*x2 +D3

        saturation_density = x2[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]

    elif (switch =="interpolation"):
        #### here k = 2,3,4 , also one can set "s = 0" forcing the curve to go through all training point
        spl_ccd_snm    = interpolate.UnivariateSpline(density_sequence,snm_data,k=4,s=0)
        spl_ccd_pnm    = interpolate.UnivariateSpline(density_sequence,pnm_data,k=4,s=0)
        #spldens        = np.linspace(density_sequence[0],density_sequence[len(density_sequence)-1],num=interpol_count)
        
        spldens  = np.arange(density_sequence.min(),density_sequence.max(),density_accuracy)

        snm = spl_ccd_snm(spldens)
        pnm = spl_ccd_pnm(spldens)
        pnm_cov = np.zeros(len(pnm))
        snm_cov = np.zeros(len(snm))
        d_pnm = np.diff(pnm) / np.diff(spldens)
        d_pnm_cov = np.zeros(len(d_pnm))
        dd_pnm = np.diff(d_pnm) / np.diff(spldens[1::])
        dd_pnm_cov = np.zeros(len(dd_pnm))

        d_snm = np.diff(snm) / np.diff(spldens)
        d_snm_cov = np.zeros(len(d_snm))
        dd_snm = np.diff(d_snm) / np.diff(spldens[1::])
        dd_snm_cov = np.zeros(len(dd_snm))

        iX=np.argmin(snm)
        saturation_density = spldens[iX]
        saturation_energy  = snm[iX]
        symmetry_energy    = pnm[iX]-snm[iX]
        s  = pnm - snm
        u  = spldens / saturation_density
        ds = np.diff(s)       
        du = np.diff(u)
        L      = 3 * u[iX]*ds[iX-1]/du[iX-1]
        L_rho  = 3 * spldens[1::] * ds /np.diff(spldens)

        df = np.diff(snm) / np.diff(spldens)
        ddf= np.diff(df) / np.diff(spldens[1::])        
        K     = 9* pow(saturation_density,2)*ddf[iX-2]

        #df = df.reshape(test_x.shape)
        #ddf = ddf.reshape(test_x.shape)
        P =  pow(spldens[1::],2) * df
        K_rho = 9* pow(spldens[2::],2)*ddf + 18 / spldens[2::] * P[1::]  

        #K_rho = 100*pow(spldens[2::]/saturation_density,2) + 18 / spldens[2::] * P[1::]  
        #K_rho =  18 / spldens[2::] * P[1::]  
        #print(9* pow(spldens[2::],2)*ddf)
        df = ds / np.diff(spldens)
        ddf= np.diff(df) / np.diff(spldens[1::])        
        K_sym = 9* pow(saturation_density,2)*ddf[iX-2]
             
        S_rho = symmetry_energy + L*((spldens - saturation_density) /3/saturation_density) + 0.5 *K_sym * pow((spldens - saturation_density)/3/saturation_density,2)

        raw_data.append(density_sequence)
        raw_data.append(pnm)
        raw_data.append(snm)
        raw_data.append(spldens)
        raw_data.append(pnm)
        raw_data.append(pnm_cov)
        raw_data.append(d_pnm)
        raw_data.append(d_pnm_cov)
        raw_data.append(dd_pnm)
        raw_data.append(dd_pnm_cov)
        raw_data.append(snm)
        raw_data.append(snm_cov)
        raw_data.append(d_snm)
        raw_data.append(d_snm_cov)
        raw_data.append(dd_snm)
        raw_data.append(dd_snm_cov)
        raw_data.append(K_rho)
        raw_data.append(S_rho)
        raw_data.append(L_rho)
        raw_data.append(iX)

        if iX == (len(snm)-1) or  iX == 0 :
            #print("iX out of range")
            saturation_density  = np.nan
            saturation_energy   = np.nan
            symmetry_energy     = np.nan
            L                   = np.nan
            K                   = np.nan

        #plot_5(density_sequence, pnm_data,snm_data,spldens,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm,d_snm_cov, dd_snm, dd_snm_cov )
       # print("ddf="+str(ddf[iX]))
       # print("test="+str(snm))
    else:
        print("error_1")

#N_132_saturation_snm = np.min(N_132_interpolation[:,1])
#temp1 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),0]
#N_132_saturation_dens = temp1[0]
#temp2 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),2]
#N_132_saturation_pnm = temp2[0]
#print ('snm='+str(N_132_saturation_snm))
#print ('dens='+str(N_132_saturation_dens))
#print ('pnm='+str(N_132_saturation_pnm))
#saturation_energy_132 = N_132_saturation_pnm- N_132_saturation_snm
#print ('saturation_energy='+str(saturation_energy_132))
#
#
#df_132 = np.diff(N_132_interpolation[:,1])/np.diff(N_132_interpolation[:,0])
#ddf_132 = np.diff(df_132) /np.diff(N_132_interpolation[1:len(N_132_interpolation),0])
#temp3 = ddf_132[np.where(N_132_interpolation[:,1]==N_132_saturation_snm)]
#ddf_saturation_dens_132 = temp3[0]
#
#K0 = 9* pow(N_132_saturation_dens,2)*ddf_saturation_dens_132
#print ('K0_132=',K0)
#
#S = N_132_interpolation[:,2] - N_132_interpolation[:,1]
#u = N_132_interpolation[:,0] / N_132_saturation_dens
#ds = np.diff(S)
#du = np.diff(u)
##u_0 is the position of saturation point
#u_0 = np.where(u[:]== 1)
#print('test='+str(u[688]))
#
#print('u_0'+str(u_0))
#L = 3 * u[u_0]*ds[u_0]/du[u_0]
    
    return saturation_density, saturation_energy, symmetry_energy, L, K, raw_data

############################################################
### generate NM observables with pnm and snm data in batches
############################################################
def generate_NM_observable_batch(pnm_batch_all,snm_batch_all,density_sequence_all,switch):
    validation_count = len(pnm_batch_all[:])
    density_count    = len(pnm_batch_all[0])
    saturation_density_batch = np.zeros(validation_count)
    saturation_energy_batch  = np.zeros(validation_count)
    symmetry_energy_batch    = np.zeros(validation_count)
    L_batch                  = np.zeros(validation_count)
    K_batch                  = np.zeros(validation_count)
    for loop in range(validation_count):
        saturation_density_batch[loop], saturation_energy_batch[loop], symmetry_energy_batch[loop],L_batch[loop], K_batch[loop], raw_data= generate_NM_observable(pnm_batch_all[loop,:],snm_batch_all[loop,:],density_sequence_all[loop,:],switch)

    return saturation_density_batch, saturation_energy_batch,symmetry_energy_batch,L_batch, K_batch
 
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


 
################################
## plots
################################
def plot_3(list_1,list_2,list_3,list_4,list_5):
    fontsize_legend = 12
    fontsize_x_label = 15
    fontsize_y_label = 15
    fig1 = plt.figure('fig1')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = -27.7
    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_2
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_2),2))
    for loop1 in range(0,len(list_2)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=5, linewidth=1, marker=".",label = r"$\rm{HM \ (589\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163, -15.386,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    #plt.xlim((0.11,0.225))
    #plt.ylim((-18.1,-11.9))
    #plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    #plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)

    plt.ylim((-21,-5))
    plt.xlim((0.11,0.21))
    plt.xlabel(r"$\rm{saturation \ density} \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{saturation \ energy} \ [\rm{MeV}]$",fontsize=fontsize_y_label)


    plot_path = 'NM_emulator_589_samples_ccd_1.pdf'
    plt.savefig(plot_path)
    plt.close('all')

####################################################
    fig2 = plt.figure('fig2')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = -27.7
    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_3
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=5, linewidth=1, marker=".",label = r"$\rm{HM \ (589\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163,31.5 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{saturation \ density} \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{symmetry \ energy} \ [\rm{MeV}]$",fontsize=fontsize_y_label)


    plot_path = 'NM_emulator_589_samples_ccd_2.pdf'
    plt.savefig(plot_path)

####################################################
    fig3 = plt.figure('fig3')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
#    y_min = 0
#    y_max = 0.04
#    x_min = -27.7
#    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_4
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g")#,bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=5, linewidth=1, marker=".",label = r"$\rm{HM \ (589\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163,251 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{saturation \ density} \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{K} \ [\rm{MeV}]$",fontsize=fontsize_y_label)
    plt.ylim((0,500))


    plot_path = 'NM_emulator_589_samples_ccd_3.pdf'
    plt.savefig(plot_path)


####################################################
    fig4 = plt.figure('fig4')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
#    y_min = 0
#    y_max = 0.04
#    x_min = -27.7
#    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_3
    y_list = list_5
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=5, linewidth=1, marker=".",label = r"$\rm{HM \ (589\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163,251 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{symmetry \ energy} \ [\rm{MeV}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{L} \ [\rm{MeV}]$",fontsize=fontsize_y_label)
    #plt.ylim((0,500))


    plot_path = 'NM_emulator_589_samples_ccd_4.pdf'
    plt.savefig(plot_path)



####################################################
    fig5 = plt.figure('fig5')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
#    y_min = 0
#    y_max = 0.04
#    x_min = -27.7
#    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_3
    y_list = list_4
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=5, linewidth=1, marker=".",label = r"$\rm{HM \ (589\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163,251 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{symmetry \ energy} \ [\rm{MeV}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{K} \ [\rm{MeV}]$",fontsize=fontsize_y_label)
    plt.ylim((-50,650))


    plot_path = 'NM_emulator_589_samples_ccd_5.pdf'
    plt.savefig(plot_path)












################################
## plots
################################
def plot_4(list_1,list_2,list_3,list_4):
    fig1 = plt.figure('fig1')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = -27.7
    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_2
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_2),2))
    for loop1 in range(0,len(list_2)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=20, linewidth=1, marker="x",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = 9)
    #l2 = plt.scatter (0.16196, -14.812,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    #plt.xlim((0.11,0.225))
    #plt.ylim((-18.1,-11.9))
    #plt.xticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)
    #plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)

    plt.xlabel(r"$\rm{Rskin} \ [\rm{fm}]$",fontsize=10)
    plt.ylabel(r"$\rm{symmetry \ energy} \ [\rm{MeV}]$",fontsize=10)


    plot_path = 'Pb208_34_sample_skin_vs_S_ccdt.pdf'
    plt.savefig(plot_path)
    plt.close('all')

####################################################
    fig2 = plt.figure('fig2')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = -27.7
    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_3
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=20, linewidth=1, marker="x",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = 9)
    #l2 = plt.scatter (0.16196,30.866 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{Rskin} \ [\rm{fm}]$",fontsize=10)
    plt.ylabel(r"$\rm{L} $",fontsize=10)


    plot_path = 'Pb208_34_sample_skin_vs_L_ccdt.pdf'
    plt.savefig(plot_path)


####################################################
    fig3 = plt.figure('fig3')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
#    y_min = 0
#    y_max = 0.04
#    x_min = -27.7
#    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_1
    y_list = list_4
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=20, linewidth=1, marker="x",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = 9)
    #l2 = plt.scatter (0.16196,30.866 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{Rskin} \ [\rm{fm}]$",fontsize=10)
    plt.ylabel(r"$\rm{K} $",fontsize=10)


    plot_path = 'Pb208_34_sample_skin_vs_K_ccdt.pdf'
    plt.savefig(plot_path)

####################################################
    fig4 = plt.figure('fig4')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
#    y_min = 0
#    y_max = 0.04
#    x_min = -27.7
#    x_max = -27.5
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_2
    y_list = list_4
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=20, linewidth=1, marker="x",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = 9)
    #l2 = plt.scatter (0.16196,30.866 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlabel(r"$\rm{S} \ [\rm{MeV}]$",fontsize=10)
    plt.ylabel(r"$\rm{K}  \ [\rm{MeV}]$",fontsize=10)


    plot_path = 'Pb208_34_sample_S_vs_K_ccdt.pdf'
    plt.savefig(plot_path)

#####################################################
### plot pnm snm and their first and second derivative
#####################################################
def plot_5(train_x, train_y_1,train_y_2,dens_list,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm, d_snm_cov, dd_snm, dd_snm_cov ):
    print("start plotting 5")
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    x_list_1     = dens_list.ravel()

    y_list_1     = pnm.ravel()
    y_list_2     = d_pnm.ravel()*0.1
    y_list_3     = dd_pnm.ravel()*0.01

    confidence_1 = 2*np.sqrt(np.diag(pnm_cov))
    confidence_2 = 2*np.sqrt(np.diag(d_pnm_cov))*0.1
    confidence_3 = 2*np.sqrt(np.diag(dd_pnm_cov))*0.01

    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.fill_between(x_list_1, y_list_1 + confidence_1, y_list_1 - confidence_1, alpha=0.1)
    plt.plot(x_list_1, y_list_1, label="GP")
    plt.fill_between(x_list_1, y_list_2 + confidence_2, y_list_2 - confidence_2, alpha=0.1)
    plt.plot(x_list_1, y_list_2, label="first derivative * 0.1")

    plt.fill_between(x_list_1, y_list_3 + confidence_3, y_list_3 - confidence_3, alpha=0.1)
    plt.plot(x_list_1, y_list_3, label="second derivative * 0.01")


    plt.scatter(train_x, train_y_1, label="train", c="black", marker="x",zorder = 5)
    plt.legend(fontsize=8)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xlim((0.12,0.2001)) 
    plt.ylim((-1,21)) 
    plt.xticks([])

### plot snm
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(212)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    y_list_4     = snm.ravel()
    y_list_5     = d_snm.ravel()*0.1
    y_list_6     = dd_snm.ravel()*0.01

    confidence_4 = 2*np.sqrt(np.diag(snm_cov))
    confidence_5 = 2*np.sqrt(np.diag(d_snm_cov))*0.1
    confidence_6 = 2*np.sqrt(np.diag(dd_snm_cov))*0.01

    plt.hlines(0,0.12,0.20,ls=':',lw = 0.5, alpha = 0.3)
    plt.vlines(0.1629,-15.387,0,ls=':',lw = 0.5, alpha = 0.3)

    plt.fill_between(x_list_1, y_list_4 + confidence_4, y_list_4 - confidence_4, alpha=0.1)
    plt.plot(x_list_1, y_list_4, label="GP")
    plt.fill_between(x_list_1, y_list_5 + confidence_5, y_list_5 - confidence_5, alpha=0.1)
    plt.plot(x_list_1, y_list_5, label="first derivative")

    plt.fill_between(x_list_1, y_list_6 + confidence_6, y_list_6 - confidence_6, alpha=0.1)
    plt.plot(x_list_1, y_list_6, label="second derivative")
    plt.scatter(train_x, train_y_2, label="train", c="black", marker="x",zorder = 5)

    plt.xlim((0.12,0.2001)) 
    plt.ylim((-17,15.1)) 
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{snm}/A}[\rm{MeV}]$",fontsize=15)
    plot_path = 'snm_pnm_with_GP.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')


#####################################################
### plot pnm snm and their first and second derivative
#####################################################
def plot_5(train_x, train_y_1,train_y_2,dens_list,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm, d_snm_cov, dd_snm, dd_snm_cov ):
    print("start plotting 5")
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    x_list_1     = dens_list.ravel()

    y_list_1     = pnm.ravel()
    y_list_2     = d_pnm.ravel()*0.1
    y_list_3     = dd_pnm.ravel()*0.01

    confidence_1 = 2*np.sqrt(np.diag(pnm_cov))
    confidence_2 = 2*np.sqrt(np.diag(d_pnm_cov))*0.1
    confidence_3 = 2*np.sqrt(np.diag(dd_pnm_cov))*0.01

    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.fill_between(x_list_1, y_list_1 + confidence_1, y_list_1 - confidence_1, alpha=0.1)
    plt.plot(x_list_1, y_list_1, label="GP")
    plt.fill_between(x_list_1, y_list_2 + confidence_2, y_list_2 - confidence_2, alpha=0.1)
    plt.plot(x_list_1, y_list_2, label="first derivative * 0.1")

    plt.fill_between(x_list_1, y_list_3 + confidence_3, y_list_3 - confidence_3, alpha=0.1)
    plt.plot(x_list_1, y_list_3, label="second derivative * 0.01")


    plt.scatter(train_x, train_y_1, label="train", c="black", marker="x",zorder = 5)
    plt.legend(fontsize=8)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xlim((0.12,0.2001)) 
    plt.ylim((-1,21)) 
    plt.xticks([])

### plot snm
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(212)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    y_list_4     = snm.ravel()
    y_list_5     = d_snm.ravel()*0.1
    y_list_6     = dd_snm.ravel()*0.01

    confidence_4 = 2*np.sqrt(np.diag(snm_cov))
    confidence_5 = 2*np.sqrt(np.diag(d_snm_cov))*0.1
    confidence_6 = 2*np.sqrt(np.diag(dd_snm_cov))*0.01

    plt.hlines(0,0.12,0.20,ls=':',lw = 0.5, alpha = 0.3)
    plt.vlines(0.1629,-15.387,0,ls=':',lw = 0.5, alpha = 0.3)

    plt.fill_between(x_list_1, y_list_4 + confidence_4, y_list_4 - confidence_4, alpha=0.1)
    plt.plot(x_list_1, y_list_4, label="GP")
    plt.fill_between(x_list_1, y_list_5 + confidence_5, y_list_5 - confidence_5, alpha=0.1)
    plt.plot(x_list_1, y_list_5, label="first derivative")

    plt.fill_between(x_list_1, y_list_6 + confidence_6, y_list_6 - confidence_6, alpha=0.1)
    plt.plot(x_list_1, y_list_6, label="second derivative")
    plt.scatter(train_x, train_y_2, label="train", c="black", marker="x",zorder = 5)

    plt.xlim((0.12,0.2001)) 
    plt.ylim((-17,15.1)) 
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{snm}/A}[\rm{MeV}]$",fontsize=15)
    plot_path = 'snm_pnm_with_GP.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')

#####################################################
### plot pnm snm and their first and second derivative
#####################################################
def plot_6(train_x, train_y_1,train_y_2,dens_list,pnm,pnm_cov,d_pnm,d_pnm_cov, dd_pnm,dd_pnm_cov,snm,snm_cov,d_snm, d_snm_cov, dd_snm, dd_snm_cov,dens_list_2,pnm_2,d_pnm_2,dd_pnm_2,snm_2,d_snm_2,dd_snm_2 ):
    print("start plotting 5")
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(14,10))
    plt.subplots_adjust(wspace = 0.2, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(221)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    x_list_1     = dens_list.ravel()
    x_list_2     = dens_list_2.ravel()

    y_list_1     = pnm.ravel()
    y_list_2     = d_pnm.ravel()*0.1
    y_list_3     = dd_pnm.ravel()*0.01
  
    y_list_11    = pnm_2
    y_list_22    = d_pnm_2*0.1
    y_list_33    = dd_pnm_2*0.01

    confidence_1 = np.sqrt(np.diag(pnm_cov))
    confidence_2 = np.sqrt(np.diag(d_pnm_cov))*0.1
    confidence_3 = np.sqrt(np.diag(dd_pnm_cov))*0.01

    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.fill_between(x_list_1, y_list_1 + 2*confidence_1, y_list_1 - 2*confidence_1, alpha=0.1,color = 'blue',zorder = 1)
    plt.fill_between(x_list_1, y_list_1 + confidence_1, y_list_1 - confidence_1, alpha=0.3,color = 'blue',zorder = 1)
    plt.plot(x_list_1, y_list_1, label="GP",color='blue',zorder = 2)
    plt.plot(x_list_2, y_list_11, label="interpolation",ls=':', color = 'cornflowerblue',zorder = 3)
    plt.fill_between(x_list_1, y_list_2 + 2*confidence_2, y_list_2 - 2*confidence_2, alpha=0.1,color = 'darkorange',zorder = 1)
    plt.fill_between(x_list_1, y_list_2 + confidence_2, y_list_2 - confidence_2, alpha=0.3,color = 'darkorange',zorder = 1)
    plt.plot(x_list_1, y_list_2, label="first derivative * 0.1",color='darkorange',zorder= 2)
    plt.plot(x_list_2[1::], y_list_22, label="first derivative * 0.1",ls=':',color = 'red',zorder = 3)

    plt.fill_between(x_list_1, y_list_3 + 2*confidence_3, y_list_3 - 2*confidence_3, alpha=0.1,color = "green",zorder=1)
    plt.fill_between(x_list_1, y_list_3 + confidence_3, y_list_3 - confidence_3, alpha=0.3,color = "green",zorder=1)
    plt.plot(x_list_1, y_list_3, label="second derivative * 0.01",color = 'green',zorder =2 )
    plt.plot(x_list_2[2::], y_list_33, label="second derivative * 0.01",ls='-.',color='yellowgreen',zorder = 3)

    plt.scatter(train_x, train_y_1, label="train", c="black", marker="x",zorder = 5)
    plt.legend(fontsize=8)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xlim((0.12,0.2001)) 
    plt.ylim((-1,21)) 
    plt.xticks([])

### plot snm
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(223)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    y_list_4     = snm.ravel()
    y_list_5     = d_snm.ravel()*0.1
    y_list_6     = dd_snm.ravel()*0.01

    y_list_44    = snm_2
    y_list_55    = d_snm_2*0.1
    y_list_66    = dd_snm_2*0.01


    confidence_4 = np.sqrt(np.diag(snm_cov))
    confidence_5 = np.sqrt(np.diag(d_snm_cov))*0.1
    confidence_6 = np.sqrt(np.diag(dd_snm_cov))*0.01

    plt.hlines(0,0.12,0.20,ls=':',lw = 0.5, alpha = 0.3)
    plt.vlines(0.1629,-15.387,0,ls=':',lw = 0.5, alpha = 0.3)

#    plt.fill_between(x_list_1, y_list_4 + confidence_4, y_list_4 - confidence_4, alpha=0.1)
#    plt.plot(x_list_1, y_list_4, label="GP")
#    plt.fill_between(x_list_1, y_list_5 + confidence_5, y_list_5 - confidence_5, alpha=0.1)
#    plt.plot(x_list_1, y_list_5, label="first derivative")
#
#    plt.fill_between(x_list_1, y_list_6 + confidence_6, y_list_6 - confidence_6, alpha=0.1)
#    plt.plot(x_list_1, y_list_6, label="second derivative")
#    plt.scatter(train_x, train_y_2, label="train", c="black", marker="x",zorder = 5)


    plt.fill_between(x_list_1, y_list_4 + 2* confidence_4, y_list_4 -2* confidence_4, alpha=0.1,color = 'blue',zorder = 1)
    plt.fill_between(x_list_1, y_list_4 + confidence_4, y_list_4 - confidence_4, alpha=0.3,color = 'blue',zorder = 1)
    plt.plot(x_list_1, y_list_4, label="GP",color='blue',zorder = 2)
    plt.plot(x_list_2, y_list_44, label="interpolation",ls=':', color = 'cornflowerblue',zorder = 3)
    plt.fill_between(x_list_1, y_list_5 + 2*confidence_5, y_list_5 - 2*confidence_5, alpha=0.1,color = 'darkorange',zorder = 1)
    plt.fill_between(x_list_1, y_list_5 + confidence_5, y_list_5 - confidence_5, alpha=0.3,color = 'darkorange',zorder = 1)
    plt.plot(x_list_1, y_list_5, label="first derivative * 0.1",color='darkorange',zorder= 2)
    plt.plot(x_list_2[1::], y_list_55, label="first derivative * 0.1",ls=':',color = 'red',zorder = 3)

    plt.fill_between(x_list_1, y_list_6 + 2*confidence_6, y_list_6 - 2*confidence_6, alpha=0.1,color = "green",zorder=1)
    plt.fill_between(x_list_1, y_list_6 + confidence_6, y_list_6 - confidence_6, alpha=0.3,color = "green",zorder=1)
    plt.plot(x_list_1, y_list_6, label="second derivative * 0.01",color = 'green',zorder =2 )
    plt.plot(x_list_2[2::], y_list_66, label="second derivative * 0.01",ls='-.',color='yellowgreen',zorder = 3)




    plt.xlim((0.12,0.2001)) 
    plt.ylim((-17,15.1)) 
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{snm}/A}[\rm{MeV}]$",fontsize=15)
#    plot_path = 'snm_pnm_GP_vs_interpolation.pdf'
#    plt.savefig(plot_path,bbox_inches='tight')
#    plt.close('all')



#    fig2 = plt.figure('fig2')
#    plt.figure(figsize=(6,10))
#    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(222)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    x_list_1     = dens_list.ravel()

    y_list_1     = abs((pnm.ravel() - pnm_2.ravel())/pnm.ravel())
    y_list_2     = abs((d_pnm[1::].ravel() - d_pnm_2.ravel())/d_pnm[1::].ravel())
    y_list_3     = abs((dd_pnm[2::].ravel() - dd_pnm_2.ravel())/dd_pnm[2::].ravel())
  
    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.plot(x_list_1, y_list_1, label="zeroth derivative",color='blue',ls="-",zorder = 2)
    plt.plot(x_list_1[1::], y_list_2, label="first derivative",color='darkorange',ls="-.",zorder = 2)
    plt.plot(x_list_1[2::], y_list_3, label="second derivative",color='green',ls=":",zorder = 2)

    plt.legend(fontsize=8)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel("relative error [pnm]",fontsize=15)
    plt.xlim((0.12,0.2001)) 
    #plt.ylim((-1,21)) 
    plt.xticks([])


    ax1 = plt.subplot(224)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    x_list_1     = dens_list.ravel()

    y_list_4     = abs((snm.ravel() - snm_2.ravel())/snm.ravel())
    y_list_5     = abs((d_snm[1::].ravel() - d_snm_2.ravel())/d_snm[1::].ravel())

    y_list_6     = abs((dd_snm[2::].ravel() - dd_snm_2.ravel())/dd_snm[2::].ravel())
  
    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.plot(x_list_1, y_list_4, label="zeroth derivative",color='blue',ls="-",zorder = 2)
    #plt.plot(x_list_1[1::], y_list_5, label="first derivative",color='darkorange',ls="-.",zorder = 2)
    plt.plot(x_list_1[2::], y_list_6, label="second derivative",color='green',ls=":",zorder = 2)

    plt.legend(fontsize=8)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel("relative error [snm]",fontsize=15)
    plt.xlim((0.12,0.2001)) 
    #plt.ylim((-0.025,0.07))

    plot_path = 'snm_pnm_GP_vs_interpolation_test.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')

######################################################                                                                 
######################################################
###  plot
######################################################
######################################################
def plot_8():

    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(212)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.fill_between(test_x.ravel(), test_y_1 + confidence_1, test_y_1 - confidence_1, alpha=0.1)
    plt.plot(test_x, test_y_1, label="GP")
    #plt.fill_between(test_x.ravel(), test_dy_1 + confidence_1_dy, test_dy_1 - confidence_1_dy, alpha=0.1)
    #plt.plot(test_x, test_dy_1, label="GP")
    plt.scatter(train_x, train_y_1, label="train", c="red", marker="x")
    #plt.legend(fontsize=15)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{snm}/A}[\rm{MeV}]$",fontsize=15)
    #plot_path = 'snm_gp_test.pdf'
    #plt.savefig(plot_path,bbox_inches='tight')
    #plt.close('all')

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)


    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    plt.fill_between(test_x.ravel(), test_y_2 + confidence_2, test_y_2 - confidence_2, alpha=0.1)
    plt.plot(test_x, test_y_2, label="GP")
    #plt.fill_between(test_x.ravel(), test_dy_2 + confidence_2_dy, test_dy_2 - confidence_2_dy, alpha=0.1)
    #plt.plot(test_x, test_dy_2, label="GP")                                                                   
    plt.scatter(train_x, train_y_2, label="train", c="red", marker="x")
    plt.legend(fontsize=15)
    #plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xticks([])
    plot_path = 'snm_pnm.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')

######################################################                                                                 
######################################################
###  plot
######################################################
######################################################
def plot_9(train_x,train_y_1,train_y_2,test_x,test_y_1,test_y_2):
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))                    
    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)

    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    #plt.fill_between(test_x.ravel(), test_y_1 + confidence_1, test_y_1 - confidence_1, alpha=0.1)
    plt.plot(test_x, test_y_1, label="GP")
    #plt.fill_between(test_x.ravel(), test_dy_1 + confidence_1_dy, test_dy_1 - confidence_1_dy, alpha=0.1)
    #plt.plot(test_x, test_dy_1, label="GP")
    plt.scatter(train_x, train_y_1, label="Emulator", c="red", marker="x")
    #plt.legend(fontsize=15)

    plt.legend(fontsize=15,loc= "upper left")
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{snm}/A}[\rm{MeV}]$",fontsize=15)
    #plt.xlim((0.115,0.265))
    #plt.ylim((-15.5,-7.5))
    #plot_path = 'snm_gp_test.pdf'
    #plt.savefig(plot_path,bbox_inches='tight')
    #plt.close('all')

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax1 = plt.subplot(212)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)


    #plt.title("l=%.2f sigma=%.2f" % (gpr.length, gpr.sigma))
    #plt.fill_between(test_x.ravel(), test_y_2 + confidence_2, test_y_2 - confidence_2, alpha=0.1)
    plt.plot(test_x, test_y_2, label="GP")
    #plt.fill_between(test_x.ravel(), test_dy_2 + confidence_2_dy, test_dy_2 - confidence_2_dy, alpha=0.1)
    #plt.plot(test_x, test_dy_2, label="GP")
    plt.scatter(train_x, train_y_2, label="Emulator", c="red", marker="x")
    plt.legend(fontsize=15,loc= "upper left")
    #plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xticks([])
    #plt.xlim((0.115,0.265))
    #plt.ylim((10,32.4))
    plot_path = 'snm_pnm.pdf'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.close('all')

#####################################################
#####################################################
#####################################################
def plot_10(list_1,list_2,list_3,list_4):
    fontsize_legend = 12
    fontsize_x_label = 15
    fontsize_y_label = 15
    fig1 = plt.figure('fig1')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    y_min = 0
    y_max = 0.04
    x_min = 0.05
    x_max = 0.25
    x_gap = 0.05
    #regulator = (x_max-x_min)/(y_max-y_min)

    x_list = list_3
    y_list = list_1*list_2
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_2),2))
    for loop1 in range(0,len(list_2)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=15, linewidth=1, marker=".",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163, -15.386,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    #plt.xlim((0.11,0.225))
    #plt.ylim((-18.1,-11.9))
    plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    #plt.yticks(np.arange(lower_range,uper_range+0.0001,gap),fontsize = 10)

    #plt.ylim((-21,-5))
    plt.xlim((x_min,x_max))
    plt.xlabel(r"$R_{\rm{skin}} \ [\rm{fm}]$",fontsize=fontsize_y_label)
    plt.ylabel(r"$\alpha_D S \ [\rm{MeV} \ \rm{fm}^{3}]$",fontsize=fontsize_x_label)

    plot_path = 'Pb208_34_samples_alphaD_Rs_CC.pdf'
    plt.savefig(plot_path)
    plt.close('all')

####################################################
    fig2 = plt.figure('fig2')
#    plt.figure(figsize=(5,10))
#    plt.subplots_adjust(wspace =0.3, hspace =0.4)

#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
#    ax = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #ax.set_title("saturation density")

#   range ajustment
    #regulator = (x_max-x_min)/(y_max-y_min)
    x_list = list_4
    y_list = list_1*list_2
#    l = plt.scatter(x_list,y_list,color='crimson',s = 20, marker = '    o')

    sns.set(color_codes=True)

    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = x_list[loop1]
        z[loop1,1] = y_list[loop1]
    #data = z[np.where((z[:,0]>x_min)&(z[:,0]<x_max)&(z[:,1]<y_max))]
    data = z
    #data = z
    #print("z(3,1)= "+str(z[3,2]))
    #input()
    #print("x="+str(x_list))
    #print("y="+str(y_list))
    #print(data)
    #input()

    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
    g.plot_joint(plt.scatter, c="m", s=15, linewidth=1, marker=".",label = r"$\rm{HM \ (34\ samples)}$")
    g.ax_joint.collections[0].set_alpha(0)
    #g.set_axis_labels("$X$", "$Y$")
    #g.ax_joint.legend_.remove()
    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #l2 = plt.scatter (0.163,31.5 ,color = 'red' ,marker = 'o',zorder=5,label = r"$\rm{DNNLO}_{\rm{GO}}(394)$")
    plt.xlim((x_min,x_max))
    plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$R_{\rm{skin}} \ [\rm{fm}]$",fontsize=fontsize_y_label)
    plt.ylabel(r"$\alpha_D S \ [\rm{MeV} \  \rm{fm}^{3}]$",fontsize=fontsize_x_label)


    plot_path = 'Pb208_34_samples_alphaD_Rs_IMSRG.pdf'
    plt.savefig(plot_path)

   

def plot_11():
    density_count = 8
    validation_count = 34
    ccd_pnm_batch_all = np.zeros((validation_count,density_count))
    ccd_snm_batch_all = np.zeros((validation_count,density_count))
    density_batch_all = np.zeros((validation_count,density_count))

    database_dir = "/home/slime/subspace_CC/test/emulator/DNNLO394/christian_34points/"
    for loop1 in range(density_count):
        dens = 0.06 + loop1 * 0.02
        input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt.out" % ('pnm',66,dens)
        ccd_pnm_batch_all[:,loop1] = read_ccd_data(input_dir = input_dir, data_count = validation_count )/66   
        
        input_dir = database_dir + "%s_%d_%.2f_DNNLO_christian_34points/ccdt_n3.out" % ('snm',132,dens)
        ccd_snm_batch_all[:,loop1] = read_ccd_data(input_dir = input_dir, data_count = validation_count )/132   
        density_batch_all[:,loop1]  = dens         

### plot
    fontsize_x_label = 10
    fontsize_y_label = 10
    fontsize_legend  = 10
    markersize       = 2
    label = []
    for loop in range(34):
        if loop == 0:
            label.append("with truncation error")
        else:
            label.append("")

    fig1 = plt.figure('fig1')
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(wspace =0.3, hspace =0.4)
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(221)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
#   range ajustment
    #regulator = (x_max-x_min)/(y_max-y_min)
    #x_list = density_batch_all[0]
    #y_list = ccd_pnm_batch_all[0]
    #l = plt.plot(x_list,y_list,color='blue',lw = 0.5, marker = '.',markersize=markersize,label="HM (34 samples)")
    for loop in range(validation_count):
        x_list = density_batch_all[loop]
        y_list = ccd_pnm_batch_all[loop]
        l = plt.plot(x_list,y_list,color='blue',lw = 0.5, marker = '.',markersize=markersize,label = label[loop])

    plt.legend(loc='upper left',fontsize = fontsize_legend)
    #plt.xlim((x_min,x_max))
    #plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$E_{\rm{pnm}}/A \ [\rm{MeV}]$",fontsize=fontsize_y_label)

    ax = plt.subplot(222)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
#   range ajustment
    #regulator = (x_max-x_min)/(y_max-y_min)

    for loop in range(validation_count):
        x_list = density_batch_all[loop]
        y_list = ccd_snm_batch_all[loop]
        l = plt.plot(x_list,y_list,color='blue',lw = 0.5, marker = '.',markersize=markersize ,label = label[loop] )

    plt.legend(loc='upper right',fontsize = fontsize_legend)
    #plt.xlim((x_min,x_max))
    #plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$E_{\rm{snm}}/A \ [\rm{MeV}]$",fontsize=fontsize_y_label)

    ax = plt.subplot(223)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    #regulator = (x_max-x_min)/(y_max-y_min)
    for loop in range(validation_count):
        saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = generate_NM_observable(ccd_pnm_batch_all[loop],ccd_snm_batch_all[loop],density_batch_all[loop],switch="interpolation")
        x_list = np.arange(density_batch_all[loop].min(),density_batch_all[loop].max(),0.0001).reshape(-1,1)#/saturation_density
        #print(x_list)
        #print(x_list*x_list)
        y_list = raw_data[10] 
        #y_list = raw_data[18]
        l = plt.plot(x_list,y_list,color='green',lw = 0.5, alpha = 0.8,label = label[loop])
        #print("saturation_density="+str(saturation_density))
    plt.legend(loc='upper left',fontsize = fontsize_legend)
    #plt.xlim((x_min,x_max))
    #plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$\rho $",fontsize=fontsize_x_label)
    plt.ylabel(r"$S(\rho) \ [\rm{MeV}]$",fontsize=fontsize_y_label)



    ax = plt.subplot(224)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    for loop in range(validation_count):
        saturation_density, saturation_energy, symmetry_energy,L,K,raw_data = generate_NM_observable(ccd_pnm_batch_all[loop],ccd_snm_batch_all[loop],density_batch_all[loop],switch="GP")
        x_list = np.arange(density_batch_all[loop].min(),density_batch_all[loop].max(),0.0001).reshape(-1,1)/saturation_density
        #y_list = raw_data[16].T[0]

        y_list = raw_data[16]
        l = plt.plot(x_list,y_list,color='green',lw = 0.1, alpha = 0.4 ,label = label[loop])

    plt.legend(loc='upper left',fontsize = fontsize_legend)
    plt.xlabel(r"$\rho / \rho_{0}$",fontsize=fontsize_x_label)
    plt.ylabel(r"$K(\rho) \ [\rm{MeV}]$",fontsize=fontsize_y_label)

    plot_path = 'Pb208_34_samples_EOS_test.pdf'
    plt.savefig(plot_path)

def plot_12(list_x,list_y_1,list_y_2,list_y_3,list_y_1cov,list_y_2cov,list_y_3cov,list_x_2,list_y_11,list_y_22,list_y_33,matter_type,observable_type): ## plot GP cn
    fontsize_x_label = 20
    fontsize_y_label = 20
    fontsize_legend  = 20
    markersize       = 2

    fig1 = plt.figure('fig1')
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(wspace =0.3, hspace =0.4)
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

   
    x_list = list_x
    y_list = list_y_1
    #confidence = 2*np.sqrt(np.diag(list_y_1cov))
    #plt.fill_between(x_list, y_list + confidence, y_list - confidence,color='orange', alpha=0.1)
    #plt.fill_between(x_list, y_list , y_list - 1, alpha=0.1)
    l = plt.scatter(list_x_2,list_y_11,s = 20,color='orange', marker = 'o')
    print(list_x_2)
    print(list_y_11)
    l = plt.plot(x_list,y_list,color='orange',lw = 2, marker = '',markersize=markersize,label = "%s0" %(observable_type))
 
    x_list = list_x
    y_list = list_y_2
    #confidence = 2*np.sqrt(np.diag(list_y_2cov))
    l = plt.scatter(list_x_2,list_y_22,s = 20,color='green',marker = 'o')
    l = plt.plot(x_list,y_list,color='green',lw = 2, marker = '',markersize=markersize,label = "%s1" %(observable_type))
    #plt.fill_between(x_list, y_list + confidence, y_list - confidence,color='green', alpha=0.1)

    x_list = list_x
    y_list = list_y_3
    #confidence = 2*np.sqrt(np.diag(list_y_3cov))
    l = plt.scatter(list_x_2,list_y_33,s = 20,color='blue', marker = 'o')
    l = plt.plot(x_list,y_list,color='blue',lw = 2, marker = '',markersize=markersize,label =  "%s2" %(observable_type))
    #plt.fill_between(x_list, y_list + confidence, y_list - confidence,color='blue', alpha=0.1)
    #plt.ylim((-2.5,2.5))
    plt.xlim((0.06,0.38))
    plt.hlines(0,0,0.40,ls='-',lw = 1, alpha = 1)
    plt.legend(loc='upper left',fontsize = fontsize_legend)
    plt.xlabel(r"$ \rm{Density} \ \it{\rho} \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{Energy} \ \rm{per} \ \rm{particle} \ \it{E/A}$",fontsize=fontsize_y_label)

    plot_path = 'Observable_coefficients_%s_%s.pdf' % (observable_type,matter_type)
    plt.savefig(plot_path)

 
def plot_13(list_1,list_2,list_3,list_4): 
    fontsize_x_label = 15
    fontsize_y_label = 15
    fontsize_legend  = 12
    markersize       = 2

    fig1 = plt.figure('fig1')
    #plt.figure(figsize=(10,8))
    #plt.subplots_adjust(wspace =0.3, hspace =0.4)
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(111)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)

    #l = plt.scatter(list_3,list_4,color='blue',s= 2, marker = '.',label = "with error",zorder= 1)

    sns.set(color_codes=True)
    z = np.zeros((len(list_3),2))
    for loop1 in range(0,len(list_3)):
        z[loop1,0] = list_3[loop1]
        z[loop1,1] = list_4[loop1]
    data = z
    df = pd.DataFrame(data, columns=["x", "y"])
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1],zorder = 1)
    g.plot_joint(plt.scatter, c="m", s=0.1, linewidth=1, marker="",label = "samples from cross covariance",zorder = 1)
    g.ax_joint.collections[0].set_alpha(0)
    plt.legend(loc='upper right',fontsize = fontsize_legend)

    l = plt.scatter(list_1,list_2,s = 20,color='red', marker = 'o',zorder = 2)

    plt.xlim((0.12,0.22))
    plt.ylim((-22,-10))
    plt.xlabel(r"$\rm{saturation \ density} \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$\rm{saturation \ energy} \ [\rm{MeV}]$",fontsize=fontsize_y_label)
    plot_path = 'saturaion_point_with_all_error_test.pdf'
    plt.savefig(plot_path)
    plt.close('all')

def plot_14( density_,pnm_,snm_,density_batch,pnm_batch,snm_batch):
    n = len(pnm_batch)
    print(n) 
### plot
    fontsize_x_label = 20
    fontsize_y_label = 20
    fontsize_legend  = 20
    markersize       = 10
    label = []
    for loop in range(n):
        if loop == 0:
            label.append("")
        else:
            label.append("")

    sns.set_style("white")
    colors = ["blue", "crimson", "green", "gold", "violet"]
#    sns.set_palette(cmap)
    fig1 = plt.figure('fig1')
    plt.figure(figsize=(10,14))
    plt.subplots_adjust(wspace =0.3, hspace =0.2)
#####################################################
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax = plt.subplot(211)
    ax.grid(False)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=1,color="k")
#   range ajustment
    #regulator = (x_max-x_min)/(y_max-y_min)
    #x_list = density_batch_all[0]
    #y_list = ccd_pnm_batch_all[0]
    #l = plt.plot(x_list,y_list,color='blue',lw = 0.5, marker = '.',markersize=markersize,label="HM (34 samples)")
    x_list = density_
    y_list = pnm_
    l = plt.plot(x_list,y_list,color='k',lw = 2, marker = 's',markersize=markersize,label="EOS without error")
    for loop in range(n):
        x_list = density_batch[loop]
        y_list = pnm_batch[loop]
        l = plt.plot(x_list,y_list,color=colors[loop],lw = 2, marker = '',markersize=markersize,label = label[loop])

    plt.legend(loc='upper left',fontsize = fontsize_legend)
    #plt.xlim((x_min,x_max))
    #plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$E_{\rm{pnm}}/A \ [\rm{MeV}]$",fontsize=fontsize_y_label)


    ax = plt.subplot(212)
    ax.grid(False)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2,color="k")

    x_list = density_
    y_list = snm_
    l = plt.plot(x_list,y_list,color='k',lw = 2, marker = 's',markersize=markersize,label= "EOS without error")

    for loop in range(n):
        x_list = density_batch[loop]
        y_list = snm_batch[loop]
        l = plt.plot(x_list,y_list,color=colors[loop],lw = 2, marker = '',markersize=markersize,label = label[loop])

    plt.legend(loc='upper left',fontsize = fontsize_legend)
    #plt.xlim((x_min,x_max))
    #plt.xticks(np.arange(x_min,x_max+0.0001,x_gap),fontsize = 10)
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$E_{\rm{snm}}/A \ [\rm{MeV}]$",fontsize=fontsize_y_label)

    plot_path = 'EOS_with_error.pdf' 
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')

#####################################################
#####################################################
#####################################################
def plot_confidence_EOS(density_series,pnm_,snm_,pnm_batch,snm_batch,q):
    sns.set_style("white")

    fig1 = plt.figure('fig1')
    plt.figure(figsize=(6,10))
    plt.subplots_adjust(wspace =0, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(211)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.grid(False)

    x_list   =  density_series
    y_list_1 =  np.zeros(len(density_series))
    y_list_2 =  np.zeros(len(density_series))
#   
    for loop in range(len(x_list)):
        y_list_temp = hdi(pnm_batch[:,loop], hdi_prob=q)
        y_list_1[loop] = y_list_temp[0] 
        y_list_2[loop] = y_list_temp[1] 

    plt.fill_between(x_list, y_list_1 , y_list_2, alpha=0.3, label="68%")
    plt.plot(x_list, pnm_, marker = "s",label="interaction #20")

    ax1.legend()
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xlim((0.10,0.1801)) 
    plt.ylim((0,30)) 
    ax1.set_xticklabels([])
    #plt.xticks([])

### plot snm
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
    ax2 = plt.subplot(212)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    ax2.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.grid(False)
    for loop in range(len(x_list)):
        y_list_temp = hdi(snm_batch[:,loop], hdi_prob=q)
        y_list_1[loop] = y_list_temp[0] 
        y_list_2[loop] = y_list_temp[1] 


    plt.fill_between(x_list, y_list_1 , y_list_2, alpha=0.3,label="68%")
    plt.plot(x_list, snm_, marker = "s",label="interaction #20")

    ax2.legend()
    plt.xlabel(r"$\rho [\rm{fm}^{-3}]$",fontsize=15)
    plt.ylabel(r"$\rm{E_{pnm}/A}[\rm{MeV}]$",fontsize=15)
    plt.xlim((0.10,0.1801)) 
    plt.ylim((-20,-7)) 
    #plt.xticks([])
    plot_path = 'EOS_confidence_region_2.pdf' 
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')

    sns.set()
    fig2 = plt.figure('fig2')
    plt.figure(figsize=(13,5))
    plt.subplots_adjust(wspace =0.1, hspace =0)
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'
### plot pnm
    ax1 = plt.subplot(121)
    print(len(pnm_batch))

    y_list_temp=[]
    density_new = [0.11]
    for loop in range(len(pnm_batch)):
        spl  = interpolate.UnivariateSpline(x_list,pnm_batch[loop,:],k=4,s=0)
        y_   = spl(density_new)
        y_list_temp.append(y_[0])        
    
     
    sns.distplot(y_list_temp)    
    standard_deviation=np.std(y_list_temp)
    ax1.set_title("pnm(%.2f fm^3): standard deviation %.4f" %(density_new[0],standard_deviation),fontsize=12)
    plt.ylabel('')

    ax2 = plt.subplot(122)
    print(len(pnm_batch))

    y_list_temp=[]
    for loop in range(len(pnm_batch)):
        spl  = interpolate.UnivariateSpline(x_list,snm_batch[loop,:],k=4,s=0)
        y_   = spl(density_new)
        y_list_temp.append(y_[0])        
    
     
    sns.distplot(y_list_temp)    
    standard_deviation=np.std(y_list_temp)
    ax2.set_title("snm(%.2f fm^3): standard deviation %.4f" %(density_new[0],standard_deviation),fontsize=12)
    plt.ylabel('')
    plot_path = 'EOS_given_density_.pdf' 
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')



#        y_list_1[loop] = y_list_temp[0] 
#        y_list_2[loop] = y_list_temp[1] 





#####################################################
#####################################################
#####################################################
def plot_confidence_ellipse(x,y):
    fontsize_x_label = 20
    fontsize_y_label = 20
    fontsize_legend  = 20

    sns.set_style("white")
    cmap = sns.cubehelix_palette(8)

    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['ytick.direction'] = 'in'

    fig, ax_nstd = plt.subplots(figsize=(6, 6))
    ax_nstd.grid(False)
    ax_nstd.spines['top'].set_linewidth(2)
    ax_nstd.spines['bottom'].set_linewidth(2)
    ax_nstd.spines['left'].set_linewidth(2)
    ax_nstd.spines['right'].set_linewidth(2)
    plt.tick_params(top=True,bottom=True,left=True,right=True,width=1,color="k")
    
    ax_nstd.scatter(x, y, s=0.1,color = cmap[0],alpha = 0.4)
    
    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label='68%', edgecolor= cmap[3],linestyle='-.',lw = 2,alpha = 0.9)
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label='95%', edgecolor=cmap[1] ,linestyle='--',lw = 2, alpha = 0.9)
    #confidence_ellipse(x, y, ax_nstd, n_std=3,
    #                   label=r'$3\sigma$', edgecolor='royalblue', linestyle=':', alpha = 0.8)
    rect = Rectangle((0.15,-16.5),0.02,1,linewidth=1.5,edgecolor='k',facecolor = 'grey',alpha = 0.3)
    ax_nstd.add_patch(rect) 
    #ax_nstd.scatter(mu[0], mu[1], c='red', s=3)
    ax_nstd.legend()
    plt.xlabel(r"$\rho \ [\rm{fm}^{-3}]$",fontsize=fontsize_x_label)
    plt.ylabel(r"$E/A \ [\rm{MeV}]$",fontsize=fontsize_y_label)
    plt.xlim((0.12,0.2001))
    plt.ylim((-20,-12))


    plot_path = 'saturation_point_credible_interval.pdf' 
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')




def plot_corner_plot(df_plot):
    obs_label = {'EA48Ca':r'$E/A(^{48}\mathrm{Ca})$ [MeV]',\
                    'E2+48Ca':r'$E_{2^+}(^{48}\mathrm{Ca})$ [MeV]',\
                    'EA208Pb':r'$E/A(^{208}\mathrm{Pb})$ [MeV]',\
                    'E3-208Pb':r'$E_{3^-}(^{208}\mathrm{Pb})$ [MeV]',\
                    'Rp208Pb':r'$R_\mathrm{pt-p}(^{208}\mathrm{Pb})$ [fm]',\
                    'Rp48Ca':r'$R_\mathrm{pt-p}(^{48}\mathrm{Ca})$ [fm]',\
                    'EA_NM':r'$E_0/A$ [MeV]',\
                    'rho_NM':r'$\rho_0$ [fm$^{-3}$]',\
                    'Rskin48Ca':r'$R_\mathrm{skin}(^{48}\mathrm{Ca})$ [fm]',\
                    'Rskin208Pb':r'$R_\mathrm{skin}(^{208}\mathrm{Pb})$ [fm]',\
                    'aD208Pb':r'$\alpha_D(^{208}\mathrm{Pb})$ [fm$^3$]',\
                    'S_NM':r'$S$ [MeV]',\
                    'L_NM':r'$L$ [MeV]',\
                    'K_NM':r'$K$ [MeV]',\
                    'E2H':r'$E(^{2}\mathrm{H})$ [MeV]',\
                    'Rp2H':r'$R_p(^{2}\mathrm{H})$ [fm]',\
                    'Q2H':r'$Q(^{2}\mathrm{H})$ [e fm$^2$]',\
                    'E3H':r'$E(^{3}\mathrm{H})$ [MeV]',\
                    'E4He':r'$E(^{4}\mathrm{He})$ [MeV]',\
                    'Rp4He':r'$R_p(^{4}\mathrm{He})$ [fm]',\
                    'E16O':r'$E(^{16}\mathrm{O})$ [MeV]',\
                    'Rp16O':r'$R_p(^{16}\mathrm{O})$ [fm]'}
    obs_exp={}
    # ---
    # 48Ca
    # ---
    # Exp. refs missing
    obs_exp['EA48Ca']={'val':-416/48, 'err':None, 'ref': None, 'author': None, 'abbrev': None, 'exp': True}
    obs_exp['E2+48Ca']={'val':3.83, 'err':None, 'ref': None, 'author': None, 'abbrev': None, 'exp': True}
    # Rch = 3.477(2), ADNDT 99 (2013) 69-95.
    # Translated to point-proton radius via scripts/radius.py
    obs_exp['Rp48Ca']={'val':3.393, 'err':None, 'ref': 'ADNDT 99 (2013) 69-95', \
                       'author': None, 'abbrev': 'ADNDT', 'exp': True}
    # Hagen et al, Nat. Phys. 12, 186-190 (2016)
    obs_exp['Rskin48Ca']={'val':0.135, 'err':0.015, 'ref': 'Nat. Phys. 12, 186-190 (2016)', \
                          'author': 'Hagen et al.', 'abbrev': 'Hagen', 'exp': False}
    # ---
    # 208Pb
    # ---
    # Exp. refs missing
    obs_exp['EA208Pb']={'val':-1636.4/208, 'err':None, 'ref': None, 'author': None, 'abbrev': None, 'exp': True}
    obs_exp['E3-208Pb']={'val':2.614, 'err':None, 'ref': None, 'author': None, 'abbrev': None, 'exp': True}
    #
    # Rch = 5.5012 +/- 0.0013, ADNDT 99 (2013) 69-95.
    # Translated to point-proton radius via scripts/radius.py
    obs_exp['Rp208Pb']={'val':5.4498, 'err':None, 'ref': 'ADNDT 99 (2013) 69-95', \
                       'author': None, 'abbrev': 'ADNDT', 'exp': True}
    # alpha_D = 20.1 +/- 0.6 fm^3, A. Tamii et al, Phys. Rev. Lett. 107, 062502 (2011).
    obs_exp['aD208Pb']={'val':20.1, 'err':0.6, 'ref': 'Phys. Rev. Lett. 107, 062502 (2011).', \
                       'author': 'Tamii et al', 'abbrev': 'Tamii', 'exp': True}
    # ---
    # NM
    # ---
    # Bender et al, Rev. Mod. Phys. 75, 121–180 (2003).
    # Hebeler, et al, Phys. Rev. C 83, 031301 (2011).
    # or E0 = −15.9±0.4 MeV, n0 = 0.164±0.007 fm−3 (Drischler et al. 2016)
    obs_exp['EA_NM']={'val':-16., 'err':0.5, 'ref': 'Rev. Mod. Phys. 75, 121–180 (2003)', \
                          'author': 'Bender et al.', 'abbrev': 'RMP(2003)', 'exp': False}
    obs_exp['rho_NM']={'val':0.16, 'err':0.01, 'ref': 'Rev. Mod. Phys. 75, 121–180 (2003)', \
                          'author': 'Bender et al.', 'abbrev': 'RMP(2003)', 'exp': False}
    # Lattimer & Lim (2013); Lattimer & Steiner (2014)
    obs_exp['S_NM']={'val':31., 'err':1., 'ref': 'Lattimer & Lim (2013); Lattimer & Steiner (2014)', \
                          'author': 'Lattimer et al.', 'abbrev': 'Lattimer', 'exp': False}
    obs_exp['L_NM']={'val':50., 'err':10., 'ref': 'Lattimer & Lim (2013); Lattimer & Steiner (2014)', \
                          'author': 'Lattimer et al.', 'abbrev': 'Lattimer', 'exp': False}
    # Shlomo et al. 2006; Piekarewicz 2010
    obs_exp['K_NM']={'val':240., 'err':20., 'ref': 'Shlomo et al. 2006; Piekarewicz 2010', \
                          'author': 'Shlomo et al.', 'abbrev': 'Shlomo', 'exp': False}
    
    
    # ---
    # A=2-16
    # ---
    # Values from PRX table.
    obs_exp['E2H']={'val':-2.2298, 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['Rp2H']={'val':np.sqrt(3.903), 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['Q2H']={'val':0.27, 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['E3H']={'val':-8.4818, 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['E4He']={'val':-28.2956, 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['Rp4He']={'val':np.sqrt(2.1176), 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['E16O']={'val':-127.62, 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}
    obs_exp['Rp16O']={'val':np.sqrt(6.66), 'err':None, 'ref': None, \
                        'author': None, 'abbrev': None, 'exp': True}

# Option 1
    weights = np.ones(len(df_plot))


### start corner plot
    #palette='hls'
    #g = sns.PairGrid(df_plot,corner=True, hue = "rho",diag_sharey=False,layout_pad=-0.5,aspect=1.05,)
    #g = sns.PairGrid(df_plot,corner=True,diag_sharey=False,layout_pad=-0.5,aspect=1.05,)
    g = sns.PairGrid(df_plot,corner=True,diag_sharey=False,layout_pad=-0.5,aspect=1.05,)
    g.map_lower(sns.histplot, fill=True, alpha=0.8, weights=weights, bins=20)
    #g.map_lower(sns.scatterplot, alpha=0.8, size=df_plot["set_num"])
    #g.map_diag(sns.histplot, kde=True, weights=weights, bins=20);
    g.map_diag(sns.kdeplot);
    g.add_legend() 

 
    for irow,row_obs in enumerate(df_plot.columns[0:4]):
        # Extract exp value
        try: 
            obs_dict = obs_exp[row_obs]
            val = obs_dict['val']
            err = obs_dict['err']
            if err is None: row_val = [val]
            else: row_val = [val-err, val+err]
        except KeyError: row_val=[]
        for icol,col_obs in enumerate(df_plot.columns[0:4]):
            ax = g.axes[irow,icol]
            # Check if axis is in upper triangle
            if ax is None: continue
            # Use correct axis label
            if ax.is_first_col():
                try: ax.set_ylabel(obs_label[row_obs])
                except KeyError: ax.set_ylabel(row_obs)
            if ax.is_last_row():
                try: ax.set_xlabel(obs_label[col_obs])
                except KeyError: ax.set_xlabel(col_obs)
            # Extract exp value
            try: 
                obs_dict = obs_exp[col_obs]
                val = obs_dict['val']
                err = obs_dict['err']
                if err is None: col_val = [val]
                else: col_val = [val-err, val+err]
            except KeyError: col_val=[]
            # Plot exp values
            if icol==irow:
                if len(row_val)==2:
                    ax.axvline(row_val[0],color='r',alpha=0.5)
                    ax.axvline(row_val[1],color='r',alpha=0.5)
                    ax.axvspan(*row_val,color='r',alpha=0.1)
                elif len(row_val)==1:
                    ax.axvline(row_val[0],color='r')
            else:
                if len(row_val)==2:
                    ax.axhline(row_val[0],color='r',alpha=0.5)
                    ax.axhline(row_val[1],color='r',alpha=0.5)
                    #if not len(col_val)==2:
                    ax.axhspan(*row_val,color='r',alpha=0.1)
                elif len(row_val)==1:
                    ax.axhline(row_val[0],color='r',alpha=0.5)
                if len(col_val)==2:
                    ax.axvline(col_val[0],color='r',alpha=0.5)
                    ax.axvline(col_val[1],color='r',alpha=0.5)
                    #if not len(row_val)==2:
                    ax.axvspan(*col_val,color='r',alpha=0.1)
                elif len(col_val)==1:
                    ax.axvline(col_val[0],color='r',alpha=0.5)

    plot_path = '800k_sample_NM_corner_plot.pdf' 
    plt.savefig(plot_path, bbox_inches = 'tight')
    plt.close('all')



################################
## main
################################

#plot_11()

