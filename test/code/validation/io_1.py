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
import pandas as pd


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
        train_x = np.arange(0.12,0.12+dens_count*0.02,0.02)
        train_x = train_x.reshape(-1,1)
        train_y_1 = pnm_data
        train_y_2 = snm_data
        test_x  = np.arange(0.12,0.20,density_accuracy).reshape(-1,1)

        gpr = GP_test()
        gaussian_noise = 0.02

        gpr.fit_data(train_x, train_y_2, gaussian_noise)

        snm, snm_cov, d_snm, d_snm_cov,dd_snm,dd_snm_cov = gpr.predict(test_x)

        #test_y_1  = snm.ravel()
        #test_dy_1 = d_snm.ravel()
        #confidence_1    = 2 * np.sqrt(np.diag(snm_cov))
        #confidence_dy_1 = 2 * np.sqrt(np.diag(d_snm_cov))


        #density_range = test_x[np.where((snm[:]<(snm[iX]+confidence_1[iX]))&(snm[:]>(snm[iX]-confidence_1[iX])))]
        #print("saturation density: %.3f +/- %.3f" % (test_x[iX], 0.5*(np.max(density_range)-np.min(density_range))))
        #print("saturation energy:  %.3f +/- %.3f" % (snm[iX] , confidence_1[iX]))
        gpr = GP_test()
        gpr.fit_data(train_x, train_y_1, gaussian_noise)

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
        K  = 9* pow(saturation_density,2)*dd_snm[iX]

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
        #interpol_count = 1000
        spl_ccd_snm    = interpolate.UnivariateSpline(density_sequence,snm_data,k=2)
        spl_ccd_pnm    = interpolate.UnivariateSpline(density_sequence,pnm_data,k=2)
        #spldens        = np.linspace(density_sequence[0],density_sequence[len(density_sequence)-1],num=interpol_count)
        
        spldens  = np.arange(0.12,0.20,density_accuracy)


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
        L  = 3 * u[iX]*ds[iX]/du[iX]

        df = np.diff(snm) / np.diff(spldens)
        ddf= np.diff(df) / np.diff(spldens[1::])        
        K = 9* pow(saturation_density,2)*ddf[iX]


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
        saturation_density_batch[loop], saturation_energy_batch[loop], symmetry_energy_batch[loop],L_batch[loop], K_batch[loop]= generate_NM_observable(pnm_batch_all[loop,:],snm_batch_all[loop,:],density_sequence_all[loop,:],switch)

    return saturation_density_batch, saturation_energy_batch,symmetry_energy_batch,L_batch, K_batch
  
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
    g=sns.jointplot(x="x", y="y", data=df, kind="kde", color="g",bbox =[3,0.1])
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


################################
## main
################################



