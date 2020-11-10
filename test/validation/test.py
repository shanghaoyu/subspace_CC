import os
import numpy as np
#import matplotlib
#matplotlib.use('PS')
#import matplotlib.pyplot as plt
import math
import re
#import scipy.linalg as spla
#from scipy import interpolate

def read_nucl_matt_out(file_path):  # converge: flag = 1    converge: flag =0
    with open(file_path,'r') as f_1:
        converge_flag = int (1)
        count = len(open(file_path,'rU').readlines())
        #if ( count > 1500 ):
        #    converge_flag =int (0)
        data =  f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        ccd = 0
        ccdt = 0
        for loop1 in range(0,count):
            if ( re.search('#Nmax, Nucleons', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1 + 1])
                ccd = float(temp_1[6])
                ccdt = float(temp_1[7])
        return ccd,ccdt   #,converge_flag
        #print ('No "E/A" found in the file:'+file_path)
        #return float('nan')

density_count = 5
density_gap = 0.02
density_min = 0.12

file_path = "validation_data.txt"
with open(file_path,'a') as f_1:
    #f_1.write('density     ccd      ccd(T)  \n')
for loop1 in range(density_count):
    density = density_min + density_gap * loop1
    #file_path = "ccm_in_DNNLO450_old"
    #LEC = read_LEC(file_path)
    #LEC_random = generate_random_LEC(LEC, LEC_range)
    #print ("LEC="+str(LEC_random))
    #LEC_random = LEC
    file_path = './pnm_rho_%.2f.out' % (density)
    ccd_cal,ccdt_cal = read_nucl_matt_out(file_path)
    file_path = "EOS_different_density.txt"
    with open(file_path,'a') as f_1:
        f_1.write('% .2f    %.12f    %.12f  \n' % (density, ccd_cal, ccdt_cal))


