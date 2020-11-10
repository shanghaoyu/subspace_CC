import os
import numpy as np
import re

def extract_ccdt(path,dens):
    ccd  = np.zeros(34) 
    ccdt = np.zeros(34)
    for loop in range(34):
        file_path = path + "nuclear_matter_CCDT_deltaHM34_%s_132nuc_nmax4_dens0.%sfm-3.out"  % (str(loop),str(dens))
#        print(file_path)
        with open(file_path,'r') as f:
            count = len(open(file_path,'rU').readlines())
            data = f.readlines()
            wtf = re.match('#', 'abc',flags=0)
            for loop1 in range(0,count):
                if ( re.search('#Nmax, Nucleons, 3NFnexp, 3NFCutoff', data[loop1],flags=0) != wtf):
                    temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                    ccd[loop]  = float(temp_1[6])
                    ccdt[loop] = float(temp_1[7])
    return ccd, ccdt

# main 
for loop in range(5):
    dens = 12+loop*2
    path = "./ccdt_dens_%s/" %(str(dens))
    ccd, ccdt = extract_ccdt(path,dens )
    print(ccdt)
    path2 = path + "ccdt.out"
    with open(path2,'w') as f1:
        for loop2 in range(34):
            f1.write("%d  ccdt=%f\n"  % (loop2+1,ccdt[loop2]*132))

