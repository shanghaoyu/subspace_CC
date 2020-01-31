import numpy as np
import math
import re
from scipy import interpolate
from scipy import linalg



def input_file(file_path,matrix):
    with open(file_path, 'r') as f_1:
        data = f_1.readlines()
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[0])
        subspace_dimension = int(temp_1[0])
        for loop1 in range (0, subspace_dimension):
            temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[1+loop1])
            matrix[loop1,:] = temp_2[:]

#LECs = [200,-91.85]
magic_no = 14

subspace_dimension = 5
N_matrix = np.zeros((subspace_dimension,subspace_dimension))
H_matrix = np.zeros((subspace_dimension,subspace_dimension))
K_matrix = np.zeros((subspace_dimension,subspace_dimension))
input_file("N_matrix.txt",N_matrix)
input_file("H_matrix_1.txt",H_matrix)
input_file("K_matrix.txt",K_matrix)

#H = LECs[0]*H_matrix + K_matrix
H = H_matrix + K_matrix
print("H="+str(H))

N = np.matrix(N_matrix)
Ni = N.I
print (Ni)
#N_I = np.linalg.inv(N_matrix)
#
#print (N_matrix*N_I)

Ni_dot_H = np.dot(Ni,H)
D,V = np.linalg.eig(Ni_dot_H)
print (Ni_dot_H)
print ("D="+str(D))
#print ("V="+str(V))

loop2 = 0
for loop1 in range(subspace_dimension):
    ev = D[loop1] 
    if ev.imag != 0:
        continue
#    if ev.real < 0:
#        continue
    loop2 = loop2+1
ev_all = np.zeros(loop2)

loop2 = 0
for loop1 in range(subspace_dimension):
    ev = D[loop1] 
    if ev.imag != 0:
        continue
#    if ev.real < 0:
#        continue
    ev_all[loop2] = ev.real
    loop2 = loop2+1


ev_sorted = sorted(ev_all)
print(ev_sorted[0])





#D,V = np.linalg.eig(H_matrix)
#print ("D="+str(D))
#print(np.linalg.matrix_rank(N_matrix))
#print(np.linalg.matrix_rank(H_matrix))


#print(N_matrix)
#print(H_matrix)
