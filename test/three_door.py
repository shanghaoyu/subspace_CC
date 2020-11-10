import random 
from random import choice
import math 
import numpy as np

door = [0, 1, 2]

win = 0
for loop in range(100000):
    subtract      = []
    car           = random.sample(door, 1)
    my_choice     = random.sample(door, 1)
    if car == my_choice:
        subtract = car
    else:
        subtract      = car + my_choice
    #print("car="+str(car)) 
    #print("my_choice="+str(my_choice)) 
    #print("subtract="+str(subtract)) 
    host_choice = np.delete(door,subtract) 
    host_choice = random.sample(host_choice.tolist(),1)
    #print("host_choice"+str(host_choice))  
    change_choice = np.delete(door,host_choice+my_choice).tolist() 
    #print("change_choice"+str(change_choice))  
    if change_choice == car:
        win = win + 1

print(win/100000)
