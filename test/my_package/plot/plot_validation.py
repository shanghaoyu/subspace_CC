import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 
import re
import seaborn as sns



from scipy import interpolate


def input_file_1(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])
                raw_data[loop2][1] = float(temp_1[1])
                #raw_data[loop2][2] = float(temp_1[4])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 

def input_file_count(file_path):
    count = len(open(file_path,'rU').readlines())
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


particle_num = 14
file_path = "validation.txt"
data_num = input_file_count(file_path)

raw_data = np.zeros((data_num, 2),dtype = np.float)
input_file_1(file_path,raw_data)

validation_data =  raw_data[np.where((raw_data[:,0]>-140)&(raw_data[:,0]<540))]

error = validation_data[:,0]-validation_data[:,1]

relative_error=2*error/(validation_data[:,0]+validation_data[:,1])
mean_relative_error = np.mean(abs(relative_error))
re_1 = relative_error[np.where((relative_error[:]<0.01))]
print(relative_error)
print(len(re_1))
print(len(validation_data))
print("<0.01 point : " +str(len(re_1)*1.0/len(validation_data)))


print(mean_relative_error)
####plot###
fig1 = plt.figure('fig1',figsize = (7,7))
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax1 = plt.subplot(111)
plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['top'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
ax1.spines['right'].set_linewidth(2)



# DNNLO450
x_list_0 = [14.6579580643]
y_list_0 = [14.6579580643]

# ccd calculation
x_list_1 = validation_data[:,0]/particle_num
# emulator 
y_list_1 = validation_data[:,1]/particle_num




l0 = plt.scatter (x_list_0,y_list_0,color = 'k', marker = 's',s = 200 ,zorder = 4, label=r'$\Delta$NNLO$_{\rm{go}}$(450)')
l1 = plt.scatter (x_list_1, y_list_1,color = 'cornflowerblue', edgecolor = 'k', marker = 'o',s = 120 ,zorder=2,label = 'SP-CC(64)')
l2 = plt.plot([-10, 40], [-10, 40], ls="-",color = 'k', lw = 3, zorder = 3)

plt.xlim((-10,40))
plt.ylim((-10,40))
plt.xticks(np.arange(-10,41,10),fontsize = 15)
plt.yticks(np.arange(-10,41,10),fontsize = 15)


plt.legend(loc='upper left',fontsize = 15)
plt.xlabel(r"$\rm{CCSD} \ [\rm{MeV}]$",fontsize=20)
plt.ylabel(r"$\rm{SP-CC} \ [\rm{MeV}]$",fontsize=20)

plot_path = 'emulator_validation.pdf'
plt.savefig(plot_path,bbox_inches='tight')
plt.close('all')


### plot 2 ###
x_tick_min = -0.05
x_tick_max =  0.05
x_tick_gap =  0.01
x_lim_min  = -0.05
x_lim_max  =  0.05
#y_lim_min =
#y_lim_max = 

fig2 = plt.figure('fig2',figsize = (7,7))
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
ax2 = plt.subplot(111)
plt.tick_params(top=True,bottom=True,left=True,right=True,width=2)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['top'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
ax2.spines['right'].set_linewidth(2)

sns.set_palette("hls")
matplotlib.rc("figure")
l2 = sns.distplot(relative_error,bins=300,kde_kws={"color":"seagreen", "lw":0 },hist_kws={ "color": "seagreen"},label='')

plt.legend(loc='upper left',fontsize = 15)
plt.xlabel(r"ralative error",fontsize=20)
plt.ylabel(r"count",fontsize=20)
plt.xticks(np.arange(x_tick_min,x_tick_max+0.00001,x_tick_gap),fontsize = 10)
#plt.yticks(np.arange(y_tick_min,y_tick_max+0.00001,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
#plt.ylim((y_lim_min,y_lim_max))





plot_path = 'ralative_error.pdf'
plt.savefig(plot_path,bbox_inches='tight')
plt.close('all')



