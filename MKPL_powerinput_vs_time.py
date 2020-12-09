import numpy as np
from matplotlib import pyplot as plt

path = r"E:\Ansys\MKPL\simulated_MKPL_dissipation_SPS_fill_cycle_Carlo.txt"
data=np.loadtxt(path,delimiter=',')


scale = 2346/7304
t = data[:,0]
pwr = scale*data[:,1]
t_sim=t
pwr_sim=pwr
for i in range(1,10):
    t_add = t + 23*2*i
    cooldown=[23 * i * 2 - 0.1]
    t_sim=np.concatenate([t_sim,cooldown])
    t_sim=np.concatenate([t_sim,t_add])
    pwr_sim=np.concatenate([pwr_sim,[0]])
    pwr_sim=np.concatenate([pwr_sim,pwr])

    print('')


plt.figure()
plt.plot(t_sim,pwr_sim)

for i in range(0,len(pwr_sim)):
    print(str(t_sim[i]) + ', ' + str(pwr_sim[i]))
    

plt.show()
