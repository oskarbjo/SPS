
import numpy as np
import matplotlib.pyplot as plt



path=r"E:\Ansys\MKPL\MKPL_full_fill_cycle_heating.csv"
data=np.loadtxt(path,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data=np.double(data)

plt.figure()
plt.plot(data[0,:],data[1,:])
plt.title('Peak temp in first MKPL yoke during 2015 June MD fill cycle. 50 TCC to HV plates')
plt.grid()
plt.ylabel('Temperature [C]')
plt.xlabel('Time [s]')


plt.figure()
plt.plot(data[0,:],data[3,:]-data[2,:])
plt.title('Temperature difference, inside and outside of ferrite leg')
plt.grid()
plt.ylabel('Temperature difference[C]')
plt.xlabel('Time [s]')


plt.figure()
plt.plot(data[0,:],data[4,:]-data[5,:])
plt.title('Temperature difference, top and foot of leg')
plt.grid()
plt.ylabel('Temperature difference [C]')
plt.xlabel('Time [s]')


plt.show()
