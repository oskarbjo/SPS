
import numpy as np
import matplotlib.pyplot as plt



path=r"E:\Ansys\MKPL\MKPL_4batch_vs_3batch_heating.csv"
path2=r"E:\Ansys\MKPL\MKPL_3batch_to_4batch.csv"
data=np.loadtxt(path,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data=np.double(data)
data2=np.loadtxt(path2,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data2=np.double(data2)


a=np.double([0])
b=np.double([25])

t = np.concatenate((a,data[0,:]))
T_4batch = np.concatenate((b,data[1,:]))
T_3batch = np.concatenate((b,data[2,:]))

t2 = np.concatenate((a,data2[0,:]))
T_3_to_4batch = np.concatenate((b,data2[1,:]))
T_3_to_4batch_outer_ferrite = np.concatenate((b,data2[2,:]))
T_3_to_4batch_inner_ferrite = np.concatenate((b,data2[3,:]))

T_3_to_4batch_RHS_ferrite = np.concatenate((b,data2[4,:]))
T_3_to_4batch_LHS_ferrite = np.concatenate((b,data2[5,:]))

T_4batch_outer_ferrite = np.concatenate((b,data2[6,:]))
T_4batch_inner_ferrite = np.concatenate((b,data2[7,:]))

T_4batch_RHS_ferrite = np.concatenate((b,data2[8,:]))
T_4batch_LHS_ferrite = np.concatenate((b,data2[9,:]))


plt.figure()
plt.plot(t,T_4batch)
plt.plot(t,T_3batch)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['4 batch fill','3 batch fill'])
plt.title('Ferrite peak temperature')

plt.figure()
plt.plot(t2,T_3_to_4batch)
plt.plot(t,T_4batch)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['3 to 4 batch fill','4 batch fill'])
plt.title('Ferrite peak temperature')

plt.figure()
plt.plot(t2,T_3_to_4batch_inner_ferrite-T_3_to_4batch_outer_ferrite)
# plt.plot(t2,T_3_to_4batch_inner_ferrite)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['Temperature difference, inner to outer ferrite'])
plt.title('3 to 4 batch fill')

plt.figure()
plt.plot(t2,T_3_to_4batch_RHS_ferrite-T_3_to_4batch_LHS_ferrite)
# plt.plot(t2,T_3_to_4batch_inner_ferrite)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['Temperature difference, bottom to top of ferrite leg'])
plt.title('3 to 4 batch fill')

plt.figure()
plt.plot(t2,T_4batch_inner_ferrite-T_4batch_outer_ferrite)
# plt.plot(t2,T_3_to_4batch_inner_ferrite)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['Temperature difference, inner to outer ferrite'])
plt.title('4 batch fill')

plt.figure()
plt.plot(t2,T_4batch_RHS_ferrite-T_4batch_LHS_ferrite)
# plt.plot(t2,T_3_to_4batch_inner_ferrite)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.legend(['Temperature difference, bottom to top of ferrite leg'])
plt.title('4 batch fill')
plt.show()