
import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'cambria',
        'weight' :  'ultralight',
        'size'   : 12}
plt.rc('font', **font)


t_meas=np.load('./SPS_timedata.npy')
T_meas=np.load('./SPS_tempdata.npy')
t_meas2=np.load('./SPS_timedata2.npy')
T_meas2=np.load('./SPS_tempdata2.npy')

t_meas = t_meas - t_meas[0]
t_meas2 = t_meas2 - t_meas2[0]

path=r"./temp_data.csv"
path2=r"./MKPL_transients_TCC_sweep.csv"
path3=r"./temp_data_TANK_full_model.csv"
path4 = r"./simulation_good_agreement.csv"
data=np.loadtxt(path,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data=np.double(data)
data_TCCsweep = np.loadtxt(path2,skiprows=2,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data_TCCsweep = np.double(data_TCCsweep)
data_rad_only=np.loadtxt(path3,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
data_rad_only=np.double(data_rad_only)

data_final_simu = np.loadtxt(path4,delimiter=',')



def plotTCCsweep():
    plt.figure(figsize=(7,4))
#     plt.plot(t_meas/3600,T_meas,'k',label='Measured (SPS MKP-L)')
    plt.plot(data_TCCsweep[0,:]/3600,data_TCCsweep[1,:],'b', label='TCC=30')
    plt.plot(data_TCCsweep[0,:]/3600,data_TCCsweep[2,:],'g', label='TCC=300')
    plt.plot(data_TCCsweep[0,:]/3600,data_TCCsweep[3,:],'r', label='TCC=600')
    plt.plot(data_TCCsweep[0,:]/3600,data_TCCsweep[4,:],'y', label='TCC=3000')
    plt.xlabel('Time [hours]')
    plt.ylabel('Temperature [$^\circ$C]')
    plt.grid(color='k', linestyle=':')
    plt.legend(loc='upper left', shadow=True, fontsize=12)
    plt.tight_layout()
    plt.tick_params(direction='in')
    plt.xlim([0,26400/3600])
    plt.ylim(34,42)

def tempTransients():
    plt.figure(figsize=(7,4))
    plt.plot(t_meas/3600,T_meas,'k',label='Measured (SPS MKP-L upstream)')
    plt.plot(t_meas2/3600,T_meas2,'g',label='Measured (SPS MKP-L downstream)')
    plt.plot(data_final_simu[:,0],data_final_simu[:,1],'b', label='Simulated (ANSYS)')
    plt.xlabel('Time [hours]')
    plt.ylabel('Temperature [$^\circ$C]')
    plt.grid(color='k', linestyle=':')
    plt.legend(loc='lower right', shadow=True, fontsize=12)
    plt.tight_layout()
    
    plt.tick_params(direction='in')
    
    plt.xlim([0,68000/3600])
    plt.ylim(34,50)

def ferriteTemp():
    path = r"E:\Ansys\MKPL\simulation_results_data\full model\temp_data_TANK_FERRITE_full_model.csv"
    path2 = r"E:\Ansys\MKPL\simulation_results_data\full model\temp_data_TANK_FERRITE_YOKE2_full_model.csv"
    data=np.loadtxt(path,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
    data=np.double(data)
    data2=np.loadtxt(path2,skiprows=1,dtype='str', delimiter=',', unpack=True, encoding="utf8")
    data2=np.double(data2)
    
    plt.figure(figsize=(7,4))
    plt.plot(data[0,:]/3600,data[1,:],'b', label='Simulated yoke 1 peak temp')
    plt.plot(data2[0,:]/3600,data2[1,:],'r', label='Simulated yoke 2 peak temp')
    plt.xlabel('Time [hours]')
    plt.ylabel('Temperature [$^\circ$C]')
    plt.grid(color='k', linestyle=':')
    plt.legend(loc='lower left', shadow=True, fontsize=12)
    plt.tight_layout()
    plt.ylim([40,np.max(data[1,:]*1.1)])
    plt.tick_params(direction='in')
    plt.xlim([0,68000/3600])


plotTCCsweep()
tempTransients()
ferriteTemp()

plt.show()