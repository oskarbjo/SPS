import pytimber as pt
import matplotlib.pyplot as plt
import numpy as np


def savedata():
    np.save('E:\Ansys\MKPL/SPS_timedata.npy',timestamps2)
    np.save('E:\Ansys\MKPL/SPS_tempdata.npy',temperatures)
    np.save('E:\Ansys\MKPL/SPS_tempdata2.npy',temperatures2)
    np.save('E:\Ansys\MKPL/SPS_timedata2.npy',timestamps3)    

db=pt.LoggingDB()

START_TIMESTAMP = '2015-06-17 02:00:00'
STOP_TIMESTAMP = '2015-06-17 02:03:00'

#FIRST SEGMENT
# START_TIMESTAMP = '2015-06-17 00:00:00'
# STOP_TIMESTAMP = '2015-06-17 04:00:00'

#SECOND SEGMENT
# START_TIMESTAMP = '2015-06-17 04:00:00'
# STOP_TIMESTAMP = '2015-06-17 06:03:00'

#THIRD SEGMENT
# START_TIMESTAMP = '2015-06-17 06:03:00'
# STOP_TIMESTAMP = '2015-06-17 08:00:00'

#FOURTH SEGMENT
# START_TIMESTAMP = '2015-06-17 08:00:00'
# STOP_TIMESTAMP = '2015-06-17 18:00:00'


# data=db.get('SPS.BCTDC.51895:TOTAL_INTENSITY','2018-06-16 18:02:00','2018-06-16 19:04:00')
data=db.get('SPS.BCTDC.31832:TOTAL_INT',START_TIMESTAMP,STOP_TIMESTAMP)
data_temp=db.get('MKP.11955:TEMPERATURE.1',START_TIMESTAMP,STOP_TIMESTAMP)
data_temp2=db.get('MKP.11955:TEMPERATURE.2',START_TIMESTAMP,STOP_TIMESTAMP)

timeScale = 40/7108
# db.searchFundamental('SPS:MD_SCRUB_%', '2018-06-03 21:02:00', '2018-06-03 23:25:00')

timestamps,intensities=data['SPS.BCTDC.31832:TOTAL_INT']
timestamps2,temperatures = data_temp['MKP.11955:TEMPERATURE.1']
timestamps3,temperatures2 = data_temp2['MKP.11955:TEMPERATURE.2']
ts1=0
intensityArray = np.array([])
dates0=[]
plt.figure()
j=0
fig, axs = plt.subplots(2)
for ts,d in zip(timestamps,intensities):
    try:
        axs[0].plot(np.linspace(timestamps[j],timestamps[j+1],len(d))-timestamps[0],d,label=pt.dumpdate(ts,fmt='%H:%M:%S'))
        plt.ylabel('Intensity [x 1e10 protons]')
        plt.xlim([np.min(timestamps),np.max(timestamps)])

    except:
        print('out of index')     
    ts1 = len(d)+ts1
    intensityArray=np.append(intensityArray,d)
    dates0.append(pt.dumpdate(ts,fmt='%H:%M:%S'))
    j=j+1

plt.legend




dates=[]
for ts,d in zip(timestamps2,temperatures):
    dates.append(pt.dumpdate(ts,fmt='%H:%M:%S'))


    
# axs[0].plot(intensityArray)
axs[1].plot(timestamps2,temperatures)
plt.xticks(timestamps2, dates)
plt.title(pt.dumpdate(ts,fmt='%Y-%m-%d'))
plt.ylabel('Temperature [C]')
plt.xlim([np.min(timestamps),np.max(timestamps)])
plt.tight_layout()
plt.legend()


print('Fill factor: ' + str(np.sum(intensityArray)/(len(intensityArray)*np.max(intensityArray))))
print('Bunch intensity: ' + str(np.format_float_scientific(1e10*np.max(intensityArray)/(72*4))))

# savedata()


plt.show()