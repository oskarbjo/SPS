
import numpy as np
import matplotlib.pyplot as plt
import os



def importImpedanceData(filepath):

    #'MKI_PostLS1_impedance.txt'
    file=open(filepath,'r')
    #file=open('MKI_PostLS1_impedance.txt','r')
    lines=file.readlines()
    
    freq=[]
    R=[]
    for n in lines:
    #    R.append(n.split()[0])
        a=n.split()
        if a != []:
            freq.append(np.float(a[0])*1e6)
            R.append(np.float(a[1]))
            
#     R=R-np.min(R) #COMMENT THIS OUT UNLESS THERE IS A SMALL NEGATIVE REAL IMPEDANCE
#     R=np.add(R,1)   #COMMENT THIS OUT UNLESS IMPEDANCE SHOULD HAVE AN OFFSETS
    file.close()
    return [freq,R]

class SNPfile: #Can only be used for S2P..
    def __init__(self,filePath):
        self.S11lin = []
        self.S21lin = []
        self.S12lin = []
        self.S22lin = []
        self.freq = []
        self.S11db = []
        self.S21db = []
        self.S12db = []
        self.S22db = []
        self.S11angle = []
        self.S21angle = []
        self.S12angle = []
        self.S22angle = []
        self.S11real = []
        self.S21real = []
        self.S12real = []
        self.S22real = []
        self.S11imag = []
        self.S21imag = []
        self.S12imag = []
        self.S22imag = []
        path = filePath
        
        file=open(filePath,'r')
        lines=file.readlines()
        i=0
        while lines[i][0] == '!' :
            lines.pop(i)

        dataTypes=lines[0].split()
        lines.pop(0)
        format = dataTypes[3]
        
        self.extractSParamData(lines,format)

        
    def extractSParamData(self,lines,format):
        if format == 'dB':
            for i in range(0,len(lines)):
                data = self.separateParameters(lines[i])
                try:
                    self.freq = self.freq + [np.double(data[0])]
                    self.S11db = self.S11db + [np.double(data[1])]
                    self.S21db = self.S21db + [np.double(data[3])]
                    self.S12db = self.S12db + [np.double(data[5])]
                    self.S22db = self.S22db + [np.double(data[7])]
                    self.S11lin = self.S11lin + [self.dBtoLin(np.double(data[1]))]
                    self.S21lin = self.S21lin + [self.dBtoLin(np.double(data[3]))]
                    self.S12lin = self.S12lin + [self.dBtoLin(np.double(data[5]))]
                    self.S22lin = self.S22lin + [self.dBtoLin(np.double(data[7]))]
                    self.S11angle = self.S11angle + [np.double(data[2])]
                    self.S21angle = self.S21angle + [np.double(data[4])]
                    self.S12angle = self.S12angle + [np.double(data[6])]
                    self.S22angle = self.S22angle + [np.double(data[8])]
                except:
                    print('Removing SNP header')
        
        if format == 'RI':
            for i in range(0,len(lines)):
                data = self.separateParameters(lines[i])
                try:
                    self.freq = self.freq + [np.double(data[0])]
                    self.S11real = self.S11real + [np.double(data[1])]
                    self.S21real = self.S21real + [np.double(data[3])]
                    self.S12real = self.S12real + [np.double(data[5])]
                    self.S22real = self.S22real + [np.double(data[7])]
                    self.S11imag = self.S11imag + [np.double(data[2])]
                    self.S21imag = self.S21imag + [np.double(data[4])]
                    self.S12imag = self.S12imag + [np.double(data[6])]
                    self.S22imag = self.S22imag + [np.double(data[8])]
                    
                except:
                    print('Removing SNP header')
                    
            self.RItoDB()
        
        if format == 'MA':
            for i in range(0,len(lines)):
                data = self.separateParameters(lines[i])
                try:
                    self.freq = self.freq + [np.double(data[0])]
                    self.S11real = self.S11real + [np.double(data[1]) * (np.cos(np.deg2rad(np.double(data[2]))))]
                    self.S21real = self.S21real + [np.double(data[3]) * (np.cos(np.deg2rad(np.double(data[4]))))]
                    self.S12real = self.S12real + [np.double(data[5]) * (np.cos(np.deg2rad(np.double(data[6]))))]
                    self.S22real = self.S22real + [np.double(data[7]) * (np.cos(np.deg2rad(np.double(data[8]))))]
                    self.S11imag = self.S11imag + [np.double(data[1]) * (np.sin(np.deg2rad(np.double(data[2]))))]
                    self.S21imag = self.S21imag + [np.double(data[3]) * (np.sin(np.deg2rad(np.double(data[4]))))]
                    self.S12imag = self.S12imag + [np.double(data[5]) * (np.sin(np.deg2rad(np.double(data[6]))))]
                    self.S22imag = self.S22imag + [np.double(data[7]) * (np.sin(np.deg2rad(np.double(data[8]))))]
                    
                except:
                    print('Removing SNP header')
                    
            self.RItoDB()
                
                
    def RItoDB(self):
        self.S11db = 20*np.log10(np.sqrt(np.square(self.S11real)+np.square(self.S11imag)))
        self.S21db = 20*np.log10(np.sqrt(np.square(self.S21real)+np.square(self.S21imag)))
        self.S12db = 20*np.log10(np.sqrt(np.square(self.S12real)+np.square(self.S12imag)))
        self.S22db = 20*np.log10(np.sqrt(np.square(self.S22real)+np.square(self.S22imag)))
        
    def separateParameters(self,line):
        line=line.strip('\n')
        line=line.split('\t')
        return line
    
    def plotParam(self,Sparam):
        plt.figure()
        plt.plot(self.freq,Sparam)
        plt.show()
    
    def dBtoLin(self,dBdata):
        linData = np.power(10,dBdata/20)
        return linData
        
    def getImpulseResponse(self):
        self.s21_impulse = np.fft.ifft(self.S22lin)
    
    
class S4Pfile: #Works for S3P or higher order..
    def __init__(self,filePath):
        
        name,extension=os.path.splitext(filePath)
        self.N = np.int(extension[2]) #Order of network
        
        file=open(filePath,'r')
        lines=file.readlines()
        i=0
        while lines[i][0] == '!' :
            lines.pop(i)

        dataTypes=lines[0].split() #Find format that data was saved in 
        lines.pop(0) #Remove format line
        format = dataTypes[3]
        
        
        
        self.S_RI = np.ones([self.N,self.N,np.int(len(lines)/4)])*1j #Create empty matrix to save Sparameters in
        self.freq = []
        
        self.extractSParamData(lines,format)

        
    def extractSParamData(self,lines,format):
        freqLineLength = len(self.separateParameters(lines[0]))
        
        if format == 'dB':
            i=0 #Line index
            j=0 #Frequency index
            while i<len(lines):
                try:
                    for k in range(0,self.N):
                        data = self.separateParameters(lines[i])
                        if k == 0:
                            self.freq = self.freq + [np.double(data[0])]
         
                        for m in range(0,self.N):
                            self.S_RI[k,m,j] = self.dBtoLin(np.double(data[m*2+1])) * (np.cos(np.deg2rad(np.double(data[m*2+2])) + 1j*np.sin(np.deg2rad(np.double(data[m*2+2])))))
                        i=i+1
                    j=j+1
                except:
                    print('Removing SNP header')
        
        elif format == 'RI':
            i=0 #Line index
            j=0 #Frequency index
            while i<len(lines):
                try:
                    for k in range(0,self.N):
                        data = self.separateParameters(lines[i])
                        if k == 0:
                            self.freq = self.freq + [np.double(data[0])]
         
                        for m in range(0,self.N):
                            self.S_RI[k,m,j] = np.double(data[m*2+1]) + 1j * np.double(data[m*2+1]) 
                        i=i+1
                    j=j+1
                except:
                    print('Removing SNP header')
         
        elif format == 'MA':
            i=0 #Line index
            j=0 #Frequency index
            while i<len(lines):
                try:
                    for k in range(0,self.N):
                        data = self.separateParameters(lines[i])
                        if k == 0:
                            self.freq = self.freq + [np.double(data[0])]
         
                        for m in range(0,self.N):
                            self.S_RI[k,m,j] = np.double(data[m*2+1]) * (np.cos(np.deg2rad(np.double(data[m*2+2])) + 1j*np.sin(np.deg2rad(np.double(data[m*2+2])))))
                        i=i+1
                    j=j+1
                except:
                    print('Removing SNP header')
            
            
    def RItoDB(self):
        self.S11db = 20*np.log10(np.sqrt(np.square(self.S11real)+np.square(self.S11imag)))
        self.S21db = 20*np.log10(np.sqrt(np.square(self.S21real)+np.square(self.S21imag)))
        self.S12db = 20*np.log10(np.sqrt(np.square(self.S12real)+np.square(self.S12imag)))
        self.S22db = 20*np.log10(np.sqrt(np.square(self.S22real)+np.square(self.S22imag)))
        
    def separateParameters(self,line):
        line=line.strip('\n')
        line=line.split('\t')
        return line
    
    def plotParam(self,Sparam):
        plt.figure()
        plt.plot(self.freq,Sparam)
        plt.show()
    
    def dBtoLin(self,dBdata):
        linData = np.power(10,dBdata/20)
        return linData
        
    def getImpulseResponse(self):
        self.s21_impulse = np.fft.ifft(self.S22lin)
    
        
def main():
    
    path00 = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\4port measurement\MKPL_4PORT.S4P"
    path01 = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\4port measurement\MKPL_4PORT_SERIGRAPHY_GROUNDED.S4P"
    
    S = S4Pfile(path00)
    S0 = S4Pfile(path01)
    mostContrFreq=np.array([1.9947895e+08, 2.3927915e+08, 2.7955730e+08, 2.3923570e+08,
       2.7951385e+08, 1.9943550e+08, 3.1931405e+08, 3.1935750e+08,
       3.5911425e+08, 1.5967875e+08])/1e6
    plt.figure()
#     plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[2,0,:]))-20*np.log10(np.abs(S.S_RI[3,0,:])))
#     plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[2,0,:])))
#     plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[3,0,:])))
#     plt.plot(np.divide(S0.freq,1e6),20*np.log10(np.abs(S0.S_RI[2,0,:]))-20*np.log10(np.abs(S0.S_RI[3,0,:])))
    plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[2,0,:])),color='blue')
    plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[3,0,:])),color='black')
#     plt.plot(np.divide(S.freq,1e6),20*np.log10(np.abs(S.S_RI[2,0,:]))-20*np.log10(np.abs(S.S_RI[3,0,:])))
    j=0
#     for i in mostContrFreq:    
#         plt.plot([i,i],[-100,100],linestyle='--',color='red')
#         j=j+1
    plt.xlim([0,500])
    plt.ylim([-90, -10])
    plt.ylabel('Transmission [dB]')
    plt.xlabel('Frequency [MHz]')
    plt.grid()
    plt.legend(['S31/Upstream busbar','S41/Downstream busbar'])


    path5=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Wire measurement\MKPL_WIRE_WITH_SERIGRAPHY_ALUMINIUMGROUNDING.S2P"
    S6 = SNPfile(path5)
    path6 = r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Wire measurement\MKPL_WIRE_WITHOUT_SERIGRAPHY.S2P"
    S7 = SNPfile(path6)
    plt.figure()
#     plt.plot(np.divide(S6.freq,1e6),20*np.log10(np.abs(S6.S_RI[1,0,:])))
    plt.plot(np.divide(S6.freq,1e6),S6.S21db)
    plt.plot(np.divide(S7.freq,1e6),S7.S21db)
    plt.ylim([-60, -0])
    plt.ylabel('Transmission [dB]')
    plt.xlabel('Frequency [MHz]')
#     plt.legend(['Upstream busbar','Downstream busbar'])
#     plt.legend(['Busbar upstream to downstream'])
    plt.grid()

    path0=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Wire measurement\data_measurements_and_model_forOskar.txt"
    data=np.loadtxt(path0)
    from scipy.io import loadmat
    d = loadmat(r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Wire measurement\serigraphy\MKPL_Longitudinal_measurements_cmp_all.fig", squeeze_me=True, struct_as_record=False)
    x=d['hgS_070000'].children[0].children[0].properties.XData
    y=d['hgS_070000'].children[0].children[0].properties.YData     
    plt.figure()
    plt.plot(data[:,0]/1e6,data[:,1],color='blue')
    plt.plot(data[:,0]/1e6,data[:,2],color='black')
    plt.grid()
    plt.xlim([0,500])
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Real($Z_{||}$) [$\Omega$]')
    plt.legend(['Wire measurement','CST model'])
    
    
    [f,R] = importImpedanceData("E:\CST\MKPL\MKPL serigraphy Mario\impedance simu\MKPL_serigraphy_ZR.txt")
    plt.figure()
    plt.plot(np.divide(f,1e9),R,color='blue')
    plt.plot(np.divide(x,1e9),y,color='black')
    plt.grid()
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Real($Z_{||}$) [$\Omega$]')
    plt.legend(['CST simulation with serigraphy','Wire measurement with serigraphy'])
    
    
    [f,R] = importImpedanceData("E:\CST\MKPL\MKPL serigraphy Mario\impedance simu\MKPL_mario_no_serigraphy_ZR.txt")
    plt.figure()
    plt.plot(np.divide(f,1e9),R)
    plt.plot(data[:,0]/1e9,data[:,1],color='blue')
    plt.plot(data[:,0]/1e9,data[:,2],color='black')
    
    
    S_highres = S4Pfile(r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\4port measurement\MKPL_4PORT_HIGHRES.S4P")
    plt.figure()
    plt.plot(np.divide(S_highres.freq,1e6),20*np.log10(np.abs(S_highres.S_RI[2,0,:])))
    plt.xlim([0,30])
    plt.ylim([-90, -10])
    plt.ylabel('Transmission [dB]')
    plt.xlabel('Frequency [MHz]')
    plt.grid()
#     path1=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Probe measurements\MKPL_PROBEMEASUREMENT_SIDE1_WIRE_AND_PROBE_SAMESIDE.S2P"
#     path2=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Probe measurements\MKPL_PROBEMEASUREMENT_SIDE1_WIRE_AND_PROBE_OPPOSITESIDE.S2P"
#     path3=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Probe measurements\MKPL_PROBEMEASUREMENT_SIDE2_WIRE_AND_PROBE_OPPOSITESIDE.S2P"
#     path4=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Probe measurements\MKPL_PROBEMEASUREMENT_SIDE2_WIRE_AND_PROBE_SAMESIDE.S2P"
#     path5=r"C:\Users\objorkqv\cernbox\Documents\Measurements\MKPL\Wire measurement\MKPL_WIRE_WITHOUT_SERIGRAPHY.S2P"
#     S1 = SNPfile(path1)
#     S2 = SNPfile(path2)
#     S3 = SNPfile(path3)
#     S4 = SNPfile(path4)
#     S5 = SNPfile(path5)
#     
#     plt.figure()
#     plt.plot(S1.freq,S1.S21db)
#     plt.plot(S3.freq,S3.S21db)
#     plt.ylabel('Transmission [dB]')
#     plt.xlabel('Frequency [Hz]')
#     plt.legend(['Probe at upstream end','Probe at downstream end'])
#     plt.grid()
# 
#     plt.figure()
#     plt.plot(S4.freq,S4.S21db)
#     plt.plot(S2.freq,S2.S21db)
#     plt.ylabel('Transmission [dB]')
#     plt.xlabel('Frequency [Hz]')
#     plt.legend(['Probe at upstream end','Probe at downstream end'])
#     plt.grid()
# 
#     plt.figure()
#     plt.plot(S5.freq,S5.S21db)
#     
#     CST_mario_impFile = r"E:\CST\MKPL\MKPL serigraphy Mario\impedance simu\MKPL_mario_no_serigraphy_ZR.txt"
#     CST_my_impFile = r"E:\CST\MKPL\Lorena_Import\Impedance simu\MKPL_real_Z_normal_beam_direction.txt"
#     
#     [f1,R1]=importImpedanceData(CST_mario_impFile)
#     [f2,R2]=importImpedanceData(CST_my_impFile)
#     
    plt.show()
    
    
    

if __name__ == "__main__":
    main()



