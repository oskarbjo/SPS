
import matplotlib.pyplot as plt
import numpy as np



def main():    

    [f0,R0] = importImpedanceData(r"E:\CST\MKPL\Lorena_Import\Impedance simu\real_impedance.txt")
    [f1,R1] = importImpedanceData(r"E:\CST\MKPL\Lorena_Import\Impedance simu\real_impedance_fixed_transition.txt")
 
    plt.figure(num=None, figsize=(10, 6), dpi=140, facecolor='w', edgecolor='k')
    plt.plot(np.asarray(f0)/1e6,R0)
    plt.plot(np.asarray(f1)/1e6,R1)
#     plt.plot(np.asarray(f3)/1e6,R3)
    plt.legend(['Original','Fixed transition'])
    plt.grid()
    plt.ylabel('Impedance [Ohms]')
    plt.xlabel('Frequency [MHz]')

    plt.show()
    
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
    file.close()
    
    return [freq,R]

    
if __name__ == "__main__":
    main()
    
