
from matplotlib import pyplot as plt
import numpy as np

P = np.array([100,200,400,600,800,1000])
T = [61,96,157,212,263,306]

plt.figure()
plt.plot(P,T)
plt.xlabel('Total MKP-L power dissipation [W]')
plt.ylabel('Ferrite peak temperature [C]')
plt.title('MKP-L peak temperature based on coupled CST and ANSYS simulations')

plt.show()