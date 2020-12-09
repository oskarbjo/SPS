import numpy as np
import matplotlib.pyplot as plt


P = [50,
100,
150,
200,
300,
400,
800,
1000,
1300]

T_3D = [59.6,
79.3,
89.7,
93.3,
121.3,
189.6,
282.1,
321.9,
372.9]

T_HVplate = [69.2,
104.2,
133.1,
158,
200.4,
236.3,
347.3,
392.4,
452.1]

T_GNDplate = [90.3,
136.7,
173,
203.6,
254.5,
296.4,
422.5,
472.1,
536.4]

order=2

z1 = np.polyfit(P, T_3D, order)
p1 = np.poly1d(z1)

z2 = np.polyfit(P, T_HVplate, order)
p2 = np.poly1d(z2)

z3 = np.polyfit(P, T_GNDplate, order)
p3 = np.poly1d(z3)


plt.figure()
plt.plot(P,T_3D)
plt.plot(P,p1(P))

plt.figure()
plt.plot(P,T_HVplate)
plt.plot(P,p2(P))

plt.figure()
plt.plot(P,T_GNDplate)
plt.plot(P,p3(P))


plt.show()