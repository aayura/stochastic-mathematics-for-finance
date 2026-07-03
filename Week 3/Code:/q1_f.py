import numpy as np
import matplotlib.pyplot as plt

Ns = [10,50,100,500,1000,5000,10000]

QV = []

for N in Ns:
    dt = 1/N
    increments = np.sqrt(dt) * np.random.randn(N)
    QV.append(np.sum(increments**2))

plt.figure(figsize=(8,5))
plt.plot(Ns,QV,'o-')
plt.axhline(y=1,color='r',linestyle='--',label='Theoretical QV = 1')
plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Quadratic Variation')
plt.legend()
plt.grid(True)
plt.show()