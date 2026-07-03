import numpy as np

T = 1
N = 10000
dt = T/N

increments = np.sqrt(dt) * np.random.randn(N)

W = np.concatenate(([0], np.cumsum(increments)))

QV = np.sum(increments**2)
TV = np.sum(np.abs(increments))

print("Quadratic Variation =", QV)
print("Theoretical Value   =", T)
print("Total Variation     =", TV)