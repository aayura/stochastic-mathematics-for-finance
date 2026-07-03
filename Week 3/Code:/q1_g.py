import numpy as np

M = 10000
t = 0.5

samples = np.sqrt(t) * np.random.randn(M)

mean_est = np.mean(samples)
var_est = np.var(samples)

print("Estimated Mean =", mean_est)
print("Estimated Variance =", var_est)