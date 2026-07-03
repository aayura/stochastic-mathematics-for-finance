import numpy as np
import matplotlib.pyplot as plt

N_values = [10**2, 10**3, 10**4, 10**5]
pi_estimates = []

for N in N_values:
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)

    inside = (x**2 + y**2 <= 1)

    pi_hat = 4 * np.sum(inside) / N
    pi_estimates.append(pi_hat)

# Plot π estimate against N
plt.figure(figsize=(8, 5))
plt.plot(N_values, pi_estimates, marker='o')
plt.axhline(np.pi, linestyle='--', label=f'True π = {np.pi:.5f}')

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Estimated π')
plt.title('Monte Carlo Estimation of π')
plt.legend()
plt.grid(True)

plt.show()