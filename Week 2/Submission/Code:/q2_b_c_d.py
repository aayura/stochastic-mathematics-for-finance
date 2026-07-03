import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# True value
f = lambda x: np.exp(-x**2)
I_true, _ = quad(f, 0, 1)

print("True integral =", I_true)

N_values = [10**2, 10**3, 10**4, 10**5]

estimates = []
errors = []

np.random.seed(42)

for N in N_values:
    U = np.random.uniform(0, 1, N)

    I_hat = np.mean(np.exp(-U**2))

    error = abs(I_hat - I_true)

    estimates.append(I_hat)
    errors.append(error)

    print(f"N={N:6d} | Estimate={I_hat:.8f} | Error={error:.8e}")

plt.figure(figsize=(8,5))

plt.plot(N_values, estimates, marker='o')
plt.axhline(I_true, linestyle='--', label='True value')

plt.xscale('log')
plt.xlabel('N')
plt.ylabel('Monte Carlo Estimate')
plt.title(r'Estimation of $\int_0^1 e^{-x^2}dx$')
plt.legend()
plt.grid(True)

plt.show()