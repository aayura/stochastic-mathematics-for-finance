import numpy as np
import matplotlib.pyplot as plt

N_values = [10**2, 10**3, 10**4, 10**5]

np.random.seed(42)

pi_estimates = []

for N in N_values:

    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)

    inside = (x**2 + y**2 <= 1)

    pi_hat = 4 * np.mean(inside)

    pi_estimates.append(pi_hat)

    print(f"N={N:6d}   π estimate = {pi_hat:.6f}")
