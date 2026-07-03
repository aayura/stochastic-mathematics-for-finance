import numpy as np

S0 = 100
K = 110
N = 100000

sigmas = [0.1, 0.2, 0.3, 0.5]

for sigma in sigmas:

    Z = np.random.normal(0, 1, N)

    ST = S0 * np.exp(sigma * Z)

    payoff = np.maximum(ST - K, 0)

    estimate = np.mean(payoff)

    print(f"sigma = {sigma:.1f}   Expected payoff = {estimate:.4f}")