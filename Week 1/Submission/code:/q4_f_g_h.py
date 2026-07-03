import numpy as np

P = np.array([
    [0.60, 0.30, 0.10, 0.00],
    [0.20, 0.50, 0.20, 0.10],
    [0.10, 0.30, 0.40, 0.20],
    [0.05, 0.15, 0.30, 0.50]
])

returns = np.array([0.012, 0.004, -0.006, -0.025])

N = 10000

states = np.zeros(N, dtype=int)

states[0] = 0

for t in range(1, N):
    current = states[t-1]
    states[t] = np.random.choice([0,1,2,3], p=P[current])

print(states[:20])

freq = np.bincount(states, minlength=4) / N

print("Empirical frequencies:")
print(freq)


pi = np.array([0.2455, 0.3455, 0.2727, 0.1364])

print("Stationary distribution:")
print(pi)

# simulated should be close to stationary distribution calculated in b.

sim_returns = returns[states]

empirical_avg = np.mean(sim_returns)

print("Empirical average return:")
print(empirical_avg)

import matplotlib.pyplot as plt

cumulative = np.cumsum(sim_returns)

plt.figure(figsize=(10,5))
plt.plot(cumulative)
plt.xlabel("Weeks")
plt.ylabel("Cumulative Return")
plt.title("Cumulative Portfolio Return")
plt.grid(True)
plt.show()