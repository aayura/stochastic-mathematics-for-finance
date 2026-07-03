import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

X0 = 3.0
theta = 2.0
sigma = 0.5

T = 10
N = 10000
n_paths = 20000

dt = T / N

X = np.full(n_paths, X0)

for _ in range(N):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)

    X = (
        X
        - theta * X * dt
        + sigma * dW
    )

XT = X

# Empirical statistics
emp_mean = np.mean(XT)
emp_var = np.var(XT)

print("Empirical Mean =", emp_mean)
print("Empirical Variance =", emp_var)

# Theory
theory_mean = 0
theory_var = sigma**2 / (2 * theta)
theory_std = np.sqrt(theory_var)

print("Theoretical Mean =", theory_mean)
print("Theoretical Variance =", theory_var)

# Histogram
plt.figure(figsize=(10,6))

plt.hist(
    XT,
    bins=80,
    density=True,
    alpha=0.6,
    label="Simulation"
)

x = np.linspace(
    XT.min(),
    XT.max(),
    1000
)

plt.plot(
    x,
    norm.pdf(x, theory_mean, theory_std),
    linewidth=2,
    label="Theoretical N(0, σ²/2θ)"
)

plt.legend()
plt.grid(True)
plt.show()