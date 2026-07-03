import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

S0 = 100.0
mu = 0.07
sigma = 0.2

T = 1.0
N = 500
n_paths = 50000

dt = T / N

S = np.full(n_paths, S0, dtype=float)

for _ in range(N):
    dW = np.random.normal(0, np.sqrt(dt), size=n_paths)
    S = S + mu * S * dt + sigma * S * dW

ST = S

plt.figure(figsize=(10, 6))

plt.hist(
    ST,
    bins=100,
    density=True,
    alpha=0.6,
    label="Euler-Maruyama Simulation"
)

shape = sigma * np.sqrt(T)
scale = S0 * np.exp((mu - 0.5 * sigma**2) * T)

x = np.linspace(ST.min(), ST.max(), 1000)

pdf = lognorm.pdf(
    x,
    s=shape,
    scale=scale
)

plt.plot(
    x,
    pdf,
    linewidth=2,
    label="Theoretical Lognormal Density"
)

plt.title("GBM Terminal Price Distribution")
plt.xlabel(r"$S_T$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.show()

emp_mean = np.mean(ST)
emp_std = np.std(ST)

print(f"Empirical Mean = {emp_mean:.4f}")
print(f"Empirical Std  = {emp_std:.4f}")

theory_mean = S0 * np.exp(mu * T)

theory_std = (
    S0
    * np.exp(mu * T)
    * np.sqrt(np.exp(sigma**2 * T) - 1)
)

print(f"Theoretical Mean = {theory_mean:.4f}")
print(f"Theoretical Std  = {theory_std:.4f}")