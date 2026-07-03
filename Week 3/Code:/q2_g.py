import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

S0 = 100.0
mu = 0.07
sigma = 0.2
T = 1.0

n_paths = 50000

Z = np.random.randn(n_paths)

ST = S0 * np.exp(
    (mu - 0.5 * sigma**2) * T
    + sigma * np.sqrt(T) * Z
)

log_returns = np.log(ST / S0)

emp_mean = np.mean(log_returns)
emp_var = np.var(log_returns)
emp_std = np.std(log_returns)

theory_mean = (mu - 0.5 * sigma**2) * T
theory_var = sigma**2 * T
theory_std = sigma * np.sqrt(T)

print("===== Log Return Statistics =====")
print(f"Empirical Mean      : {emp_mean:.6f}")
print(f"Theoretical Mean    : {theory_mean:.6f}")
print()

print(f"Empirical Variance  : {emp_var:.6f}")
print(f"Theoretical Variance: {theory_var:.6f}")
print()

print(f"Empirical Std Dev   : {emp_std:.6f}")
print(f"Theoretical Std Dev : {theory_std:.6f}")

plt.figure(figsize=(10,6))

plt.hist(
    log_returns,
    bins=80,
    density=True,
    alpha=0.6,
    label="Simulated Log-Returns"
)

# Theoretical Normal PDF
x = np.linspace(
    log_returns.min(),
    log_returns.max(),
    1000
)

pdf = norm.pdf(
    x,
    loc=theory_mean,
    scale=theory_std
)

plt.plot(
    x,
    pdf,
    linewidth=2,
    label="Theoretical Normal Density"
)

plt.title("Distribution of GBM Log-Returns")
plt.xlabel(r"$\ln(S_T/S_0)$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)

plt.show()