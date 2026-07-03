import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

N = 10000
n_values = [10, 50, 200]

Zn_samples = {}

for n in n_values:

    S = np.random.binomial(n=n, p=0.5, size=N)

    Z = (S - n/2) / np.sqrt(n/4)

    Zn_samples[n] = Z

fig, axes = plt.subplots(1, 3, figsize=(15,4))

for ax, n in zip(axes, n_values):

    ax.hist(Zn_samples[n],
            bins=30,
            density=True,
            alpha=0.7)

    ax.set_title(f"n={n}")
    ax.set_xlabel("Z")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,3, figsize=(15,4))

x = np.linspace(-4,4,500)

for ax, n in zip(axes, n_values):

    ax.hist(Zn_samples[n],
            bins=30,
            density=True,
            alpha=0.7)

    ax.plot(x,
            norm.pdf(x),
            linewidth=2)

    ax.set_title(f"n={n}")

plt.tight_layout()
plt.show()

for n in n_values:

    Z = Zn_samples[n]

    print("n =", n)
    print("Mean =", np.mean(Z))
    print("Variance =", np.var(Z))