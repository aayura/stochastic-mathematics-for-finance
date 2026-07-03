import numpy as np
import matplotlib.pyplot as plt

X0 = 3.0
theta = 2.0
sigma = 0.5

T = 5
N = 5000
n_paths = 10

dt = T / N

t = np.linspace(0, T, N + 1)

X = np.zeros((n_paths, N + 1))
X[:, 0] = X0

for k in range(N):
    dW = np.random.normal(0, np.sqrt(dt), n_paths)

    X[:, k + 1] = (
        X[:, k]
        - theta * X[:, k] * dt
        + sigma * dW
    )

stationary_std = sigma / np.sqrt(2 * theta)

plt.figure(figsize=(10,6))

for i in range(n_paths):
    plt.plot(t, X[i], lw=1)

plt.axhline(
    y=0,
    color='black',
    linestyle='--',
    label='Stationary Mean'
)

plt.fill_between(
    t,
    -2 * stationary_std,
    2 * stationary_std,
    alpha=0.2,
    label='±2σ Stationary Band'
)

plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("OU Process Paths")
plt.legend()
plt.grid(True)
plt.show()