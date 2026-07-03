import numpy as np
import matplotlib.pyplot as plt

theta_values = [0.5, 2.0, 5.0]

sigma = 0.5
X0 = 3.0

T = 5
N = 5000

dt = T / N

t = np.linspace(0, T, N + 1)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for idx, theta in enumerate(theta_values):

    X = np.zeros((10, N + 1))
    X[:, 0] = X0

    for k in range(N):
        dW = np.random.normal(0, np.sqrt(dt), 10)

        X[:, k + 1] = (
            X[:, k]
            - theta * X[:, k] * dt
            + sigma * dW
        )

    for path in X:
        axes[idx].plot(t, path)

    axes[idx].set_title(f"Theta = {theta}")
    axes[idx].grid(True)

plt.tight_layout()
plt.show()