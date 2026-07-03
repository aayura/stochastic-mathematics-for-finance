import numpy as np
import matplotlib.pyplot as plt

N = 5000

x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)

inside = (x**2 + y**2 <= 1)

plt.figure(figsize=(6,6))

plt.scatter(
    x[inside],
    y[inside],
    s=5,
    label='Inside Circle'
)

plt.scatter(
    x[~inside],
    y[~inside],
    s=5,
    label='Outside Circle'
)

plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.legend()
plt.title('Monte Carlo Estimation of π')
plt.show()

