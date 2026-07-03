import numpy as np

P = np.array([
    [0.60, 0.25, 0.05, 0.10],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.00, 0.00, 1.00]
])

n_steps = 100

states = [0]

current = 0

for _ in range(n_steps):
    current = np.random.choice([0,1,2,3], p=P[current])
    states.append(current)

print(states)