import numpy as np

P = np.array([
    [0.60, 0.25, 0.05, 0.10],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.50, 0.50, 0.00],
    [0.00, 0.00, 0.00, 1.00]
])

n_sim = 1000
n_steps = 100

final_states = []

for _ in range(n_sim):

    current = 0

    for _ in range(n_steps):
        current = np.random.choice([0,1,2,3], p=P[current])

    final_states.append(current)

final_states = np.array(final_states)

prop_state3 = np.mean(final_states == 3)

prop_class12 = np.mean(
    (final_states == 1) | (final_states == 2)
)

print("Proportion in state 3:", prop_state3)
print("Proportion in class {1,2}:", prop_class12)