import numpy as np

def returns_to_origin(d, n_steps):

    position = np.zeros(d, dtype=int)

    for _ in range(n_steps):

        coord = np.random.randint(d)
        step = np.random.choice([-1, 1])

        position[coord] += step

        if np.all(position == 0):
            return True

    return False


def estimate_return_probability(d, n_paths=1000, n_steps=1000):

    count = 0

    for _ in range(n_paths):

        if returns_to_origin(d, n_steps):
            count += 1

    return count / n_paths


for d in [1, 2, 3]:

    prob = estimate_return_probability(d)

    print(f"d={d}, estimated return proportion = {prob:.3f}")