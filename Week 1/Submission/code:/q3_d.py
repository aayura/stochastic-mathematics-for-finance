import numpy as np
import matplotlib.pyplot as plt

def random_walk(d, n_steps):
    position = np.zeros(d, dtype=int)

    path = [position.copy()]

    for _ in range(n_steps):

        coord = np.random.randint(d)

        step = np.random.choice([-1, 1])

        position[coord] += step

        path.append(position.copy())

    return np.array(path)


# Simulate for d = 1, 2, 3
n_steps = 1000

walk1 = random_walk(1, n_steps)
walk2 = random_walk(2, n_steps)
walk3 = random_walk(3, n_steps)


# ---- Plotting ----

plt.figure(figsize=(14,4))

# d = 1
plt.subplot(1,3,1)
plt.plot(walk1[:,0])
plt.title("1D Random Walk")
plt.xlabel("Time")
plt.ylabel("Position")

# d = 2
plt.subplot(1,3,2)
plt.plot(walk2[:,0], walk2[:,1])
plt.title("2D Random Walk")
plt.xlabel("x")
plt.ylabel("y")

# d = 3
ax = plt.subplot(1,3,3, projection='3d')
ax.plot(walk3[:,0], walk3[:,1], walk3[:,2])
ax.set_title("3D Random Walk")

plt.tight_layout()
plt.show()