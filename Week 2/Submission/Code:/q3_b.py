import numpy as np

N_values = [10**2, 10**3, 10**4, 10**5]

for N in N_values:
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)

    print(f"\nN = {N}")
    print("First 5 sampled points:")
    
    for i in range(5):
        print(f"({x[i]:.4f}, {y[i]:.4f})")