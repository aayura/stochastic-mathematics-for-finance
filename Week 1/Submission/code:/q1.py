import numpy as np

n = 10000

LB = np.random.rand(n) < 0.20

LA = np.zeros(n, dtype=bool)

for i in range(n):
    if LB[i]:
        LA[i] = np.random.rand() < 0.35
    else:
        LA[i] = np.random.rand() < 0.0375

IA = LA.astype(int)
IB = LB.astype(int)

N = IA + IB

p_inter = np.mean(LA & LB)
p_union = np.mean(LA | LB)
ExpN = np.mean(N)
Var_N = np.var(N)

print("P(LA ∩ LB):", p_inter)
print("P(LA ∪ LB):", p_union)
print("E[N]:", ExpN)
print("Var(N):", Var_N)