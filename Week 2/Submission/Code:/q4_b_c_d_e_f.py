import numpy as np
import matplotlib.pyplot as plt

S0 = 100
sigma = 0.2
K = 110
N = 100000

Z = np.random.normal(0, 1, N)

ST = S0 * np.exp(sigma * Z)

print("First 10 simulated stock prices:")
print(ST[:10])

payoff = np.maximum(ST - K, 0)

print("First 10 payoffs:")
print(payoff[:10])

call_price_estimate = np.mean(payoff)

print("Estimated expected payoff:")
print(call_price_estimate)

plt.figure(figsize=(8,5))
plt.hist(ST, bins=50)

plt.xlabel("Terminal Stock Price")
plt.ylabel("Frequency")
plt.title("Histogram of Simulated Terminal Stock Prices")
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))
plt.hist(payoff, bins=50)

plt.xlabel("Option Payoff")
plt.ylabel("Frequency")
plt.title("Histogram of Simulated Call Option Payoffs")
plt.grid(True)

plt.show()