"""
End-to-end demo: NIFTY 50 data -> BS pricing/Greeks -> Monte Carlo
convergence -> historical delta-hedging backtest -> risk/scenario plots.

Run: python run_demo.py
Outputs written to ../outputs/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import black_scholes as bs
import monte_carlo as mc
import delta_hedging as dh
import data_utils as du

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 9})

# ----------------------------------------------------------------
# 1. Load NIFTY 50 data (real via yfinance if network available,
#    else synthetic GARCH-vol series -- see data_utils.py)
# ----------------------------------------------------------------
print("=" * 60)
print("1. LOADING MARKET DATA")
print("=" * 60)
data = du.load_nifty_data(start="2024-01-01", end="2026-01-01")
data["rv_21d"] = du.realized_vol(data["close"], window=21)
S0 = float(data["close"].iloc[-1])
current_rv = float(data["rv_21d"].dropna().iloc[-1])
print(f"Spot (latest): {S0:,.2f}   21d realized vol: {current_rv:.2%}")

fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(data["date"], data["close"], color="#1f4e79", lw=1)
ax1.set_ylabel("NIFTY 50 Close")
ax2 = ax1.twinx()
ax2.plot(data["date"], data["rv_21d"] * 100, color="#c0392b", lw=0.8, alpha=0.7)
ax2.set_ylabel("21d Realized Vol (%)")
ax1.set_title("NIFTY 50 Price & Rolling Realized Volatility")
fig.tight_layout()
fig.savefig(f"{OUT}/01_nifty_price_vol.png")
plt.close(fig)

# ----------------------------------------------------------------
# 2. Black-Scholes pricing + Greeks surface
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("2. BLACK-SCHOLES PRICING & GREEKS")
print("=" * 60)
r = 0.065          # approx Indian risk-free rate (T-bill proxy)
sigma = current_rv
K_atm = round(S0 / 50) * 50   # nearest 50-point strike, NIFTY convention
T = 30 / 365

atm = bs.all_greeks(S0, K_atm, T, r, sigma, "call")
print(f"ATM Call (K={K_atm}, T=30d, sigma={sigma:.2%}):")
for k, v in atm.items():
    print(f"  {k:8s}: {v:,.4f}")

# Greeks vs spot, across strikes
strikes = np.linspace(S0 * 0.85, S0 * 1.15, 200)
call_price = bs.price(S0, strikes, T, r, sigma, "call")
call_delta = bs.delta(S0, strikes, T, r, sigma, "call")
call_gamma = bs.gamma(S0, strikes, T, r, sigma)
call_vega = bs.vega(S0, strikes, T, r, sigma) / 100

fig, axes = plt.subplots(2, 2, figsize=(9, 6))
for ax, y, title, color in zip(
    axes.flat,
    [call_price, call_delta, call_gamma, call_vega],
    ["Call Price", "Delta", "Gamma", "Vega (per 1% vol)"],
    ["#1f4e79", "#2e7d32", "#c0392b", "#6a1b9a"],
):
    ax.plot(strikes, y, color=color)
    ax.axvline(S0, color="gray", ls="--", lw=0.8, label="Spot")
    ax.set_title(title)
    ax.set_xlabel("Strike")
axes[0, 0].legend(fontsize=7)
fig.suptitle(f"BS Greeks vs Strike (S0={S0:,.0f}, T=30d, sigma={sigma:.1%})")
fig.tight_layout()
fig.savefig(f"{OUT}/02_greeks_vs_strike.png")
plt.close(fig)

# ----------------------------------------------------------------
# 3. Monte Carlo pricing + convergence vs Black-Scholes
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("3. MONTE CARLO SIMULATION")
print("=" * 60)
bs_price_atm = bs.price(S0, K_atm, T, r, sigma, "call")
mc_p, mc_se = mc.mc_price(S0, K_atm, T, r, sigma, "call", n_paths=200_000,
                           seed=42, return_stderr=True)
print(f"BS price:  {bs_price_atm:.4f}")
print(f"MC price:  {mc_p:.4f}  +/- {1.96*mc_se:.4f} (95% CI)")

conv = mc.convergence_study(S0, K_atm, T, r, sigma, "call")
ns = [c[0] for c in conv]
ps = [c[1] for c in conv]
ses = [c[2] for c in conv]

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(ns, ps, yerr=[1.96 * s for s in ses], fmt="o-", capsize=3,
            color="#2e7d32", label="MC price (95% CI)")
ax.axhline(bs_price_atm, color="#c0392b", ls="--", label="BS closed-form")
ax.set_xscale("log")
ax.set_xlabel("Number of paths")
ax.set_ylabel("Option price")
ax.set_title("Monte Carlo Convergence to Black-Scholes")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/03_mc_convergence.png")
plt.close(fig)

mc_delta_val = mc.mc_delta(S0, K_atm, T, r, sigma, "call", n_paths=200_000, seed=1)
mc_gamma_val = mc.mc_gamma(S0, K_atm, T, r, sigma, "call", n_paths=200_000, seed=1)
mc_vega_val = mc.mc_vega(S0, K_atm, T, r, sigma, "call", n_paths=200_000, seed=1) / 100
print(f"MC delta: {mc_delta_val:.4f}  (BS: {atm['delta']:.4f})")
print(f"MC gamma: {mc_gamma_val:.6f}  (BS: {atm['gamma']:.6f})")
print(f"MC vega:  {mc_vega_val:.4f}  (BS: {atm['vega']:.4f})")

# ----------------------------------------------------------------
# 4. Historical delta hedging backtest
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("4. HISTORICAL DELTA HEDGING BACKTEST")
print("=" * 60)
n_days = 30
window = data["close"].iloc[-(n_days + 1):].values
K_hedge = round(window[0] / 50) * 50
T_hedge = n_days / 252

# Compare hedging with "correct" realized vol vs a mis-specified
# (understated) vol -- classic vol-forecast-error demonstration
realized_vol_window = np.std(np.diff(np.log(window))) * np.sqrt(252)
print(f"Window realized vol: {realized_vol_window:.2%}")

hedge_correct = dh.backtest_delta_hedge(
    window, K_hedge, T_hedge, r, sigma_hedge=realized_vol_window,
    option_type="call", transaction_cost_bps=2.0)
hedge_understated = dh.backtest_delta_hedge(
    window, K_hedge, T_hedge, r, sigma_hedge=realized_vol_window * 0.6,
    option_type="call", transaction_cost_bps=2.0)
hedge_overstated = dh.backtest_delta_hedge(
    window, K_hedge, T_hedge, r, sigma_hedge=realized_vol_window * 1.4,
    option_type="call", transaction_cost_bps=2.0)

print(f"Terminal hedge P&L (sigma_hedge = realized): "
      f"{hedge_correct['hedge_pnl'].iloc[-1]:,.2f}")
print(f"Terminal hedge P&L (sigma_hedge = 0.6x realized, understated): "
      f"{hedge_understated['hedge_pnl'].iloc[-1]:,.2f}")
print(f"Terminal hedge P&L (sigma_hedge = 1.4x realized, overstated): "
      f"{hedge_overstated['hedge_pnl'].iloc[-1]:,.2f}")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(hedge_correct["spot"], label="Spot", color="#1f4e79")
axes[0].axhline(K_hedge, color="gray", ls="--", lw=0.8, label=f"Strike {K_hedge}")
axes[0].set_ylabel("NIFTY Spot")
axes[0].legend(fontsize=8)
axes[0].set_title(f"Delta-Hedged Short Call: {n_days}d Backtest (K={K_hedge})")

axes[1].plot(hedge_correct["hedge_pnl"], label="sigma_hedge = realized",
             color="#2e7d32")
axes[1].plot(hedge_understated["hedge_pnl"], label="sigma_hedge = 0.6x realized",
             color="#c0392b")
axes[1].plot(hedge_overstated["hedge_pnl"], label="sigma_hedge = 1.4x realized",
             color="#6a1b9a")
axes[1].axhline(0, color="black", lw=0.6)
axes[1].set_ylabel("Hedge P&L")
axes[1].set_xlabel("Trading day")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/04_delta_hedge_pnl.png")
plt.close(fig)

# ----------------------------------------------------------------
# 5. Scenario analysis / stress P&L grid (spot x vol shocks)
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("5. SCENARIO ANALYSIS")
print("=" * 60)
spot_shocks = np.linspace(-0.10, 0.10, 9)
vol_shocks = np.linspace(-0.5, 0.5, 9)
pnl_grid = np.zeros((len(vol_shocks), len(spot_shocks)))

base_price = bs.price(S0, K_atm, T, r, sigma, "call")
for i, vs in enumerate(vol_shocks):
    for j, ss in enumerate(spot_shocks):
        new_S = S0 * (1 + ss)
        new_sigma = max(sigma * (1 + vs), 0.01)
        new_price = bs.price(new_S, K_atm, T, r, new_sigma, "call")
        pnl_grid[i, j] = new_price - base_price

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.imshow(pnl_grid, cmap="RdYlGn", aspect="auto",
               extent=[spot_shocks[0]*100, spot_shocks[-1]*100,
                       vol_shocks[0]*100, vol_shocks[-1]*100],
               origin="lower")
ax.set_xlabel("Spot Shock (%)")
ax.set_ylabel("Vol Shock (%)")
ax.set_title(f"Scenario P&L Grid: ATM Call (K={K_atm})")
fig.colorbar(im, ax=ax, label="P&L")
fig.tight_layout()
fig.savefig(f"{OUT}/05_scenario_pnl_grid.png")
plt.close(fig)

print(f"Max gain scenario: {pnl_grid.max():,.2f}")
print(f"Max loss scenario: {pnl_grid.min():,.2f}")

# ----------------------------------------------------------------
# Summary CSV
# ----------------------------------------------------------------
summary = pd.DataFrame([{
    "spot": S0, "strike_atm": K_atm, "T_days": 30, "vol_used": sigma,
    "bs_price": atm["price"], "bs_delta": atm["delta"],
    "bs_gamma": atm["gamma"], "bs_vega": atm["vega"], "bs_theta": atm["theta"],
    "mc_price": mc_p, "mc_stderr": mc_se,
    "hedge_pnl_correct_vol": hedge_correct["hedge_pnl"].iloc[-1],
    "hedge_pnl_understated_vol": hedge_understated["hedge_pnl"].iloc[-1],
    "hedge_pnl_overstated_vol": hedge_overstated["hedge_pnl"].iloc[-1],
}])
summary.to_csv(f"{OUT}/summary.csv", index=False)

print("\nAll outputs written to:", os.path.abspath(OUT))
