# Stochastic Mathematics for Finance

A quantitative finance toolkit for **derivative valuation, hedging, and risk
analysis** built on stochastic asset price models (GBM / Black-Scholes).
Applied to NIFTY 50 data.

## What's in here

| Module | Purpose |
|---|---|
| `src/black_scholes.py` | Closed-form BS pricing, full Greeks (delta, gamma, vega, theta, rho), Newton-Raphson implied vol solver |
| `src/monte_carlo.py` | GBM path simulation (antithetic variates), MC pricing, Greeks via common-random-number finite differences, convergence study |
| `src/delta_hedging.py` | Daily-rebalanced historical delta-hedging backtest with transaction costs, vol-forecast-error comparison |
| `src/data_utils.py` | NIFTY 50 data loader (real via `yfinance`, or synthetic GARCH(1,1)-vol fallback), rolling realized vol |
| `scripts/run_demo.py` | End-to-end pipeline: data → pricing → Greeks → MC convergence → hedge backtest → scenario P&L grid |
| `tests/test_black_scholes.py` | Put-call parity, MC-vs-BS convergence, delta bounds, IV round-trip |

## Quickstart

```bash
pip install -r requirements.txt
cd scripts
python run_demo.py
```

Outputs (5 plots + `summary.csv`) are written to `outputs/`.

## Using real NIFTY 50 data

`data_utils.load_nifty_data()` tries `yfinance` (ticker `^NSEI`) first. If
there's no network access, it falls back to a synthetic series generated
from a GARCH(1,1) volatility process calibrated to realistic NIFTY
parameters (~12% drift, ~14% long-run vol) — so the rest of the pipeline
still runs, but note in your writeup / demo video which mode was used.
To force real data:

```python
from data_utils import load_nifty_data
data = load_nifty_data(start="2024-01-01", end="2026-01-01", use_real=True)
```

## Design decisions

- **Vectorized BS module**: every function accepts scalars or numpy arrays,
  so Greeks-vs-strike or Greeks-vs-spot surfaces are one function call, no
  loops.
- **Antithetic variates in MC**: halves simulation variance for the same
  path count vs naive MC.
- **Greeks via common random numbers**: finite-difference Greeks (delta,
  gamma, vega) reuse the same random draws across bumped/unbumped runs to
  cancel simulation noise — otherwise FD Greeks are unusably noisy.
- **Delta hedging P&L decomposition**: cash account accrues at `r`,
  transaction costs are charged in bps of notional traded per rebalance,
  and the backtest is run at three vol assumptions (realized, 0.6x, 1.4x)
  to demonstrate the classic result that **hedging error is driven by the
  gap between realized and hedged-with vol**, not by delta hedging itself.
- **Scenario grid**: revalues the same option under a spot × vol shock
  grid (not a single stress point) to show the full P&L surface a risk
  desk would look at.

## Validation

`tests/test_black_scholes.py` checks:
- Put-call parity holds to 1e-8
- Deep ITM/OTM limits match intrinsic value / zero
- Delta bounds ∈ [0,1] (call) / [-1,0] (put), and `delta_call - delta_put = 1`
- MC price converges to BS within 4 standard errors
- Implied vol solver round-trips a known price to 1e-4

Run with:
```bash
cd tests && python test_black_scholes.py
```

## Extending this

- Swap `black_scholes.py` for a local-vol or Heston model — `monte_carlo.py`'s
  path simulator is already structured to support non-GBM drift/diffusion.
- `delta_hedging.py` generalizes to gamma/vega hedging by adding a second
  hedging instrument (another option) and solving for both hedge ratios.
- `mc_price`'s `n_steps` parameter already supports path-dependent payoffs
  (Asian, barrier) — only the payoff function in `mc_price` needs to change.
