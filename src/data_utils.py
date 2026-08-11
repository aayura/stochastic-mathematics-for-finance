"""
Market data utilities.

load_nifty_data() tries yfinance first (real NIFTY 50 data, needs
internet). If yfinance is unavailable or the fetch fails, it falls
back to a synthetic GBM-with-vol-clustering series calibrated to
realistic NIFTY parameters (drift ~12% p.a., vol ~14% p.a., with a
GARCH-like vol regime so the series isn't flat-vol i.i.d.) so the rest
of the pipeline (pricing, hedging, backtest) always has data to run on.
"""

import numpy as np
import pandas as pd


def load_nifty_data(start="2023-01-01", end="2026-01-01", use_real=True):
    """Load NIFTY 50 daily close prices.

    Parameters
    ----------
    start, end : str, date range
    use_real : bool, try yfinance first if True

    Returns
    -------
    pd.DataFrame with columns ['date', 'close'], daily frequency
    (business days only), plus a boolean flag on whether real data
    was used, printed to stdout.
    """
    if use_real:
        try:
            import yfinance as yf
            df = yf.download("^NSEI", start=start, end=end, progress=False)
            if df is not None and len(df) > 50:
                out = df[["Close"]].reset_index()
                out.columns = ["date", "close"]
                print(f"[data_utils] Loaded {len(out)} real NIFTY 50 "
                      f"rows from yfinance.")
                return out
        except Exception as e:
            print(f"[data_utils] yfinance unavailable/failed ({e}); "
                  f"falling back to synthetic data.")

    return _synthetic_nifty(start, end)


def _synthetic_nifty(start, end, seed=7):
    """Generate a synthetic NIFTY-50-like series using a GARCH(1,1)-style
    volatility process feeding a GBM-ish return process, so realized
    vol clusters the way real index vol does (calmer periods,
    occasional stress spikes) rather than being i.i.d.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    S0 = 21_000.0          # rough NIFTY 50 level
    mu_annual = 0.12        # long-run drift
    long_run_vol = 0.14     # long-run annualized vol
    dt = 1 / 252

    # GARCH(1,1) parameters (annualized vol as the driven quantity)
    omega = 0.02
    alpha = 0.10
    beta = 0.85

    vol = np.empty(n)
    vol[0] = long_run_vol
    shocks = rng.standard_normal(n)

    log_returns = np.empty(n)
    for t in range(n):
        if t > 0:
            var_t = (omega * long_run_vol ** 2
                     + alpha * (log_returns[t - 1] ** 2) / dt
                     + beta * vol[t - 1] ** 2)
            vol[t] = np.sqrt(var_t)
        drift = (mu_annual - 0.5 * vol[t] ** 2) * dt
        log_returns[t] = drift + vol[t] * np.sqrt(dt) * shocks[t]

    log_path = np.cumsum(log_returns)
    close = S0 * np.exp(log_path)

    out = pd.DataFrame({"date": dates, "close": close})
    print(f"[data_utils] Generated {len(out)} synthetic NIFTY-50-like "
          f"rows (GARCH(1,1) vol process, no internet access in this "
          f"sandbox). Swap in real data with load_nifty_data(use_real=True) "
          f"once yfinance can reach the network.")
    return out


def realized_vol(close_prices, window=21, annualize_factor=252):
    """Rolling annualized realized (close-to-close) volatility."""
    log_ret = np.log(close_prices / close_prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(annualize_factor)
