"""
Historical delta-hedging backtest.

Simulates selling a European option at t=0 and dynamically delta-
hedging the position daily using realized historical prices, then
computes the terminal hedging P&L and its path. Compares hedging
using a fixed (implied/model) vol vs. rolling realized vol to show
the vol-forecast-error effect on hedge performance.
"""

import numpy as np
import pandas as pd

import black_scholes as bs


def backtest_delta_hedge(prices, K, T_years, r, sigma_hedge, option_type="call",
                          q=0.0, transaction_cost_bps=0.0, notional=1):
    """Backtest a daily-rebalanced delta hedge over a historical price path.

    Parameters
    ----------
    prices : array-like of daily spot prices (chronological), length N+1
        representing N trading days to expiry (last price = terminal spot)
    K : strike
    T_years : initial time to maturity in years (e.g. 30/252)
    r : risk-free rate
    sigma_hedge : volatility used to compute the hedge ratio (the
        "hedger's belief" about vol -- can differ from realized vol)
    option_type : 'call' or 'put'
    q : dividend yield
    transaction_cost_bps : round-trip cost in bps of notional traded,
        applied to each rebalance
    notional : number of option contracts (unit multiplier)

    Returns
    -------
    pd.DataFrame with columns: spot, tau, option_price, delta,
    shares_held, cash, portfolio_value, hedge_pnl
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    dt = T_years / (n - 1)
    taus = T_years - np.arange(n) * dt
    taus[-1] = max(taus[-1], 1e-8)  # avoid literal 0 at expiry

    option_price = bs.price(prices, K, taus, r, sigma_hedge, option_type, q)
    delta = bs.delta(prices, K, taus, r, sigma_hedge, option_type, q)

    # Hedger SHORTS the option (collects premium), and holds `delta`
    # shares of the underlying to hedge. Cash account accrues at r.
    shares_held = np.zeros(n)
    cash = np.zeros(n)
    cost = np.zeros(n)

    # t=0: sell option, buy delta[0] shares
    cash[0] = notional * option_price[0] - notional * delta[0] * prices[0]
    shares_held[0] = notional * delta[0]
    cost[0] = transaction_cost_bps / 1e4 * abs(shares_held[0] * prices[0])
    cash[0] -= cost[0]

    for t in range(1, n):
        # accrue interest on cash
        cash[t] = cash[t - 1] * np.exp(r * dt)
        # rebalance to new delta
        d_shares = notional * delta[t] - shares_held[t - 1]
        cash[t] -= d_shares * prices[t]
        shares_held[t] = notional * delta[t]
        cost[t] = transaction_cost_bps / 1e4 * abs(d_shares * prices[t])
        cash[t] -= cost[t]

    # Portfolio value = cash + stock - option liability
    option_liability = notional * option_price
    stock_value = shares_held * prices
    portfolio_value = cash + stock_value - option_liability

    # Final payoff settlement at expiry already embedded since
    # option_price[-1] equals intrinsic value (tau ~ 0)
    df = pd.DataFrame({
        "spot": prices,
        "tau": taus,
        "option_price": option_price,
        "delta": delta,
        "shares_held": shares_held,
        "cash": cash,
        "transaction_cost": cost,
        "portfolio_value": portfolio_value,
    })
    df["hedge_pnl"] = df["portfolio_value"] - df["portfolio_value"].iloc[0]
    return df


def hedge_pnl_distribution(price_paths, K, T_years, r, sigma_hedge,
                            option_type="call", q=0.0,
                            transaction_cost_bps=0.0):
    """Run backtest_delta_hedge across many simulated/historical price
    paths and return the terminal hedge P&L for each -- useful for
    building a P&L distribution / hedging-error histogram."""
    terminal_pnls = []
    for path in price_paths:
        df = backtest_delta_hedge(path, K, T_years, r, sigma_hedge,
                                   option_type, q, transaction_cost_bps)
        terminal_pnls.append(df["portfolio_value"].iloc[-1])
    return np.array(terminal_pnls)
