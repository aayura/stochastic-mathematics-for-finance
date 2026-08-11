"""
Black-Scholes closed-form pricing and Greeks for European options.

All functions are vectorized (numpy-friendly) so they work on scalars
or arrays of strikes/spots/maturities alike.
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute d1, d2 for the Black-Scholes formula.

    S : spot price
    K : strike price
    T : time to maturity (years)
    r : risk-free rate (annualized, continuous compounding)
    sigma : volatility (annualized)
    q : continuous dividend yield
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Guard against T=0 or sigma=0 (avoid div-by-zero -> NaN storms)
    T_safe = np.where(T <= 0, 1e-10, T)
    sigma_safe = np.where(sigma <= 0, 1e-10, sigma)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe ** 2) * T_safe) / (
        sigma_safe * np.sqrt(T_safe)
    )
    d2 = d1 - sigma_safe * np.sqrt(T_safe)
    return d1, d2


def price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under Black-Scholes."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if option_type == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if option_type == "call":
        return disc_q * norm.cdf(d1)
    elif option_type == "put":
        return disc_q * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    T_safe = np.where(np.asarray(T) <= 0, 1e-10, T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T_safe))


def vega(S, K, T, r, sigma, q=0.0):
    """Vega per 1.00 (100%) change in vol. Divide by 100 for per-1% vega."""
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    T_safe = np.where(np.asarray(T) <= 0, 1e-10, T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T_safe)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """Theta per year. Divide by 365 for per-day theta."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    T_safe = np.where(np.asarray(T) <= 0, 1e-10, T)
    term1 = -(S * disc_q * norm.pdf(d1) * sigma) / (2 * np.sqrt(T_safe))

    if option_type == "call":
        term2 = -r * K * disc_r * norm.cdf(d2)
        term3 = q * S * disc_q * norm.cdf(d1)
        return term1 + term2 + term3
    elif option_type == "put":
        term2 = r * K * disc_r * norm.cdf(-d2)
        term3 = -q * S * disc_q * norm.cdf(-d1)
        return term1 + term2 + term3
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """Rho per 1.00 (100%) change in rate. Divide by 100 for per-1% rho."""
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if option_type == "call":
        return K * T * disc_r * norm.cdf(d2)
    elif option_type == "put":
        return -K * T * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Convenience: return price + full Greek set in one dict."""
    return {
        "price": price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q) / 100,   # per 1% vol
        "theta": theta(S, K, T, r, sigma, option_type, q) / 365,  # per day
        "rho": rho(S, K, T, r, sigma, option_type, q) / 100,  # per 1% rate
    }


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                 tol=1e-6, max_iter=100):
    """Newton-Raphson implied volatility solver with bisection fallback."""
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        p = price(S, K, T, r, sigma, option_type, q)
        v = vega(S, K, T, r, sigma, q)
        diff = market_price - p
        if abs(diff) < tol:
            return sigma
        if v < 1e-10:
            break
        sigma += diff / v
        if sigma <= 0:
            sigma = 0.01

    # Bisection fallback if Newton-Raphson didn't converge
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        p = price(S, K, T, r, mid, option_type, q)
        if abs(p - market_price) < tol:
            return mid
        if p > market_price:
            hi = mid
        else:
            lo = mid
    return mid
