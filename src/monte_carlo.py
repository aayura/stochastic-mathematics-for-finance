"""
Monte Carlo engine for European option pricing under GBM, with
variance reduction (antithetic variates) and Greeks via
finite-difference / pathwise estimators.
"""

import numpy as np


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q=0.0,
                        antithetic=True, seed=None):
    """Simulate GBM price paths.

    Returns array of shape (n_paths, n_steps + 1), including S0 at t=0.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        half = n_paths // 2
        z = rng.standard_normal((half, n_steps))
        z = np.vstack([z, -z])
        if z.shape[0] < n_paths:  # odd n_paths
            z = np.vstack([z, rng.standard_normal((1, n_steps))])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    log_returns = drift + diffusion

    log_paths = np.cumsum(log_returns, axis=1)
    log_paths = np.hstack([np.zeros((z.shape[0], 1)), log_paths])
    paths = S0 * np.exp(log_paths)
    return paths


def mc_price(S0, K, T, r, sigma, option_type="call", q=0.0,
             n_paths=100_000, n_steps=1, antithetic=True, seed=None,
             return_stderr=False):
    """Monte Carlo European option price (terminal-value only, no need
    for full path if n_steps=1, but path simulation supports Asian/
    barrier extensions later)."""
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, q,
                                antithetic, seed)
    S_T = paths[:, -1]

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    disc_payoff = np.exp(-r * T) * payoff
    price = disc_payoff.mean()

    if return_stderr:
        stderr = disc_payoff.std(ddof=1) / np.sqrt(len(disc_payoff))
        return price, stderr
    return price


def mc_delta(S0, K, T, r, sigma, option_type="call", q=0.0,
             n_paths=100_000, bump=1e-2, seed=None):
    """Delta via central finite-difference (common random numbers)."""
    p_up = mc_price(S0 * (1 + bump), K, T, r, sigma, option_type, q,
                     n_paths, seed=seed)
    p_down = mc_price(S0 * (1 - bump), K, T, r, sigma, option_type, q,
                       n_paths, seed=seed)
    return (p_up - p_down) / (2 * S0 * bump)


def mc_gamma(S0, K, T, r, sigma, option_type="call", q=0.0,
             n_paths=100_000, bump=1e-2, seed=None):
    """Gamma via central finite-difference (common random numbers)."""
    p_up = mc_price(S0 * (1 + bump), K, T, r, sigma, option_type, q,
                     n_paths, seed=seed)
    p_mid = mc_price(S0, K, T, r, sigma, option_type, q, n_paths, seed=seed)
    p_down = mc_price(S0 * (1 - bump), K, T, r, sigma, option_type, q,
                       n_paths, seed=seed)
    h = S0 * bump
    return (p_up - 2 * p_mid + p_down) / (h ** 2)


def mc_vega(S0, K, T, r, sigma, option_type="call", q=0.0,
            n_paths=100_000, bump=1e-3, seed=None):
    """Vega via central finite-difference (common random numbers)."""
    p_up = mc_price(S0, K, T, r, sigma + bump, option_type, q, n_paths,
                     seed=seed)
    p_down = mc_price(S0, K, T, r, sigma - bump, option_type, q, n_paths,
                       seed=seed)
    return (p_up - p_down) / (2 * bump)


def convergence_study(S0, K, T, r, sigma, option_type="call", q=0.0,
                       path_counts=(1_000, 5_000, 10_000, 50_000,
                                    100_000, 500_000), seed=42):
    """Return MC price + stderr for a range of path counts, for
    convergence-vs-Black-Scholes plots."""
    results = []
    for n in path_counts:
        p, se = mc_price(S0, K, T, r, sigma, option_type, q, n_paths=n,
                          seed=seed, return_stderr=True)
        results.append((n, p, se))
    return results
