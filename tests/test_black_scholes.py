"""
Sanity tests:
- Put-call parity holds for BS closed-form.
- Monte Carlo price converges to BS within its confidence interval.
- Deep ITM/OTM limits behave correctly.
- Implied vol solver round-trips a known price.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import black_scholes as bs
import monte_carlo as mc


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
    c = bs.price(S, K, T, r, sigma, "call")
    p = bs.price(S, K, T, r, sigma, "put")
    lhs = c - p
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-8, f"Put-call parity violated: {lhs} vs {rhs}"


def test_deep_itm_call_approaches_intrinsic():
    S, K, T, r, sigma = 1000, 10, 0.1, 0.05, 0.2
    c = bs.price(S, K, T, r, sigma, "call")
    intrinsic = S - K * np.exp(-r * T)
    assert abs(c - intrinsic) < 1e-4


def test_deep_otm_call_near_zero():
    S, K, T, r, sigma = 10, 1000, 0.1, 0.05, 0.2
    c = bs.price(S, K, T, r, sigma, "call")
    assert c < 1e-6


def test_delta_bounds():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    d_call = bs.delta(S, K, T, r, sigma, "call")
    d_put = bs.delta(S, K, T, r, sigma, "put")
    assert 0 <= d_call <= 1
    assert -1 <= d_put <= 0
    assert abs((d_call - d_put) - 1) < 1e-8  # delta_call - delta_put = 1


def test_mc_converges_to_bs():
    S, K, T, r, sigma = 100, 105, 0.5, 0.03, 0.25
    bs_p = bs.price(S, K, T, r, sigma, "call")
    mc_p, se = mc.mc_price(S, K, T, r, sigma, "call", n_paths=200_000,
                            seed=123, return_stderr=True)
    assert abs(mc_p - bs_p) < 4 * se, (
        f"MC price {mc_p} not within 4 stderr of BS {bs_p} (se={se})")


def test_implied_vol_roundtrip():
    S, K, T, r, true_sigma = 100, 95, 0.75, 0.04, 0.28
    price = bs.price(S, K, T, r, true_sigma, "call")
    recovered = bs.implied_vol(price, S, K, T, r, "call")
    assert abs(recovered - true_sigma) < 1e-4


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS: {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed.")
