from scipy.stats import norm
from math import log, exp, sqrt
import pandas as pd
import os


def get_black_scholes_d1_d2(
    u_price,
    strike,
    expiry,
    sigma,
    r
):
    sigma_sqrt = sigma * sqrt(expiry)
    d1 = (log(u_price / strike) + (r + sigma * sigma / 2) * expiry) / sigma_sqrt
    d2 = d1 - sigma_sqrt
    return (d1, d2)


def get_black_scholes_price(
    is_call,
    u_price,
    strike,
    expiry,
    sigma,
    r
):
    d1, d2 = get_black_scholes_d1_d2(
        u_price,
        strike,
        expiry,
        sigma,
        r
    )

    if is_call:
        ret = u_price * norm.cdf(d1) - strike * exp(-r * expiry) * norm.cdf(d2)
    else:
        ret = strike * exp(-r * expiry) * norm.cdf(-d2) - u_price * norm.cdf(-d1)

    return ret


def get_black_scholes_greeks(
    is_call,
    u_price,
    strike,
    expiry,
    sigma,
    r
):
    d1, d2 = get_black_scholes_d1_d2(
        u_price,
        strike,
        expiry,
        sigma,
        r
    )
    sqrtt = sqrt(expiry)

    gamma = norm.pdf(d1) / (u_price * sigma * sqrtt)
    vega = u_price * sqrtt * norm.pdf(d1)

    if is_call:
        delta = norm.cdf(d1)
        theta = - (u_price * sigma * norm.pdf(d1)) / (2 * sqrtt) - r * strike * exp(-r * expiry) * norm.cdf(d2)
        rho = strike * expiry * exp(-r * expiry) * norm.cdf(d2)
    else:
        delta = -norm.cdf(-d1)
        theta = - (u_price * sigma * norm.pdf(d1)) / (2 * sqrtt) + r * strike * exp(-r * expiry) * norm.cdf(-d2)
        rho = -strike * expiry * exp(-r * expiry) * norm.cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }


def get_annual_option_cost_panel(
    is_call,
    sigmas,
    strike_mults,
    expiries,
    r
):
    panel = []
    for sigma in sigmas:
        rows = []
        for strike in strike_mults:
            row = []
            for expiry in expiries:
                row.append(
                    get_black_scholes_price(
                        is_call,
                        1.0,
                        strike,
                        expiry,
                        sigma,
                        r
                    ) / expiry
                )
            rows.append(row)
        panel.append(rows)
    return pd.Panel(
        panel,
        items=[("%.1f%%" % (s * 100)) for s in sigmas],
        major_axis=[("%.1f%%" % (100 * s)) for s in strike_mults],
        minor_axis=[("%.3f" % e) for e in expiries],
    )


if __name__ == "__main__":
    is_call = False
    sigmas = [x * 0.05 for x in range(2, 11)]
    strike_mults = [1.0 - x * 0.01 for x in range(26)]
    expiries = [x / 52.0 for x in range(1, 26)]
    r = 0.01
    panel = get_annual_option_cost_panel(
        is_call,
        sigmas,
        strike_mults,
        expiries,
        r
    )
    df = panel.to_frame()
    print df
    file_name = os.path.expanduser("~") + "/annual_option_cost.csv"
    df.to_csv(file_name, float_format="%.3f")
