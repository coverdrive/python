from __future__ import absolute_import
from scipy.optimize import newton, brentq
import math
import numpy as np
import random
from gen_utils.dates_math import add_years_to_date, dates_diff_years
import pandas as pd
from operator import itemgetter
# from pandas import datetime
from scipy.interpolate import interp1d


def cc_val_func(inv):
    return lambda r, inv=inv: sum((1 + pi) * math.exp(-r * ti) - math.exp(-r * si)
                                  for pi, si, ti in inv)


def cc_deriv_func(inv):
    return lambda r, inv=inv: sum(si * math.exp(-r * si) - (1 + pi) * ti * math.exp(-r * ti)
                                  for pi, si, ti in inv)


def cc_deriv_deriv_func(inv):
    return lambda r, inv=inv: sum((1 + pi) * ti * ti * math.exp(-r * ti) - si * si * math.exp(-r * si)
                                  for pi, si, ti in inv)


def ac_val_func(inv):
    return lambda r, inv=inv: sum((1 + pi) * pow(1 + r, -ti) - pow(1 + r, -si)
                                  for pi, si, ti in inv)


def ac_deriv_func(inv):
    return lambda r, inv=inv: sum(si * pow(1 + r, - si - 1) - (1 + pi) * ti * pow(1 + r, - ti - 1)
                                  for pi, si, ti in inv)


def ac_deriv_deriv_func(inv):
    return lambda r, inv=inv: sum((1 + pi) * ti * (ti + 1) * pow(1 + r, - ti - 2) - si * (si + 1) * pow(1 + r, - si - 2)
                                  for pi, si, ti in inv)


def solve_for_rate(investments, compounding="Continuous"):
    assert compounding == "Annual" or compounding == "Continuous", "Compounding must be Annual or Continuous"

    # low = -5.0
    # high = 5.0
    # show_func(cc_val_func(investments), low, high)
    # show_func(cc_deriv_func(investments), low, high)

    # return float(brentq(
    #     val_func,
    #     low,
    #     high,
    #     xtol=1e-5,
    #     maxiter=100
    # ))

    if len(investments):
        if compounding == "Annual":
            fs = (ac_val_func(investments), ac_deriv_func(investments), ac_deriv_deriv_func(investments))
        else:
            fs = (cc_val_func(investments), cc_deriv_func(investments), cc_deriv_deriv_func(investments))

        guess = sum(pi for pi, _, _ in investments) / sum(ti - si for _, si, ti in investments)
        # print guess

        try:

            ret = float(newton(
                fs[0],
                guess,
                fprime=fs[1],
                tol=1e-10,
                maxiter=100,
                fprime2=fs[2]
            ))

        except:
            bounds = [(-0.5, 1.0), (-2.0, 5.0), (-5.0, 25.0), (-10.0, 100.0)]
            vf = fs[0]
            try:
                ret = next(float(brentq(vf, low, high, xtol=1e-10)) for low, high in bounds
                           if vf(low) * vf(high) <= 0)
            except StopIteration:
                # for pair in bounds:
                #     for val in pair:
                #         print val, vf(val)
                ret_guess = min([(val, vf(val)) for pair in bounds for val in pair], key=lambda x: abs(x[1]))[0]
                try:
                    ret = float(newton(
                        fs[0],
                        ret_guess,
                        fprime=fs[1],
                        tol=1e-10,
                        maxiter=100,
                        fprime2=fs[2]
                    ))
                except:
                        ret = None

    else:
        ret = None

    return ret


def get_moment_stats(vals, weights=[]):
    if len(vals):
        if len(weights) and sum(weights):
            wts = weights
        else:
            wts = np.ones(len(vals))
        mean = np.average(vals, weights=wts)
        stdev = math.sqrt(np.average(pow(vals - mean, 2.0), weights=wts))
        if stdev:
            skew = np.average(pow(vals - mean, 3.0), weights=wts) / pow(stdev, 3.0)
            kurtosis = np.average(pow(vals - mean, 4.0), weights=wts) / pow(stdev, 4.0) - 3.0
            sharpe = mean / stdev
        else:
            skew = None
            kurtosis = None
            sharpe = None

        lstdev = - np.average(np.minimum(vals, 0.0), weights=wts)
        if lstdev:
            lsharpe = mean / lstdev
        else:
            lsharpe = None

        ret = {
            "Count": len(vals),
            "Mean": mean,
            "Stdev": stdev,
            "Skew": skew,
            "Kurtosis": kurtosis,
            "Sharpe": sharpe,
            "LSharpe": lsharpe,
            "Sterr": stdev / math.sqrt(len(vals))
        }
    else:
        ret = {
            "Count": None,
            "Mean": None,
            "Stdev": None,
            "Skew": None,
            "Kurtosis": None,
            "Sharpe": None,
            "LSharpe": None,
            "Sterr": None
        }
    return ret


def get_series_descriptive_stats(series):
    count = series.count()
    if count:
        mean = series.mean()
        stdev = series.std()
        if stdev:
            sharpe = mean / stdev
        else:
            sharpe = None

        lstdev = -(np.minimum(series, 0.0).mean())
        if lstdev:
            lsharpe = mean / lstdev
        else:
            lsharpe = None

        ret = {
            "Count": count,
            "Mean": mean,
            "Stdev": stdev,
            "Sharpe": sharpe,
            "LSharpe": lsharpe,
            "Sterr": stdev / math.sqrt(count)
        }
    else:
        ret = {
            "Count": None,
            "Mean": None,
            "Stdev": None,
            "Sharpe": None,
            "LSharpe": None,
            "Sterr": None
        }
    return ret


def get_series_descriptive_stats_table(series):
    stats_dict = get_series_descriptive_stats(series)
    stat_names = [
        "Count",
        "Mean",
        "Stdev",
        "Sharpe",
        "LSharpe",
        "Sterr"
    ]
    return pd.DataFrame({
        "Stats": stat_names,
        "Values": [stats_dict[x] for x in stat_names]
    })


def get_simulation_date_pairs(
    num_simulations,
    sim_start_date,
    sim_end_date,
    years_range
):
    simulation_years = dates_diff_years(sim_end_date, sim_start_date)
    date_pairs = []
    for i in range(num_simulations):
        x = random.uniform(years_range * 2 / 3.0, float(simulation_years))
        start_date = add_years_to_date(sim_end_date, -x)
        y = random.uniform(years_range / 3.0, min(years_range, x))
        end_date = add_years_to_date(start_date, y)
        date_pairs.append((start_date, end_date))

    return date_pairs


def exp_mvg_avg(ts, lookback):
    return pd.ewma(np.array(ts), span=lookback)


def exp_mvg_avgs_diff(ts, l1, l2):
    return exp_mvg_avg(ts, l1) - exp_mvg_avg(ts, l2)


def wtd_mvg_avg(ts, leng):
    ret = np.zeros(len(ts), float)
    for i in range(len(ts)):
        running_sum = 0.0
        running_wts_sum = 0.0
        lookback = min(i + 1, leng)
        for j in range(lookback):
            wt = lookback - j
            running_sum += ts[i - j] * wt
            running_wts_sum += wt
        ret[i] = running_sum / running_wts_sum
    return ret


def mvg_func(ts, leng, f):
    return np.array([f(ts[max(0, i - leng + 1):(i + 1)]) for i in range(len(ts))])


def mvg_sum(ts, leng):
    return mvg_func(ts, leng, sum)


def mvg_min(ts, leng):
    return mvg_func(ts, leng, min)


def mvg_max(ts, leng):
    return mvg_func(ts, leng, max)


def running_func(ts, f):
    return np.array([f(ts[:(i + 1)]) for i in range(len(ts))])


def running_sum(ts):
    return running_func(ts, sum)


def running_min(ts):
    return running_func(ts, min)


def running_max(ts):
    return running_func(ts, max)


def get_angle_diff(a1, a2):
    diff = (a1 - a2) % 360.0
    return diff if (diff < 180.0) else (diff - 360.0)


def is_angle_within(x, a, b):
    x1 = get_angle_diff(x, 0.0)
    a1 = get_angle_diff(a, 0.0)
    b1 = get_angle_diff(b, 0.0)
    cond1 = (x1 >= a1)
    cond2 = (x1 < b1)
    return (cond1 and cond2) if (a1 < b1) else (cond1 or cond2)


def get_spike_control_func(thresh, steep):
    def scf(x):
        if x < 0.0:
            ret = 1.0
        elif x < thresh * steep:
            ret = 1.0 - (1.0 - steep) / (steep * thresh) * x
        elif x < thresh:
            ret = steep / ((1.0 - steep) * thresh) * (thresh - x)
        else:
            ret = 0.0
        return ret
    return scf


def get_large_moves(
    ts,
    min_days=10,
    count=50,
    overlap=False,
    descending=True,
    rel=True,
    slope_criterion=True
):
    size = len(ts)
    dates = ts.index
    all_res = []
    columns = ["Start", "End", "Days", "Start Val", "End Val", "Change"]
    for i, (d1, v1) in enumerate(ts.iteritems()):
        if v1:
            this_res = []
            for j in range(i + min_days, min(i + max(20, min_days * 20) + 1, size)):
                d2 = dates[j]
                v2 = ts[j]
                if (v2 > v1 and descending) or (v2 < v1 and not descending):
                    this_res.append([
                        # datetime.date(d1),
                        # datetime.date(d2),
                        d1,
                        d2,
                        j - i,
                        v1,
                        v2,
                        (v2 - v1) / (v1 if rel else 1.0) /
                        ((j - i) if slope_criterion else 1.0)
                    ])
            this_res.sort(key=itemgetter(5), reverse=descending)
            take_res = this_res[:count]
            all_res += take_res

    all_res.sort(key=itemgetter(5), reverse=descending)
    if not overlap:
        flags = [True] * len(all_res)
        for i, res in enumerate(all_res):
            if flags[i]:
                for j in range(i + 1, len(all_res)):
                    if flags[j] and all_res[j][0] <= res[1] and all_res[j][1] >= res[0]:
                        flags[j] = False
        filtered_res = [res for res, flag in zip(all_res, flags) if flag]
    else:
        filtered_res = all_res

    df = pd.DataFrame(filtered_res[:count], columns=columns)

    return df


def get_fine_knot_points(floor_x, floor_y, peak_x, peak_y, num_slopes):
    slope = (peak_y - floor_y) / (peak_x - floor_x)
    l = [(floor_x, floor_y), (peak_x, peak_y)]
    x_interval = (peak_x - floor_x) / (num_slopes * 2)
    # Slopes (numbering 2n) are:
    # s*2/(n+1), s*4/(n+1), ..., s*2*n/(n+1),
    # s*2*n/(n+1), ..., s*4/(n+1), s*2/(n+1)
    # where s = (peak_y - floor_y) / (peak_x - floor_x)
    # and n = num_slopes
    for i in xrange(num_slopes - 1):
        cum_slope = slope * (i + 2) / (num_slopes + 1)
        delta_x = (i + 1) * x_interval
        delta_y = delta_x * cum_slope
        l.append((floor_x + delta_x, floor_y + delta_y))
        l.append((peak_x - delta_x, peak_y - delta_y))
    return sorted(l, key=lambda x: x[0])


def get_gen_bell_curve(
    l_floor_x,
    r_floor_x,
    l_ceil_x,
    r_ceil_x,
    l_floor_y=0.0,
    r_floor_y=0.0,
    ceil_y=1.0,
    num_slopes=5
):
    knots1 = get_fine_knot_points(l_floor_x, l_floor_y, l_ceil_x, ceil_y, num_slopes)
    knots2 = get_fine_knot_points(r_floor_x, r_floor_y, r_ceil_x, ceil_y, num_slopes)
    x, y = zip(*(sorted(knots1 + knots2, key=itemgetter(0))))
    f = interp1d(x, y, kind='linear', assume_sorted=True)

    def ret_func(inp):
        if inp >= l_floor_x and inp <= r_floor_x:
            ret = f(inp)
        elif inp < l_floor_x:
            ret = l_floor_y
        else:
            ret = r_floor_y
        return ret

    return ret_func


if __name__ == '__main__':
    # f = lambda x: 3 * x + 5
    # show_func(f, -10.0, +5.0)
    # x = mvg_sum([3, 5, 2, 5, 1, 2], 3)
    # y = running_sum([3, 5, 2, 5, 1, 2])
    # print x
    # print y
    from libs.data_files_read_write import get_ticker_raw_data_from_file
    ticker = "SPY"
    ts = get_ticker_raw_data_from_file(ticker)["pivot"]
    # from IPython.terminal.embed import InteractiveShellEmbed
    # InteractiveShellEmbed()()
    min_days = 3
    count = 20
    overlap = False
    descending = False
    rel = True
    slope_criterion = True
    res = get_large_moves(
        ts,
        min_days=min_days,
        count=count,
        overlap=overlap,
        descending=descending,
        rel=rel,
        slope_criterion=slope_criterion,
        tickers=[]
    )
    print res
