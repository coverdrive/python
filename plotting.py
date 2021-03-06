import pylab
import numpy as np
from numpy import array, meshgrid
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator, WeekdayLocator, MONDAY
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def plot_pairs(x_vals, y_vals):
    pylab.plot(np.array(x_vals), np.array(y_vals))
    pylab.show()


def plot_func(f, low, high, count=2001):
    x_vals = np.linspace(low, high, count)
    y_vals = np.array([f(x) for x in x_vals])
    # print "Min Val = %.3f" % min(y_vals)
    # print "Max Val = %.3f" % max(y_vals)
    # print "Val at Min X = %.3f" % plot_func(min(x_vals))
    # print "Val at Max X = %.3f" % plot_func(max(x_vals))
    pylab.plot(x_vals, y_vals)
    pylab.show()


def plot_date_series(dates, ds1, ds2=None):
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    y1_range = max(ds1) - min(ds1)
    ax.set_ylim(min(ds1) - 0.02 * y1_range, max(ds1) + 0.02 * y1_range)
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_minor_locator(WeekdayLocator(MONDAY))
    ax.xaxis.set_major_formatter(DateFormatter('%b %y'))
    ax.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='0.7')
    ax.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='0.7')
    ax.plot(dates, ds1, color='red', label='Date Series 1', linestyle='-')

    if ds2 is not None and len(ds2) == len(ds1):
        ax1 = ax.twinx()
        y2_range = max(ds2) - min(ds2)
        ax.set_ylim(min(ds2) - 0.02 * y2_range, max(ds2) + 0.02 * y2_range)
        ax1.plot(dates, ds2, color='blue', label='Date Series 2', linestyle='-')

    ax.xaxis_date()
    ax.autoscale_view()
    fig.autofmt_xdate()

    plt.show()

def plot_3D(x_vals, y_vals, f, x_label, y_label, z_label, label):
    x, y = meshgrid(x_vals, y_vals)
    z = array([[f(a, b) for a in x_vals] for b in y_vals])
    # Plot the surface.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.grid(True)
    # ax.xaxis.set_major_locator(LinearLocator(21))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_locator(LinearLocator(6))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax.zaxis.set_major_locator(LinearLocator(17))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(label)
    plt.show()    
    

if __name__ == '__main__':
    from gen_utils.fin_quant import get_gen_bell_curve
    f = get_gen_bell_curve(
        l_floor_x=-90.0,
        r_floor_x=90.0,
        l_ceil_x=-35.0,
        r_ceil_x=30.0
    )
    plot_func(f, -180.0, 180.0, 1441)
