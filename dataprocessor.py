import numpy as np
from scipy.optimize import curve_fit
import logging
LOG = logging.getLogger(__name__)


def fourier_fit_func(x, a, b, n):
    """
    The fourier fit function
    :param x:
    :param a: An n sized array containing the coefficients for the sine components
    :param b: An n sized array containing the coefficients for the cosine components
    :param n: The number of fourier coefficients / 2
    :return:
    """
    total = 0
    for s in range(0, n):
        total += a[s] * np.sin(s * x) + b[s] * np.cos(s * x)
    return total


def fit_func_wrapper(series_size: int):
    def internal_fit_func(x, *args):
        a, b, = list(args[:series_size]), list(args[series_size:2 * series_size])
        return fourier_fit_func(x, a, b, series_size)

    return internal_fit_func


def get_fit(x, y, series_size: int):
    """
    Extract the fit for the x, y datapoints
    :param x:
    :param y:
    :param series_size: The size of the fourier series
    :return: the fourier series parameters (and their covariance matrices)
    """
    dist = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)

    final = np.cumsum(dist)[-1]
    time = (np.cumsum(dist) / final) * 2 * np.pi
    fourier_wrapper = fit_func_wrapper(series_size)

    popt_x, pcov_x = curve_fit(fourier_wrapper, time, x[:-1], np.zeros(series_size * 2), maxfev=100_000)
    popt_y, pcov_y = curve_fit(fourier_wrapper, time, y[:-1], np.zeros(series_size * 2), maxfev=100_000)
    return popt_x, popt_y, pcov_x, pcov_y
