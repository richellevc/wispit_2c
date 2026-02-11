
import numpy as np
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from statsmodels.graphics.plot_grids import scatter_ellipse


def fit_gaussian(image, x0=None, y0=None, xstd=2, ystd=2, theta=0, amp=None, set_bounds=None, verbose=True):
    """
    Fit a 2D Gaussian to the image.
    """
    if amp is None:
        amp = np.max(image)

    if x0 is None:
        x0 = np.unravel_index(np.argmax(image), shape=image.shape)[0]

    if y0 is None:
        y0 = np.unravel_index(np.argmax(image), shape=image.shape)[1]

    if set_bounds is None:
        bounds = {}
        if verbose:
            print("\nFitting with no bounds.")
    else:
        bounds = set_bounds
        # make sure initial parameters are within bounds
        if "x_mean" in bounds:
            if x0 < bounds["x_mean"][0] or x0 > bounds["x_mean"][1]:
                x0 = (bounds["x_mean"][0] + bounds["x_mean"][1]) / 2
                if verbose:
                    print(f"x0 of {x0} was out of bounds, setting to mean of bounds")
        if "y_mean" in bounds:
            if y0 < bounds["y_mean"][0] or y0 > bounds["y_mean"][1]:
                y0 = (bounds["y_mean"][0] + bounds["y_mean"][1]) / 2
                if verbose:
                    print(f"y0 of {y0} was out of bounds, setting to mean of bounds")
        if "x_stddev" in bounds:
            if xstd < bounds["x_stddev"][0] or xstd > bounds["x_stddev"][1]:
                xstd = (bounds["x_stddev"][0] + bounds["x_stddev"][1]) / 2
                if verbose:
                    print(f"xstd of {xstd} was out of bounds, setting to mean of bounds")
        else:
            bounds["x_stddev"] = (1.1754943508222875e-38, None) # default bounds
        if "y_stddev" in bounds:
            if ystd < bounds["y_stddev"][0] or ystd > bounds["y_stddev"][1]:
                ystd = (bounds["y_stddev"][0] + bounds["y_stddev"][1]) / 2
                if verbose:
                    print(f"ystd of {ystd} was out of bounds, setting to mean of bounds")
        else:
            bounds["y_stddev"] = (1.1754943508222875e-38, None) # default bounds

        if verbose:
            print("\nUsing bounds:")
            for key, value in bounds.items():
                print(f'{key}: {value}')

    if verbose:
        # print initial parameters
        print(f'\nInitial parameters: x0={x0}, y0={y0}, xstd={xstd}, ystd={ystd}, theta={theta}, amp={amp}')

    gauss_init = models.Gaussian2D(amplitude=amp,
                                   x_mean=x0,
                                   y_mean=y0,
                                   x_stddev=xstd,
                                   y_stddev=ystd,
                                   theta=theta,
                                   bounds=bounds)

    y_fit, x_fit = np.mgrid[:image.shape[1], :image.shape[0]]

    if set_bounds is None:
        fit_gauss = fitting.LMLSQFitter(calc_uncertainties=True)

    else:
        fit_gauss = fitting.TRFLSQFitter(calc_uncertainties=True)

    gauss = fit_gauss(gauss_init, x_fit, y_fit, image)

    return gauss, fit_gauss.fit_info


def weighted_avg_and_std(values, values_std):
    """
    Return weighted mean and weighted standard deviation
    of values with standard deviation values_std.
    See https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance-defined_weights
    and https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Correcting_for_over-_or_under-dispersion
    """
    n = len(values)
    weights = 1.0 / values_std**2
    wmean = np.sum(values * weights) / np.sum(weights)
    wvar = 1.0 / np.sum(weights)    # weighted variance
    wstd = np.sqrt(wvar)            # weighted standard deviation

    # Calculate reduced chi-squared
    chisq = (1.0 / (n-1)) * np.sum((values - wmean)**2 / values_std**2)
    # Get standard error of the weighted mean, scale corrected
    wstd_cor = np.sqrt(chisq * wvar)

    #print(wstd, wstd_cor)

    return wmean, wstd_cor


def sample_magnitude(fluxratio, fluxratio_err):
    """
    Draw 1000000 samples from the flux ratio and flux ratio error distributions and calculate the magnitude and magnitude
    error for each sample. Return the median and upper and lower 1-sigma values of the magnitude distribution.

    :param fluxratio:
    :param fluxratio_err:
    :return:
    """
    loc = fluxratio
    scale = fluxratio_err
    a = 0             # lower bound
    b = np.inf        # upper bound
    a_transformed, b_transformed = (a - loc) / scale, (b - loc) / scale
    rv = truncnorm(a_transformed, b_transformed, loc=loc, scale=scale)
    samples = rv.rvs(1000000)
    mags = -2.5 * np.log10(samples)
    sigma_l = np.median(mags) - np.percentile(mags, 15.9)
    sigma_u = np.percentile(mags, 84.1) - np.median(mags)
    return np.median(mags), sigma_l, sigma_u


def sample_percentile_asymmetric(median, err_lower, err_upper, size=1000):
    """Sample from an asymmetric distribution based on 16th and 84th percentiles."""
    # ensure size is an int
    size = int(size)

    n_lower = size // 2
    n_upper = size - n_lower

    # Left side: truncated normal below median
    left = truncnorm.rvs(a=-np.inf, b=0, loc=median, scale=err_lower, size=n_lower)

    # Right side: truncated normal above median
    right = truncnorm.rvs(a=0, b=np.inf, loc=median, scale=err_upper, size=n_upper)

    return np.concatenate([left, right])