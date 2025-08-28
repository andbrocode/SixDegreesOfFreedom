"""
Functions for computing kernel density estimation statistics.
"""
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def get_kde_stats(_baz, _ccc, _baz_steps=5, Ndegree=180, plot=False):
    """
    Get the statistics of the kernel density estimation (KDE) of backazimuth and CC values.
    
    Args:
        _baz (array-like): Array of backazimuth values
        _ccc (array-like): Array of cross-correlation coefficients
        _baz_steps (float): Step size for backazimuth bins
        Ndegree (int): Number of degrees for angle division
        plot (bool): Whether to plot the KDE results
        
    Returns:
        dict: Dictionary containing:
            - baz_estimate: Estimated backazimuth
            - kde_max: Maximum of the KDE
            - shift: Applied shift to center data
            - kde_values: KDE probability density values
            - kde_angles: Angles corresponding to KDE values
            - kde_dev: KDE deviation
    """
    # define angles for kde and histogram
    kde_angles = np.arange(0, 361, 1)
    hist_angles = np.arange(0, 365, 5)

    # get first kde estimate to determine the shift
    kde = sts.gaussian_kde(_baz, weights=_ccc, bw_method='scott')
    kde_max = np.argmax(kde.pdf(kde_angles))

    # determine the shift with respect to 180°
    shift = 180 - kde_max

    # shift the backazimuth array to the center of the x-axis
    _baz_shifted = (_baz + shift) % 360

    # get second kde estimate
    kde_shifted = sts.gaussian_kde(_baz_shifted, weights=_ccc, bw_method='scott')
    kde_max_shifted = np.argmax(kde_shifted.pdf(kde_angles))

    # get the estimate of the backazimuth corrected for the shift
    baz_estimate = kde_max_shifted - shift

    # shift new kde
    kde_angles_new = (kde_angles - shift) % 360
    kde_values_new = (kde_shifted.pdf(kde_angles)) % 360

    # resort the new kde
    idx = np.argsort(kde_angles_new)
    kde_angles_new = kde_angles_new[idx]
    kde_values_new = kde_values_new[idx]

    # get deviation
    dev = int(np.round(np.sqrt(np.cov(_baz_shifted, aweights=_ccc)), 0))

    if plot:
        plt.figure(figsize=(10, 5))

        plt.hist(_baz, bins=hist_angles, weights=_ccc, density=True, alpha=0.5)
        plt.hist(_baz_shifted, bins=hist_angles, weights=_ccc, density=True, alpha=0.5)

        plt.plot(kde_angles, kde.pdf(kde_angles), color='tab:blue')
        
        plt.plot(kde_angles, kde_shifted.pdf(kde_angles), color='tab:orange')

        plt.plot(kde_angles_new, kde_values_new, color='k')
        
        plt.scatter([kde_max], [max(kde.pdf(kde_angles))],
                    color='w', edgecolor='tab:blue', label=f'Max: {kde_max:.0f}°')
        plt.scatter([kde_max_shifted], [kde_shifted.pdf(kde_max_shifted)],
                    color='w', edgecolor='tab:orange', label=f'Max: {kde_max_shifted:.0f}° (shifted)')
        plt.scatter([baz_estimate], [max(kde_values_new)],
                    color='w', edgecolor='k', label=f'Estimate: {baz_estimate:.0f}°')
        
        # plot line between max and estimate
        plt.plot([kde_max, kde_max], [0, max(kde.pdf(kde_angles))], color='tab:blue', ls='--')
        plt.plot([kde_max_shifted, kde_max_shifted], [0, max(kde_values_new)], color='tab:orange', ls='--')
        plt.plot([baz_estimate, baz_estimate], [0, max(kde_values_new)], color='k', ls='--')

        plt.xlabel('Backazimuth (°)')   
        plt.ylabel('Density')
        plt.title('KDE of Backazimuth weighted by the CC value')

        plt.legend()

    # output
    out = {
        'baz_estimate': baz_estimate,
        'kde_max': kde_max_shifted,
        'shift': shift,
        'kde_values': kde_values_new,
        'kde_angles': kde_angles_new,
        'kde_dev': dev,
    }

    return out
