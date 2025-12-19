import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt


def get_kde_stats_velocity(velocities, cc_values, plot=False):
    """
    Compute KDE statistics for velocity data (not backazimuth).
    
    Parameters:
    -----------
    velocities : array-like
        Array of velocity values
    cc_values : array-like
        Array of cross-correlation coefficients for weighting
    plot : bool, optional
        Whether to plot the KDE results
        
    Returns:
    --------
    dict
        'count': number of valid measurements
        'max': maximum velocity
        'hmhw': half maximum width
        'dev': standard deviation
        'mad': median absolute deviation
    """
    
    # initialize output
    out = {
        'count': 0,
        'max': np.nan,
        'hmhw': np.nan,
        'dev': np.nan,
        'mad': np.nan,
        'n_samples': 0,
    }

    # Filter out invalid values
    valid_mask = ~(np.isnan(velocities) | np.isnan(cc_values))
    velocities = velocities[valid_mask]
    cc_values = cc_values[valid_mask]
    
    # Count valid samples
    n_samples = len(velocities)
    
    if n_samples < 5:
        out['n_samples'] = n_samples
        return out  # Return empty dict instead of tuple
    
    # Ensure positive weights
    cc_values = np.maximum(cc_values, 1e-6)
    
    try:
        # Create KDE for velocity data
        kde = sts.gaussian_kde(velocities, weights=cc_values, bw_method='scott')

        # Create evaluation points
        eval_points = np.arange(0, 6000, 100)
        
        # Evaluate KDE
        kde_values = kde.pdf(eval_points)
        
        # Find maximum
        max_idx = np.argmax(kde_values)
        velocity_max = eval_points[max_idx]
        
        # Estimate deviation from the KDE
        # Find points where KDE is > 0.5 * max
        half_max = kde_values[max_idx] * 0.5
        valid_points = eval_points[kde_values >= half_max]

        if len(valid_points) > 5:
            hmhw = (valid_points.max() - valid_points.min()) / 2.0
        else:
            hmhw = np.nan
        
        # get standard deviation
        dev = int(np.round(np.sqrt(np.cov(velocities, aweights=cc_values)), 0))

        # get median absolute deviation
        mad = int(np.round(np.median(np.abs(velocities - velocity_max)), 0))

        # output
        out = {
            'count': len(velocities),
            'max': velocity_max,
            'hmhw': hmhw,
            'dev': dev,
            'mad': mad,
            'n_samples': n_samples,
        }
    except Exception as e:
        print(f"Error computing KDE statistics for velocity: {e}")

    if plot:
        plt.figure(figsize=(10, 5))

        plt.hist(velocities, bins=eval_points, weights=cc_values, density=True, alpha=0.5)

        plt.plot(eval_points  , kde.pdf(eval_points), color='tab:blue')

        plt.scatter([velocity_max], [max(kde.pdf(eval_points))],
                    color='k', edgecolor='tab:blue', label=f'Max: {velocity_max:.0f} m/s')
        plt.plot([velocity_max], [kde.pdf(velocity_max)],
                    color='k', ls='--')

        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Density')
        plt.title('KDE of Backazimuth weighted by the CC value')

        plt.legend()

    return out