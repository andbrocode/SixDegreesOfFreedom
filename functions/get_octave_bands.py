def __get_octave_bands(fmin, fmax, faction_of_octave=1, plot=False):

    """
    Computing octave / one-third-octave bands

    Arguments:
        - fmin:    (float) minimum center frequency
        - fmax:    (float) maximum center frequency
        - fraction_of_octave:    (int) octave fraction (e.g. [1] = octaves, 3 = third octaves, 12 = 12th octaves)
        - plot:    (bool) show frequency bands

    Example:

    >>> flower, fupper, fcenter = __get_octave_bands(f_min, f_max, fband_type="octave", plot=False)

    """

    import matplotlib.pyplot as plt
    from acoustics.octave import Octave
    from numpy import array

    ## avoid fmin = zero
    if fmin == 0:
        print(f" -> set fmin to 1e-10")
        fmin = 1e-10

    f_lower, f_upper, f_centers = [], [], []

    _octaves = Octave(fraction=faction_of_octave, interval=None, fmin=fmin, fmax=fmax, unique=False, reference=1000.0)

    f_centers = _octaves.center
    f_lower = _octaves.lower
    f_upper = _octaves.upper

    if plot:
        plt.figure(figsize=(15, 5))
        for fl, fc, fu in zip(f_lower, f_centers, f_upper):
            plt.axvline(fu, color="r")
            plt.axvline(fl, color="r")
            plt.axvline(fc, ls="--")
            plt.axvline(fmin, color="g")
            plt.axvline(fmax, color="g")
            plt.xscale("log")
        plt.show();

    return array(f_lower), array(f_upper), array(f_centers)