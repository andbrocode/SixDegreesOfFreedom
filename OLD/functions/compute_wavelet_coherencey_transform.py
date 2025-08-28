def __compute_wavelet_coherency_transform(times, arr1, arr2, tdelta, fmax_limit=None, normalize=True, plot=True):

    from pycwt import wct, xwt, Morlet, ar1, significance
    from numpy import std, nanmean, nan, nanmax, nanmin, nanvar, ones, nan_to_num, zeros, pi
    import matplotlib.pyplot as plt

    if len(arr1) != len(arr1):
        print(" -> different lenght of arrays!")
        return

    def __mask_cone(arr2d, ff, thresholds, fill=nan):
        mask = ones(arr2d.shape)
        for k in range(arr2d.shape[0]):  ##85
            for l in range(arr2d.shape[1]):  ## 1401
                 if ff[k] < thresholds[l]:
                    mask[k,l] = fill
        return mask

    def __mask_smaller_threshold(arr2d, thres, fill=nan):
        mask = ones(arr2d.shape)
        for k in range(arr2d.shape[0]):
            for l in range(arr2d.shape[1]):
                if arr2d[k,l] < thres:
                    mask[k,l] = fill
        return mask

    def __mask_bigger_threshold(arr2d, thres, fill=nan):
        mask = ones(arr2d.shape)
        for k in range(arr2d.shape[0]):
            for l in range(arr2d.shape[1]):
                if arr2d[k,l] > thres:
                    arr2d[k,l], mask[k,l] = fill
        return mask

    def __mask_unequal_threshold(arr2d, thres, fill=nan, tolerance=0):
        mask = ones(arr2d.shape)
        for k in range(arr2d.shape[0]):
            for l in range(arr2d.shape[1]):
                if arr2d[k,l] < (thres-tolerance) or arr2d[k,l] > (thres+tolerance):
                    mask[k,l] = fill
        return mask

    ## specify parameters
    N = len(arr1)
    dt = tdelta
    df = 1/dt
    significance_threshold = 0.95

    ## normalize, if desired
    if normalize:
        arr1 /= nanmax(abs(arr1))
        arr2 /= nanmax(abs(arr2))

    ## create mother wavelet
    mother_wavelet = Morlet(6)
    s0_set = 2 * dt  # Starting scale
    dj_set = 1 / 12  # Twelve sub-octaves per octaves
    J_set = int(7 / dj_set)  # Seven powers of two with dj sub-octaves
    #print(s0_set, dj_set, J_set)

    ## compute variance of array
    variance1 = nanvar(arr1)
    variance2 = nanvar(arr2)

    wave_wct, phases, cone_p, ff_wct, sig = wct(
    # wave_wct, scales, coi, ff_wct, fftfreqs = wct(
                                                arr1,
                                                arr2,
                                                dt=dt,
                                                dj=dj_set,
                                                s0=s0_set,
                                                sig=True,
                                                J=J_set,
                                                significance_level=significance_threshold,
                                                wavelet=mother_wavelet,
                                                normalize=normalize,
                                            )
    cone_f = 1/cone_p

    ## building cone
    mask_cone = __mask_cone(wave_wct, ff_wct, cone_f, fill=nan)

    ## compute absolute power
    wct_power = abs(wave_wct)**2

    print(variance1, variance2)
    ## Lag-1 autocorrelation for red noise
    alpha, _, _ = ar1(arr1)

    ## compute significance test
    signif, theor_red_noise_fft = significance(
                                                variance1,
                                                dt,
                                                phases,
                                                sigma_test=0,
                                                alpha=alpha,
                                                significance_level=significance_threshold,
                                                dof=-1,
                                                wavelet=mother_wavelet,
                                                )

    signif /= nanmax(signif.reshape((1,signif.size))[0])

    ## replace values below threshold with nan values
    mask_signif = __mask_smaller_threshold(signif, significance_threshold, fill=nan)

    ## filter phases with threshold
    phase_threshold = 0
    phase_tolerance = 0.5
    mask_phases = __mask_unequal_threshold(phases, phase_threshold, fill=nan, tolerance=phase_tolerance)
    # phases = __filter_smaller_threshold(phases, 3., fill=nan)


    ## filter power
    # wct_power_masked = wct_power * mask_phases * mask_cone
    wct_power_masked = wct_power * mask_signif * mask_cone

    ## compute global power along both axes
    global_power_f = nan_to_num(nanmean(wct_power_masked,axis=1), nan=0)
    global_power_t = nan_to_num(nanmean(wct_power_masked,axis=0), nan=0)



    ## ____________________________________________________
    ## plotting
    if plot:

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        N = 7

        fig, ax = plt.subplots(N, 1, figsize=(15,10))

        caxs = []

        plt.subplots_adjust(hspace=0.3)

        ax[0].plot(times, arr1, alpha=1, color="black", lw=1)
        ax[0].set_xlim(min(times), max(times))
        ax[0].set_ylim(-1,1)
        ax[0].set_ylabel("norm. \n Amp. (rad/s)")

        ax[1].plot(times, arr2, alpha=1, color="tab:red", lw=1)
        ax[1].set_xlim(min(times), max(times))
        ax[1].set_ylim(-1,1)
        ax[1].set_ylabel("norm. \n Amp. (rad/s)")

        ca2 = ax[2].pcolormesh(times, ff_wct, wct_power, vmin=0, vmax=1)
        ax[2].set_ylabel("f (Hz)")
        ax[2].plot(times, cone_f, color="white")
        ax[2].set_ylim(min(ff_wct), max(ff_wct))

        ca3 = ax[3].pcolormesh(times, ff_wct, phases, cmap="Spectral", vmin=-pi, vmax=pi)
        # cs = ax[3].contourf(times, ff_wct, mask_phases, cmap="binary_r", vmin=0, vmax=1)
        ax[3].contour(times, ff_wct, nan_to_num(mask_phases), linewidths=0.5, colors="k", levels=1)
        ax[3].set_ylabel("f (Hz)")
        ax[3].plot(times, cone_f, color="white")
        ax[3].set_ylim(min(ff_wct), max(ff_wct))

        ca4 = ax[4].pcolormesh(times, ff_wct, signif)
        ax[4].contour(times, ff_wct, nan_to_num(mask_signif), linewidths=0.3, colors="k", levels=1)
        ax[4].set_ylabel("f (Hz)")
        ax[4].plot(times, cone_f, color="white")
        ax[4].set_ylim(min(ff_wct), max(ff_wct))

        ca5 = ax[5].pcolormesh(times, ff_wct, wct_power_masked, vmin=0, vmax=1)
        ax[5].set_ylabel("f (Hz)")

        ax[6].plot(ff_wct, global_power_f, color="black", label="mean global power (masked)")
        ax[6].legend()
        ax[6].set_ylabel("norm. Power")
        ax[6].set_xlabel("Frequency (Hz)")
        ax[6].set_xscale("log")


        for n in range(N):
            caxs.append(make_axes_locatable(ax[n]).append_axes("right", size="2%", pad=0.05))
        cbar2 = plt.colorbar(ca2, cax=caxs[2], label="norm. Power")
        cbar3 = plt.colorbar(ca3, cax=caxs[3], label="Phase (rad)")
        cbar4 = plt.colorbar(ca4, cax=caxs[4], label="Significance")
        cbar5 = plt.colorbar(ca5, cax=caxs[5], label="norm. Power \n (masked)")
        caxs[0].remove()
        caxs[1].remove()
        caxs[6].remove()

        cbar3.set_ticks([-pi, 0, pi])
        cbar3.set_ticklabels([r"-$\pi$", "0", "$\pi$"])
        
        if fmax_limit:
            if fmax_limit*2 <= 20:
                ax[2].set_ylim(0, fmax_limit*2)
                ax[3].set_ylim(0, fmax_limit*2)
                ax[4].set_ylim(0, fmax_limit*2)
                ax[5].set_ylim(0, fmax_limit*2)
                ax[6].set_xlim(0, fmax_limit*2)

        plt.show()

    ## prepare dict for return
    out = {}
    out['times'] = times
    out['frequencies'] = ff_wct
    out['wct_power'] = wct_power
    out['cone_mask'] = mask_cone
    out['phase_mask'] = mask_phases
    out['mean_global_wct'] = global_power_f

    if plot:
        out['fig'] = fig

    return out