def __compute_cwt(times, arr1, tdelta, datalabel="dat1", log=False, period=False, tscale='sec', scale_value=2, ymax=None, normalize=True, plot=True):

    from pycwt import wct, xwt, Morlet, ar1, significance, cwt
    from numpy import std, nanmean, nan, nansum, nanmax, nanmin, nanvar, ones, nan_to_num, polyfit, polyval, array, reshape, nanpercentile
    import matplotlib.pyplot as plt
    from numpy import sum as npsum

    times = array(times, dtype='float64')
    arr1 = array(arr1, dtype='float64')

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


    ## specify parameters
    N = len(arr1)
    dt = tdelta
    df = 1/dt
    significance_threshold = 0.95

    ## detrend data
    p = polyfit(times - times[0], arr1, 1)
    dat_notrend = arr1 - polyval(p, times - times[0])
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    arr1 = dat_notrend / std  # Normalized dataset

    ## create mother wavelet
    mother_wavelet = Morlet(6)
    s0_set = scale_value * dt  # Starting scale
    dj_set = 1 / 12  # Twelve sub-octaves per octaves
    J_set = int(7 / dj_set)  # Seven powers of two with dj sub-octaves
    #print(s0_set, dj_set, J_set)


    cwt, scales, ff_cwt, cone_p, fft, fftfreqs = cwt(
                                                        arr1,
                                                        dt=dt,
                                                        dj=dj_set, #0.05,
                                                        s0=s0_set, #-1,
                                                        J=J_set, #-1,
                                                        wavelet=mother_wavelet,  # u'morlet',
                                                    )

    cone_f = 1/cone_p
    pp_cwt = 1/ff_cwt

    if tscale == "min":
        times /= 60
        cone_p /= 60
        pp_cwt /= 60
        unit = "min"
    elif tscale == "hour":
        times /= 3600
        cone_p /= 3600
        pp_cwt /= 3600
        unit = "hour"
    else:
        unit = "s"

    ## building cone
    mask_cone = __mask_cone(cwt, ff_cwt, cone_f, fill=nan)


    ## get real part
    cwt_power = abs(cwt)

    ## normalize cross wavelet transform
    if normalize:
        cwt_power /= nanmax(cwt_power.reshape((1, cwt_power.size))[0])

    ## apply masks
    # cwt_power_masked = cwt_power * mask_cwt * mask_cone
    cwt_power_masked = cwt_power * mask_cone

    ## compute global cross wavelet transform power
    global_mean_cwt_f = nanmean(cwt_power_masked, axis=1)
    global_sum_cwt_f = nansum(cwt_power_masked, axis=1)

    # if normalize:
    #     global_sum_cwt_f /= max(global_sum_cwt_f)
    #     global_mean_cwt_f /= max(global_mean_cwt_f)


    ## ____________________________________________________
    ## plotting
    if plot:

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        Ncol, Nrow = 4, 3

        font = 12

        fig = plt.figure(figsize=(15, 8))

        gs = GridSpec(Nrow, Ncol, figure=fig)

        ax1 = fig.add_subplot(gs[0, :-1])
        ax2 = fig.add_subplot(gs[1:, :-1])
        ax3 = fig.add_subplot(gs[1:, 3:])

        ax1.set_xticklabels([])
        ax3.set_yticklabels([])

        plt.subplots_adjust(hspace=0.1, wspace=0.1)


        ax1.plot(times, arr1, alpha=1, color="black", lw=1, label=datalabel)

        ax1.set_xlim(min(times), max(times))
        ax1.set_ylabel("Amplitude", fontsize=font)
        ax1.legend(loc=1)



        if period:
            if log:
                ca2 = ax2.pcolormesh(
                                    times,
                                    pp_cwt,
                                    cwt_power,
                                    vmin=nanpercentile(reshape(cwt_power, cwt_power.size), 1),
                                    vmax=nanpercentile(reshape(cwt_power, cwt_power.size), 99),
                                    rasterized=True,
                                    norm="log",
                                    )
            else:
                ca2 = ax2.pcolormesh(
                                    times,
                                    pp_cwt,
                                    cwt_power,
                                    vmin=nanpercentile(reshape(cwt_power, cwt_power.size), 1),
                                    vmax=nanpercentile(reshape(cwt_power, cwt_power.size), 99),
                                    rasterized=True,
                                    )

            ax3.plot(global_mean_cwt_f, pp_cwt, color="black", label="global mean power")

            ax33 = ax3.twiny()
            ax33.plot(global_sum_cwt_f, pp_cwt, color="darkred", label="global sum power")

        else:
            if log:
                ca2 = ax2.pcolormesh(
                                    times,
                                    ff_cwt,
                                    cwt_power,
                                    vmin=nanpercentile(reshape(cwt_power, cwt_power.size), 1),
                                    vmax=nanpercentile(reshape(cwt_power, cwt_power.size), 99),
                                    rasterized=True,
                                    norm="log"
                                    )
            else:
                ca2 = ax2.pcolormesh(
                                    times,
                                    ff_cwt,
                                    cwt_power,
                                    vmin=nanpercentile(reshape(cwt_power, cwt_power.size), 1),
                                    vmax=nanpercentile(reshape(cwt_power, cwt_power.size), 99),
                                    rasterized=True,
                                    )


            ax3.plot(global_mean_cwt_f, ff_cwt, color="black", label="global mean power")

            ax33 = ax3.twiny()
            ax33.plot(global_sum_cwt_f, ff_cwt, color="darkred", label="global sum power")

        if period:
            ax2.plot(times, cone_p, color="white", ls="--")
            ax2.fill_between(times, cone_p, max(pp_cwt), color="white", alpha=0.2)
            ax2.set_ylabel(f"Period ({unit})", fontsize=font)
        else:
            ax2.plot(times, cone_f, color="white")
            ax2.fill_between(times, cone_f, min(ff_cwt), color="white", alpha=0.2)
            ax2.set_ylabel("Frequency (Hz)", fontsize=font)
            ax3.set_xlabel("Frequency (Hz)", fontsize=font)

        # ax3.legend()
        ax3.set_xlabel("global mean power", fontsize=font)
        ax33.set_xlabel("global sum power", fontsize=font, color="darkred")
        ax33.tick_params(axis='x', labelcolor="darkred")
        ax2.set_xlabel(f"Time ({unit})", fontsize=font)


        ## add colorbar
        cbar_ax = fig.add_axes([0.73, 0.75, 0.17, 0.08]) #[left, bottom, width, height]
        cb = plt.colorbar(ca2, cax=cbar_ax, orientation="horizontal", extend="both")
        cb.set_label("CWT power", fontsize=font, color="black")



        if ymax is not None:
            print(f"set frequency limit: {ymax}")
            if period:
                ax3.set_xlim(0, ymax)
                ax2.set_ylim(0, ymax)
            else:
                ax3.set_xlim(0, ymax)
                ax2.set_ylim(0, ymax)
        else:
            if period:
                ax2.set_ylim(min(pp_cwt), max(pp_cwt))
            else:
                ax2.set_ylim(min(ff_cwt), max(ff_cwt))



        plt.show();

    ## prepare dict for return
    out = {}
    out['times'] = times
    out['frequencies'] = ff_cwt
    out['cwt_power'] = cwt_power
    out['cone_mask'] = mask_cone
    out['cone'] = cone_f
    out['global_mean_cwt'] = global_mean_cwt_f
    out['global_sum_cwt'] = global_sum_cwt_f


    if plot:
        out['fig'] = fig

    return out