def __makeplot_comparison_ccf(rot0, acc0, baz, twin_sec=5, twin_overlap=0.5, fmin=None, fmax=None):

    import matplotlib.pyplot as plt
    from obspy.signal.cross_correlation import correlate
    from obspy.signal.rotate import rotate_ne_rt
    from numpy import linspace

    def __cross_correlation_function_windows(arr1, arr2, dt, Twin, overlap=0, demean=True):

        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import arange, array, roll, linspace

        N = len(arr1)
        n_interval = int(Twin/dt)
        n_overlap = int(overlap*Twin/dt)

        # time = arange(0, N*dt, dt)

        times, samples = [], []
        n1, n2 = 0, n_interval
        while n2 <= N:
            samples.append((n1, n2))
            times.append(int(n1+(n2-n1)/2)*dt)
            n1 = n1 + n_interval - n_overlap
            n2 = n2 + n_interval - n_overlap

        cc, mm, ss = [], [], []
        for _n, (n1, n2) in enumerate(samples):

            _arr1 = arr1[n1:n2]
            _arr2 = arr2[n1:n2]

            num = len(_arr1)

            ccf = correlate(_arr1, _arr2, num, demean=demean, normalize='naive', method='fft')
            shift, val = xcorr_max(ccf)

            cc.append(ccf)
            mm.append(val)
            ss.append(shift)

        num = int(len(cc[0]))
        tlags = linspace(-num*dt, num*dt, num)

        return array(times), array(cc), tlags, array(ss), array(mm)

    rot = rot0.copy()
    acc = acc0.copy()

    Nrow, Ncol = 3, 1

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 6), sharex=True)

    acc_scaling, acc_unit = 1e3, f"mm/s$^2$"
    rot_scaling, rot_unit = 1e6, f"$\mu$rad/s"

    lw = 1

    font = 12

    acc_z = acc.select(channel="*Z")[0].data
    rot_z = rot.select(channel="*Z")[0].data

    acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data, acc.select(channel="*E")[0].data, baz)
    rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data, rot.select(channel="*E")[0].data, baz)

    rot_z*=rot_scaling
    rot_r*=rot_scaling
    rot_t*=rot_scaling

    acc_z*=acc_scaling
    acc_r*=acc_scaling
    acc_t*=acc_scaling

    acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
    acc_r_max = max([abs(min(acc_r)), abs(max(acc_r))])
    acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])

    rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
    rot_r_max = max([abs(min(rot_r)), abs(max(rot_r))])
    rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])

    dt = rot[0].stats.delta


    cmap = plt.get_cmap("coolwarm", 12)

    ttt0, ccf0, tlags0, shifts0, maxima0 = __cross_correlation_function_windows(-acc_t, rot_z, dt, twin_sec, overlap=twin_overlap, demean=True)
    ttt1, ccf1, tlags1, shifts1, maxima1 = __cross_correlation_function_windows(acc_z, rot_t, dt, twin_sec, overlap=twin_overlap, demean=True)
    ttt2, ccf2, tlags2, shifts2, maxima2 = __cross_correlation_function_windows(acc_r, rot_z, dt, twin_sec, overlap=twin_overlap, demean=True)


    cm0 = ax[0].pcolormesh(ttt0, tlags0, ccf0.T, rasterized=True, vmin=-1, vmax=1, cmap=cmap)
    ax[0].set_ylim(min(tlags0), max(tlags0))
    ax[0].set_xlim(min(ttt0), max(ttt0))
    ax[0].scatter(ttt0, shifts0*dt, s=2, color="k", alpha=0., zorder=4, label="-1x ACC-T & ROT-Z")

    ax[1].pcolormesh(ttt1, tlags1, ccf1.T, rasterized=True, vmin=-1, vmax=1, cmap=cmap)
    ax[1].set_ylim(min(tlags1), max(tlags1))
    ax[1].set_xlim(min(ttt1), max(ttt1))
    ax[1].scatter(ttt1, shifts1*dt, s=2, color="k", alpha=0., zorder=4, label="ACC-Z & ROT-T")

    ax[2].pcolormesh(ttt2, tlags2, ccf2.T, rasterized=True, vmin=-1, vmax=1, cmap=cmap)
    ax[2].set_ylim(min(tlags2), max(tlags2))
    ax[2].set_xlim(min(ttt2), max(ttt2))
    ax[2].scatter(ttt2, shifts2*dt, s=2, color="k", alpha=0., zorder=4, label="ACC-R & ROT-Z")


    for i in range(3):
        ax[i].legend(loc=2, ncols=4)
        ax[i].grid(which="both", alpha=0.5)
        ax[i].set_ylabel(f"Lagtime (s)", fontsize=font)

    ax[2].set_xlabel("Time (s)", fontsize=font)

    tbeg = acc[0].stats.starttime
    ax[0].set_title(f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC  |  f = {fmin}-{fmax} Hz  |  BAz = {round(baz, 1)}°  |  ED = {round(dist/1000,1)} km  |  T = {twin_sec}s ({int(100*twin_overlap)}%)")

    ## add colorbar
    cax = ax[Nrow-1].inset_axes([0.8, -0.3, 0.2, 0.1], transform=ax[Nrow-1].transAxes)
    cb = plt.colorbar(cm0, cax=cax, shrink=0.4, location='bottom', orientation='horizontal')
    cm0.set_clim(-1, 1)
    cb.set_label("CC-Coeff.", labelpad=-40)

    plt.show();
    return fig