def __makeplot_waveform_comparison_cc(rot0, acc0, baz, fmin, fmax, distance=None, twin_sec=5, twin_overlap=0.5):

    from obspy.signal.cross_correlation import correlate
    from obspy.signal.rotate import rotate_ne_rt
    from numpy import linspace, ones
    import matplotlib.pyplot as plt

    def __cross_correlation_windows(arr1, arr2, dt, Twin, overlap=0, lag=0, demean=True, plot=False):

        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import arange, array, roll

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

        cc = []
        for _n, (n1, n2) in enumerate(samples):

            _arr1 = roll(arr1[n1:n2], lag)
            _arr2 = arr2[n1:n2]
            ccf = correlate(_arr1, _arr2, 0, demean=demean, normalize='naive', method='fft')
            shift, val = xcorr_max(ccf, abs_max=False)
            cc.append(val)

        return array(times), array(cc)


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

    rot0 ,acc0, rot0_lbl, acc0_lbl = rot_z, acc_t, "ROT-Z", "ACC-T"
    rot1 ,acc1, rot1_lbl, acc1_lbl = rot_t, acc_z, "ROT_T", "ACC-Z"
    rot2 ,acc2, rot2_lbl, acc2_lbl = rot_z, -acc_r, "ROT-Z", "-1xACC-R"

    tt0, cc0 = __cross_correlation_windows(rot0 ,acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
    tt1, cc1 = __cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
    tt2, cc2 = __cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

    cmap = plt.get_cmap("coolwarm", 12)


    ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
    ax00 = ax[0].twinx()
    ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
    ax01 = ax[0].twinx()
    cm = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, label="")

    ax[0].set_ylim(-rot_z_max, rot_z_max)
    ax00.set_ylim(-acc_t_max, acc_t_max)
    ax01.set_ylim(-1, 1)
    ax01.yaxis.set_visible(False)

    ax[1].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
    ax11 = ax[1].twinx()
    ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
    ax12 = ax[1].twinx()
    ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, label="")

    ax[1].set_ylim(-rot_t_max, rot_t_max)
    ax11.set_ylim(-acc_z_max, acc_z_max)
    ax12.set_ylim(-1, 1)
    ax12.yaxis.set_visible(False)

    ax[2].plot(rot.select(channel="*N")[0].times(), rot2, label=rot2_lbl, color="tab:red", lw=lw, zorder=3)
    ax22 = ax[2].twinx()
    ax22.plot(acc.select(channel="*Z")[0].times(), acc2, label=acc2_lbl, color="black", lw=lw)
    ax23 = ax[2].twinx()
    ax23.scatter(tt2, ones(len(tt2))*-0.9, c=cc2, alpha=abs(cc2), cmap=cmap, label="")

    ax[2].set_ylim(-rot_z_max, rot_z_max)
    ax22.set_ylim(-acc_r_max, acc_r_max)
    ax23.set_ylim(-1, 1)
    ax23.yaxis.set_visible(False)

    cc0 = round(correlate(rot0, acc0, 0, demean=True, normalize='naive', method='auto')[0], 2)
    cc1 = round(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='auto')[0], 2)
    cc2 = round(correlate(rot2, acc2, 0, demean=True, normalize='naive', method='auto')[0], 2)
    cc = [cc0, cc1, cc2]

    ## sync twinx
    ax[0].set_yticks(linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-1], len(ax[0].get_yticks())))
    ax00.set_yticks(linspace(ax00.get_yticks()[0], ax00.get_yticks()[-1], len(ax[0].get_yticks())))

    ax[1].set_yticks(linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-1], len(ax[1].get_yticks())))
    ax11.set_yticks(linspace(ax11.get_yticks()[0], ax11.get_yticks()[-1], len(ax[1].get_yticks())))

    ax[2].set_yticks(linspace(ax[2].get_yticks()[0], ax[2].get_yticks()[-1], len(ax[2].get_yticks())))
    ax22.set_yticks(linspace(ax22.get_yticks()[0], ax22.get_yticks()[-1], len(ax[2].get_yticks())))

    for i in range(3):
        ax[i].legend(loc=1, ncols=4)
        ax[i].grid(which="both", alpha=0.5)
        ax[i].set_ylabel(f"$\Omega$ ({rot_unit})", fontsize=font)
        ax[i].text(0.05, 0.9, f"CC={cc[i]}", ha='left', va='top', transform=ax[i].transAxes, fontsize=font-1)

    for _ax in [ax00, ax11, ax22]:
        _ax.legend(loc=4)
        _ax.set_ylabel(f"$a$ ({acc_unit})", fontsize=font)

    ax[2].set_xlabel("Time (s)", fontsize=font)

    tbeg = acc[0].stats.starttime
    ax[0].set_title(f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC  |  f = {fmin}-{fmax} Hz  |  BAz = {round(baz, 1)}°  |  ED = {round(distance/1000,1)} km  |  T = {twin_sec}s ({int(100*twin_overlap)}%)")

    cax = ax[Nrow-1].inset_axes([0.8, -0.35, 0.2, 0.1], transform=ax[Nrow-1].transAxes)
    cb = plt.colorbar(cm, cax=cax, shrink=0.4, location='bottom', orientation='horizontal')
    cm.set_clim(-1, 1)
    cb.set_label("Cross-Correlation", fontsize=font, loc="left", labelpad=-43, color="black", backgroundcolor="w")

    plt.show();
    return fig