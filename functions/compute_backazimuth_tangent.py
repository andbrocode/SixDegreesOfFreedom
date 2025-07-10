def __compute_backazimuth_tangent(rot0, acc0, win_time_s=0.5, overlap=0.5, baz_theo=None, cc_thres=None, plot=False, invert_acc_z=False):

    from numpy import zeros, nan, ones, nanmean, array, nanmax
    from numpy import arctan, pi, linspace, cov, argsort, corrcoef, correlate
    from obspy.signal.rotate import rotate_ne_rt
    from numpy.linalg import eigh
    from obspy.signal.cross_correlation import correlate
    import matplotlib.pyplot as plt

    npts = rot0[0].stats.npts

    df = rot0[0].stats.sampling_rate

    ## windows
    t_win = win_time_s
    n_win = int(win_time_s*df)
    nover = int(overlap*n_win)

    ## extract components
    rot_n = rot0.select(channel="*N")[0].data
    rot_e = rot0.select(channel="*E")[0].data

    if invert_acc_z:
        acc_z = -acc0.select(channel="*Z")[0].data
    else:
        acc_z = acc0.select(channel="*Z")[0].data

    ## define windows
    n, windows = 0, []
    while n < npts-n_win:
        windows.append((n, n+n_win))
        n += n_win

    ## add overlap
    if overlap != 0:
        windows_overlap = []
        for i, w in enumerate(windows):
            if i == 0:
                windows_overlap.append((w[0], w[1]+nover))
            elif i >= int(len(windows)-nover):
                windows_overlap.append((w[0]-nover, w[1]))
            else:
                windows_overlap.append((w[0]-nover, w[1]+nover))
    else:
        windows_overlap = windows

    ## compute baz and ccorr for each window
    baz, ccor = ones(len(windows_overlap))*nan, ones(len(windows_overlap))*nan

    for j, (w1, w2) in enumerate(windows_overlap):

        if int(w2-w1) < 10:
            print(f" -> not enough samples in window (<10)")

        dat = (zeros((len(rot_n[w1:w2]), 2)))
        dat[:, 0] = rot_e[w1: w2]
        dat[:, 1] = rot_n[w1: w2]

        covar = cov(dat, rowvar=False)

        Cprime, Q = eigh(covar, UPLO='U')

        loc = argsort(abs(Cprime))[::-1]

        Q = Q[:, loc]

        baz0 = -arctan((Q[1, 0]/Q[0, 0]))*180/pi

        ## make sure baz is between 0 - 360
        if baz0 <= 0:
            baz0 += 180

        ## __________________________
        ## remove 180° ambiguity

        rot_r, rot_t = rotate_ne_rt(rot_n[w1:w2], rot_e[w1:w2], baz0)

        # corr_baz = corrcoef(acc_z[w1:w2], rot_t)[0][1]
        corr_baz = correlate(acc_z[w1:w2], rot_t, 0, 'auto')[0]

#         if (corr_baz > 0): ## original
#             baz0 += 180

        if (corr_baz < 0): ## original
            baz0 += 180

        ## add new values to array
        if abs(corr_baz) > cc_thres:
            baz[j] = baz0
            ccor[j] = abs(corr_baz)

    ## define time axis
    t1 = array([w1/df for (w1, w2) in windows_overlap])
    t2 = array([w2/df for (w1, w2) in windows_overlap])

    time = array([((w2-w1)/2+w1)/df for (w1, w2) in windows_overlap])
    terr = (t2-t1)/2

    win_center = array([(((w2-w1)/2)+w1) for (w1, w2) in windows_overlap])
    t_win_center = win_center/df

    ## Plotting
    if plot:

        rot0_r, rot0_t = rotate_ne_rt(rot_n, rot_e, baz_theo)

        cmap = plt.get_cmap("viridis", 10)

        fig, ax = plt.subplots(1, 1, figsize=(15,5))

        ax.plot(array(range(len(rot0_t)))/df, rot0_t/max(abs(rot0_t)), alpha=1, color="grey", label="rotation rate T (rad/s)")
        ax.plot(array(range(len(acc_z)))/df, acc_z/max(abs(acc_z)), alpha=0.5, color="tab:red", label=r"acceleration Z (m/s$^2$)")

        ax.set_ylim(-1, 1)
        # ax.set_xlim(0, len(rot0_t)/df)
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Norm. Amplitude", fontsize=14)
        ax.grid(zorder=0)
        ax.legend(loc=4, fontsize=13)

        ax2 = ax.twinx()
        cax = ax2.scatter(time, baz, c=ccor, s=50, cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        ax2.errorbar(time, baz, xerr=terr, yerr=None, zorder=1, color="lightgrey", marker='o', markersize=2, ls="None")
        ax2.set_ylabel("Backazimuth (°)", fontsize=14)
        ax2.set_ylim(0, 360)
        ax2.set_yticks(linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks())))

        cbar = plt.colorbar(cax, pad=0.08)
        cbar.set_label("Cross-Correlation Coefficient", fontsize=14)

        cax.set_clip_on(False)

        if baz_theo:
            ax2.axhline(baz_theo, ls=":", c="k")

        plt.show();

        out = {"time":time, "baz_est":baz, "ccoef":ccor, "baz_theo":baz_theo, "t_win_center":t_win_center, "fig":fig}
    else:
        out = {"time":time, "baz_est":baz, "ccoef":ccor, "baz_theo":baz_theo, "t_win_center":t_win_center}

    return out