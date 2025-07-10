def __makeplot_spectra_comparison_fill(st_rot, st_acc, fmin=None, fmax=None, ylog=False, xlog=False, fill=False):

    import matplotlib.pyplot as plt

    def __multitaper_psd(arr, dt, n_win=5, time_bandwidth=4.0):

        import multitaper as mt

        out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=2)

        _f, _psd = out_psd.rspec()

        f = _f.reshape(_f.size)
        psd = _psd.reshape(_psd.size)

        ## 95% confidence interval
        # _psd95 = out_psd.jackspec()
        # psd95_lower, psd95_upper = psd95[::2, 0], psd95[::2, 1]

        return f, psd

    Tsec = 5
    f1_Z, psd1_Z = __multitaper_psd(st_rot.select(channel="*Z")[0].data, st_rot[0].stats.delta ,n_win=Tsec)
    f2_Z, psd2_Z = __multitaper_psd(st_acc.select(channel="*Z")[0].data, st_acc[0].stats.delta ,n_win=Tsec)

    f1_U, psd1_U = __multitaper_psd(st_rot.select(channel="*N")[0].data, st_rot[0].stats.delta ,n_win=Tsec)
    f2_N, psd2_N = __multitaper_psd(st_acc.select(channel="*N")[0].data, st_acc[0].stats.delta ,n_win=Tsec)

    f1_V, psd1_V = __multitaper_psd(st_rot.select(channel="*E")[0].data, st_rot[0].stats.delta ,n_win=Tsec)
    f2_E, psd2_E = __multitaper_psd(st_acc.select(channel="*E")[0].data, st_acc[0].stats.delta ,n_win=Tsec)


    Nrow, Ncol = 1, 3

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    plt.subplots_adjust(wspace=0.2)

    font = 14

    rot_scaling = 1e9

    lw = 1

    if fill:

        ax[0].fill_between(f1_Z, psd1_Z, lw=lw, label=f"{st_rot[0].stats.station}.{st_rot.select(channel='*Z')[0].stats.channel}", color="darkred", alpha=0.5, zorder=3)
        ax00 = ax[0].twinx()
        ax00.fill_between(f2_Z, psd2_Z, lw=lw, label=f"{st_acc[0].stats.station}.{st_acc.select(channel='*Z')[0].stats.channel}", color="black", alpha=0.5, zorder=2)

        ax[1].fill_between(f1_U, psd1_U, lw=lw, label=f"{st_rot[1].stats.station}.{st_rot.select(channel='*N')[0].stats.channel}", color="darkred", alpha=0.5, zorder=3)
        ax11 = ax[1].twinx()
        ax11.fill_between(f2_N, psd2_N, lw=lw, label=f"{st_acc[1].stats.station}.{st_acc.select(channel='*N')[0].stats.channel}", color="black", alpha=0.5, zorder=2)

        ax[2].fill_between(f1_V, psd1_V, lw=lw, label=f"{st_rot[2].stats.station}.{st_rot.select(channel='*E')[0].stats.channel}", color="darkred", alpha=0.5, zorder=3)
        ax22 = ax[2].twinx()
        ax22.fill_between(f2_E, psd2_E, lw=lw, label=f"{st_acc[2].stats.station}.{st_acc.select(channel='*E')[0].stats.channel}", color="black", alpha=0.5, zorder=2)

    else:
        ax[0].plot(f1_Z, psd1_Z, lw=lw, label=f"{st_rot[0].stats.station}.{st_rot.select(channel='*Z')[0].stats.channel}", color="darkred", ls="-", zorder=3)

        ax00 = ax[0].twinx()
        ax00.plot(f2_Z, psd2_Z, lw=lw, label=f"{st_acc[0].stats.station}.{st_acc.select(channel='*Z')[0].stats.channel}", color="black", zorder=2)

        ax[1].plot(f1_U, psd1_U, lw=lw, label=f"{st_rot[1].stats.station}.{st_rot.select(channel='*N')[0].stats.channel}", color="darkred", ls="-", zorder=3)
        ax11 = ax[1].twinx()
        ax11.plot(f2_N, psd2_N, lw=lw, label=f"{st_acc[1].stats.station}.{st_acc.select(channel='*N')[0].stats.channel}", color="black", zorder=2)

        ax[2].plot(f1_V, psd1_V, lw=lw, label=f"{st_rot[2].stats.station}.{st_rot.select(channel='*E')[0].stats.channel}", color="darkred", ls="-", zorder=3)
        ax22 = ax[2].twinx()
        ax22.plot(f2_E, psd2_E, lw=lw, label=f"{st_acc[2].stats.station}.{st_acc.select(channel='*E')[0].stats.channel}", color="black", zorder=2)


#     ax[0].set_ylim(0, max(psd1_Z)+0.1*max(psd1_Z))
#     ax[1].set_ylim(0, max(psd1_U)+0.1*max(psd1_U))
#     ax[2].set_ylim(0, max(psd1_V)+0.1*max(psd1_V))

#     ax00.set_yticks(np.linspace(ax00.get_yticks()[0], ax00.get_yticks()[-1], len(ax[0].get_yticks())))
#     ax11.set_yticks(np.linspace(ax11.get_yticks()[0], ax11.get_yticks()[-1], len(ax[1].get_yticks())))
#     ax22.set_yticks(np.linspace(ax22.get_yticks()[0], ax22.get_yticks()[-1], len(ax[2].get_yticks())))

    for i in range(3):
        ax[i].legend(loc=1, ncols=4)
        if xlog:
            ax[i].set_xscale("log")
        if ylog:
            ax[i].set_yscale("log")
        ax[i].grid(which="both", alpha=0.5)
        ax[i].tick_params(axis='y', colors='darkred')
        ax[i].set_xlabel("Frequency (Hz)")
        ax[i].set_ylim(bottom=0)

        if fmin:
            ax[i].set_xlim(left=fmin)
        if fmax:
            ax[i].set_xlim(right=fmax)
        else:
            ax[i].set_xlim(right=st_rot[0].stats.sampling_rate*0.5)

    for _ax in [ax00, ax11, ax22]:
        _ax.set_xlim(2e-2, 2e1)
        _ax.legend(loc=2)
        _ax.set_ylim(bottom=0)

        if ylog:
            _ax.set_yscale("log")
        if fmin:
            _ax.set_xlim(left=fmin)
        if fmax:
            _ax.set_xlim(right=fmax)
        else:
            _ax.set_xlim(right=st_rot[0].stats.sampling_rate*0.5)

    ax[0].set_ylabel(r"PSD (rad$^2$/s$^2$/Hz)", color="darkred")
    ax22.set_ylabel(r"PSD (m$^2$/s$^4$/Hz)")

    # ax[2].set_xlabel("Frequency (Hz)")
    # ax[0].set_title(f"{config['tbeg'].date} {str(config['tbeg'].time).split('.')[0]} UTC | {config['fmin']}-{config['fmax']} Hz ")

    plt.show();
    return fig