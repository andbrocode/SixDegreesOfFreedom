def __makeplot_cwt_all(st0, out, clog=False, ylim=None):

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    def __rmse(arr1, arr2):
        return np.round(np.sqrt(np.mean((arr1-arr2)**2)), 2)

    tscale = 1

    times = st0.select(channel="*N")[0].times()*tscale

    tbeg = st0.select(channel="*N")[0].stats.starttime

    Ncol, Nrow = 1, 9

    font = 12

    cmap = plt.get_cmap("viridis")

    tilt_scale, tilt_unit = 1e6, f"$\mu$rad"


    fig = plt.figure(figsize=(15, 8))

    gs = GridSpec(Nrow, Ncol, figure=fig, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1:3, :])
    ax3 = fig.add_subplot(gs[3, :])
    ax4 = fig.add_subplot(gs[4:6, :])
    ax5 = fig.add_subplot(gs[6, :])
    ax6 = fig.add_subplot(gs[7:9, :])

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    Z = st0.select(channel="*Z")[0].data*tilt_scale
    N = st0.select(channel="*N")[0].data*tilt_scale
    E = st0.select(channel="*E")[0].data*tilt_scale

    periodZ = out['Z']['frequencies']
    periodN = out['N']['frequencies']
    periodE = out['N']['frequencies']

    # _______________________________________________________________
    #
    ax1.plot(times, N, color="k", label="N", lw=1)
    ax1.set_xlim(min(times), max(times))
    ax1.legend(loc=1)
    ax1.set_xticklabels([])

    if clog:
        vmin, vmax, norm = 0.01, 1, "log"
    else:
        vmin, vmax, norm = 0.0, 0.9, None
    # _______________________________________________________________
    #
    im1 = ax2.pcolormesh(out['N']['times']*tscale, periodN, out['N']['cwt_power'],
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         norm=norm,
                        )
    ax2.plot(out['N']['times']*tscale, out['N']['cone'], color="white", ls="--")
    ax2.fill_between(out['N']['times']*tscale, out['N']['cone'], min(periodN)*np.ones(len(out['N']['cone'])), color="white", alpha=0.2)
    ax2.set_xticklabels([])

    if ylim is None:
        ax2.set_ylim(min(periodN), max(periodN))
    else:
        ax2.set_ylim(min(periodN), ylim)

    # _______________________________________________________________
    #
    ax3.plot(times, E, color="k", label="E", lw=1)
    ax3.set_xlim(min(times), max(times))
    ax3.legend(loc=1)
    ax3.set_xticklabels([])

    # _______________________________________________________________
    #
    im2 = ax4.pcolormesh(out['E']['times']*tscale, periodE, out['E']['cwt_power'],
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         norm=norm,
                        )
    ax4.plot(out['E']['times']*tscale, out['E']['cone'], color="white", ls="--")
    ax4.fill_between(out['E']['times']*tscale, out['E']['cone'], min(periodE)*np.ones(len(out['E']['cone'])), color="white", alpha=0.2)
    ax4.set_xticklabels([])

    if ylim is None:
        ax4.set_ylim(min(periodE), max(periodE))
    else:
        ax4.set_ylim(min(periodE), ylim)

    # _______________________________________________________________
    #
    ax5.plot(times, Z, color="k", label="Z", lw=1)
    ax5.set_xlim(min(times), max(times))
    ax5.legend(loc=1)
    ax5.set_xticklabels([])

    # _______________________________________________________________
    #
    im3 = ax6.pcolormesh(out['Z']['times']*tscale, periodE, out['Z']['cwt_power'],
                         cmap=cmap,
                         vmin=vmin,
                         vmax=vmax,
                         norm=norm,
                        )
    ax6.plot(out['Z']['times']*tscale, out['Z']['cone'], color="white", ls="--")
    ax6.fill_between(out['Z']['times']*tscale, out['Z']['cone'], min(periodZ)*np.ones(len(out['Z']['cone'])), color="white", alpha=0.2)

    if ylim is None:
        ax6.set_ylim(min(periodZ), max(periodZ))
    else:
        ax6.set_ylim(min(periodZ), ylim)


    for a in [ax1, ax3, ax5]:
        _yl = max(a.get_ylim())
        a.set_ylim(-_yl, _yl)

    # _______________________________________________________________
    #
    ax1.set_ylabel(f"$\Omega$ ({tilt_unit})", fontsize=font)
    ax2.set_ylabel(f"Frequency\n(Hz)", fontsize=font)
    ax3.set_ylabel(f"$\Omega$ ({tilt_unit})", fontsize=font)
    ax4.set_ylabel(f"Frequency\n(Hz)", fontsize=font)
    ax5.set_ylabel(f"$\Omega$ ({tilt_unit})", fontsize=font)
    ax6.set_ylabel(f"Frequency\n(Hz)", fontsize=font)


    ax6.set_xlabel(f"Time (s) from {tbeg.date} {str(tbeg.time).split('.')[0]} UTC", fontsize=font)

    ax1.text(.005, .97, "(a)", ha='left', va='top', transform=ax1.transAxes, fontsize=font+2)
    ax2.text(.005, .97, "(b)", ha='left', va='top', transform=ax2.transAxes, fontsize=font+2, color="w")
    ax3.text(.005, .97, "(c)", ha='left', va='top', transform=ax3.transAxes, fontsize=font+2)
    ax4.text(.005, .97, "(d)", ha='left', va='top', transform=ax4.transAxes, fontsize=font+2, color="w")
    ax5.text(.005, .97, "(e)", ha='left', va='top', transform=ax5.transAxes, fontsize=font+2)
    ax6.text(.005, .97, "(f)", ha='left', va='top', transform=ax6.transAxes, fontsize=font+2, color="w")

    # add colorbar
    cbar_ax = fig.add_axes([0.92, 0.11, 0.01, 0.77]) #[left, bottom, width, height]
    cb = plt.colorbar(im1, cax=cbar_ax, extend="max")
    cb.set_label("norm. continuous wavelet transform power", fontsize=font, labelpad=-55, color="black")

#     ax1.set_title(f"bandpass = {pmax} - {pmin} hours")

    plt.plot();
    return fig