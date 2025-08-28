def __makeplotStream_velocities(rot, acc, fmin, fmax, baz, overlap=0.5, cc_threshold=0.2, vmax=None, twin=None, reverse_rotZ=False, reverse_accZ=False):

    import matplotlib.pyplot as plt
    import numpy as np
    from functions.compute_velocity_from_amplitude_ratio import __compute_velocity_from_amplitude_ratio
    from obspy.signal.rotate import rotate_ne_rt

    rot00 = rot.copy()
    acc00 = acc.copy()

    if reverse_rotZ:
        rot00.select(channel="*Z")[0].data *= -1

    if reverse_accZ:
        acc00.select(channel="*Z")[0].data *= -1

    rot00 = rot00.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True, corners=4)
    acc00 = acc00.detrend("demean").taper(0.1).filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True, corners=4)

    df = rot00[0].stats.sampling_rate

    if not twin:
        twin = 1/fmin

    out1 = __compute_velocity_from_amplitude_ratio(
                                                    rot00,
                                                    acc00,
                                                    baz=baz,
                                                    mode='love',
                                                    win_time_s=twin,
                                                    cc_thres=cc_threshold,
                                                    overlap=overlap,
                                                    flim=(fmin, fmax),
                                                    plot=False,
                                                    reverse_rotZ=False,
                                                    reverse_accZ=False,
                                                    )

    out2 = __compute_velocity_from_amplitude_ratio(
                                                    rot00,
                                                    acc00,
                                                    baz=baz,
                                                    mode='rayleigh',
                                                    win_time_s=twin,
                                                    cc_thres=cc_threshold,
                                                    overlap=overlap,
                                                    flim=(fmin, fmax),
                                                    plot=False,
                                                    reverse_rotZ=False,
                                                    reverse_accZ=False,
                                                    )

    rot_r, rot_t = rotate_ne_rt(rot00.select(channel='*N')[0].data,
                                rot00.select(channel='*E')[0].data,
                                baz
                                )
    acc_r, acc_t = rotate_ne_rt(acc00.select(channel='*N')[0].data,
                                acc00.select(channel='*E')[0].data,
                                baz
                                )

    rot_z = rot00.select(channel="*Z")[0].data
    rot_n = rot00.select(channel="*N")[0].data
    rot_e = rot00.select(channel="*E")[0].data

    acc_z = acc00.select(channel="*Z")[0].data
    acc_n = acc00.select(channel="*N")[0].data
    acc_e = acc00.select(channel="*E")[0].data

    Nrow, Ncol = 4, 1

    font = 12

    fig, ax = plt.subplots(Nrow, Ncol, figsize=(15, 7), sharex=True)

    plt.subplots_adjust(hspace=0.1)

    cmap = plt.get_cmap("viridis", 10)


    ## waveform axis
    if reverse_rotZ:
        ax[0].plot(np.array(range(len(rot_z)))/df, rot_z/max(abs(rot_z)), alpha=1, color="black", label="-1*$\Omega_Z$ (rad/s)")
    else:
        ax[0].plot(np.array(range(len(rot_z)))/df, rot_z/max(abs(rot_z)), alpha=1, color="black", label="$\Omega_Z$ (rad/s)")

    ax[0].plot(np.array(range(len(acc_t)))/df, acc_t/max(abs(acc_t)), alpha=1, color="tab:red", label=r"$a_T$ (m/s$^2$)")
    ax[2].plot(np.array(range(len(rot_t)))/df, rot_t/max(abs(rot_t)), alpha=1, color="black", label="$\Omega_T$ (rad/s)")

    if reverse_accZ:
        ax[2].plot(np.array(range(len(acc_z)))/df, acc_z/max(abs(acc_z)), alpha=1, color="tab:red", label=r"-1x $a_Z$ (m/s$^2$)")
    else:
        ax[2].plot(np.array(range(len(acc_z)))/df, acc_z/max(abs(acc_z)), alpha=1, color="tab:red", label=r"$a_Z$ (m/s$^2$)")


    ## velocity axis
    caa = ax[1].scatter(out1['time'], out1['velocity'], c=out1['ccoef'], s=60,
                        cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1,
                        label="phase velocity", zorder=2,
                        )

    ax[1].errorbar(out1['time'], out1['velocity'], xerr=out1['terr'], yerr=None,
                 zorder=1, color="black", alpha=0.4, marker='o', markersize=2, ls="None",
                )

    ax[1].set_ylabel(f"Love Phase \n Velocity (m/s)", fontsize=font)
    ax[1].set_ylim(bottom=0)
    if vmax is not None:
        ax[1].set_ylim(top=vmax)

    # ax[1].set_yticks(np.linspace(ax0.get_yticks()[0], ax0.get_yticks()[-1], len(ax[0].get_yticks())))
    ax[1].legend(loc=4, fontsize=font-2)

    # cbar = plt.colorbar(caa, pad=0.08)
    # cbar.set_label("CC-Coefficient", fontsize=font)
    # # cbar.set_clip_on(False)

    caa = ax[3].scatter(out2['time'], out2['velocity'], c=out2['ccoef'], s=60,
                        cmap=cmap, edgecolors="k", lw=1, vmin=0, vmax=1,
                        label="phase velocity", zorder=2,
                       )

    ax[3].errorbar(out2['time'], out2['velocity'], xerr=out2['terr'], yerr=None,
                 zorder=1, color="black", alpha=0.4, marker='o', markersize=2, ls="None",
                )

    ax[3].set_ylabel(f" Rayleigh Phase \n Velocity (m/s)", fontsize=font)
    ax[3].set_ylim(bottom=0)
    if vmax is not None:
        ax[3].set_ylim(top=vmax)

    # ax[3].set_yticks(np.linspace(ax[3].get_yticks()[0], ax[3].get_yticks()[-1], len(ax[1].get_yticks())))
    ax[3].legend(loc=4, fontsize=font-2)

    # cbar = plt.colorbar(caa, pad=0.08)
    # cbar.set_label("CC-Coefficient", fontsize=font)
    # # cbar.set_clip_on(False)

    # add colorbar
    cbar_ax = fig.add_axes([0.91, 0.11, 0.021, 0.77]) #[left, bottom, width, height]
    cb = plt.colorbar(caa, cax=cbar_ax)
    cb.set_label("CC-Coefficient", fontsize=font, labelpad=5, color="k")


    for _n in range(Nrow):
        # ax[_n].grid(ls=":", zorder=0)
        ax[_n].legend(loc=1, ncol=2)
        ax[_n].set_xlim(0, len(rot_z)/df)

    ax[0].set_ylim(-1.1, 1.1)
    ax[2].set_ylim(-1.1, 1.1)

    ax[0].set_ylabel(f"norm.\nAmplitude", fontsize=font)
    ax[2].set_ylabel(f"norm.\nAmplitude", fontsize=font)

    ax[3].set_xlabel("Time (s)", fontsize=font)

    title_str = f"f = {fmin}-{fmax} Hz | T = {twin} s | Overlap = {int(overlap*100)}% | CC > {cc_threshold}"
    ax[0].set_title(title_str, fontsize=font+1)

    for _k, ll in enumerate(['(a)', '(b)', '(c)', '(d)']):
        ax[_k].text(.005, .97, ll, ha='left', va='top', transform=ax[_k].transAxes, fontsize=font+2)

    for a in [ax[1], ax[3]]:
        a.minorticks_on()
        a.grid(ls=":", zorder=0, which="both")

    plt.show();
    return fig