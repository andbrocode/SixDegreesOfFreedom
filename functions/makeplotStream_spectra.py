def __makeplotStreamSpectra(st, fmin=None, fmax=None, fscale=None):

    from scipy import fftpack
    from andbro__fft import __fft
    import matplotlib.pyplot as plt

    NN = len(st)
    rot_scaling, rot_unit = 1e6, r"$\mu$rad/s"
    trans_scaling, trans_unit = 1e3, r"mm/s"

    fig, axes = plt.subplots(NN,2,figsize=(15,int(NN*2)), sharex='col')

    font = 14

    plt.subplots_adjust(hspace=0.3)

    colors = ["black", "tab:red", "tab:blue", "tab:green"]

    ## _______________________________________________

    st = st.sort(keys=['location','channel'], reverse=True)

    c = -1
    for i, tr in enumerate(st):

        if i %3 == 0:
            c+=1

#         comp_fft = abs(fftpack.fft(tr.data))
#         ff       = fftpack.fftfreq(comp_fft.size, d=1/tr.stats.sampling_rate)
#         comp_fft = fftpack.fftshift(comp_fft)
#         ff, spec = ff[1:len(ff)//2], abs(fftpack.fft(tr.data)[1:len(ff)//2])

        if tr.stats.channel[-2] == "J":
            scaling = rot_scaling
        elif tr.stats.channel[-2] == "H":
            scaling = trans_scaling

        spec, ff, ph = __fft(tr.data*scaling, tr.stats.delta, window="hanning", normalize=None)


        ## _________________________________________________________________
        if tr.stats.channel[-2] == "J":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*rot_scaling,
                        color=colors[c],
                        label='{}.{}.{}'.format(tr.stats.station, tr.stats.location, tr.stats.channel),
                        lw=1.0,
                        )
            axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color=colors[c], lw=0)


        elif tr.stats.channel[-2] == "H":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*trans_scaling,
                        color=colors[c],
                        label='{}.{}.{}'.format(tr.stats.station, tr.stats.location, tr.stats.channel),
                        lw=1.0,
                        )
            axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color=colors[c], lw=0)


        ## _________________________________________________________________
#         if fscale == "loglog":
# #             axes[i,1].loglog(ff, spec, color='black', lw=1.0)
#             axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color='black', lw=1.)

#         elif fscale == "loglin":
# #             axes[i,1].semilogx(ff, spec, color='black', lw=1.0)
#             axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color='black', lw=1.)

#         elif fscale == "linlog":
# #             axes[i,1].semilogy(ff, spec, color='black', lw=1.0)
#             axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color='black', lw=1.)

#         else:
# #             axes[i,1].plot(ff, spec, color='black', lw=1.0)
#             axes[i,1].fill_between(ff, 0, spec, alpha=0.7, color='black')


        if tr.stats.channel[1] == "J":
            sym, unit = r"$\Omega$", rot_unit
        elif tr.stats.channel[1] == "H":
            sym, unit = "v", trans_unit
        else:
            unit = "Amplitude", "a.u."

        axes[i,0].set_ylabel(f'{sym} ({unit})',fontsize=font)
        axes[i,1].set_ylabel(f'ASD \n({unit}/Hz)',fontsize=font)
        axes[i,0].legend(loc='upper left',bbox_to_anchor=(0.8, 1.10), framealpha=1.0)

        # axes[i,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        # axes[i,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axes[i,0].ticklabel_format(useOffset=False, style='plain')
        axes[i,1].ticklabel_format(useOffset=False, style='plain')

    if fmin is not None and fmax is not None:
        axes[i,1].set_xlim(fmin, fmax)

    axes[NN-1,0].set_xlabel(f"Time from {tr.stats.starttime.date} {str(tr.stats.starttime.time)[:8]} (s)",fontsize=font)
    axes[NN-1,1].set_xlabel(f"Frequency (Hz)",fontsize=font)

    plt.tight_layout()

    return fig