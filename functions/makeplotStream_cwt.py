#!/bin/python3

def __makeplotStream_cwt(st, config, fscale=None):

    from scipy import fftpack
    from andbro__fft import __fft
    from numpy import array, log10, logspace, linspace, meshgrid, abs
    from obspy.signal.tf_misfit import cwt
    from obspy.imaging.cm import obspy_sequential
    
    import matplotlib.pyplot as plt

    keys_expected = ['fmin', 'fmax']
    for key in keys_expected:
        if key not in config.keys():
            print(" -> keys missing in config")
            print(keys_expected)
            return
    
    
    NN = len(st)
    rot_scaling, rot_unit = 1e6, r"$\mu$rad/s"
    trans_scaling, trans_unit = 1e3, r"mm/s"
        
    fig, axes = plt.subplots(NN,2,figsize=(15,int(NN*2)), sharex='col')

    font = 14
    
    plt.subplots_adjust(hspace=0.3)

    ## _______________________________________________

    st.sort(keys=['channel'], reverse=True)
    

    for i, tr in enumerate(st):

        if tr.stats.channel[-2] == "J":
            scaling = rot_scaling
        elif tr.stats.channel[-2] == "H":
            scaling = trans_scaling
            
#         t = tr.times()
        t = linspace(0, tr.stats.delta * tr.stats.npts, tr.stats.npts)
        
#         tr = tr.normalize()
        
        scalogram = cwt(tr.data, tr.stats.delta, 8, config['fmin'], config['fmax'])

        x, y = meshgrid(t, logspace(log10(config['fmin']), log10(config['fmax']), scalogram.shape[0]))
         
    
        ## _________________________________________________________________
        if tr.stats.channel[-2] == "J":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*rot_scaling,
                        color='black',
                        label='{} {}'.format(tr.stats.station, tr.stats.channel),
                        lw=1.0,
                        )
            axes[i,1].pcolormesh(x, y, abs(scalogram), cmap=obspy_sequential)    


        elif tr.stats.channel[-2] == "H":
            axes[i,0].plot(
                        tr.times(),
                        tr.data*trans_scaling,
                        color='darkblue',
                        label='{} {}'.format(tr.stats.station, tr.stats.channel),
                        lw=1.0,
                        )
            axes[i,1].pcolormesh(x, y, abs(scalogram), cmap=obspy_sequential)    
        
        axes[i,1].set_yscale("log")
        
        if tr.stats.channel[1] == "J":
            sym, unit = r"rot.", rot_unit
        elif tr.stats.channel[1] == "H":
            sym, unit = "acc.", trans_unit
        else:
            unit = "Amplitude", "a.u."
            
        axes[i,0].set_ylabel(f'{sym} ({unit})',fontsize=font)    
        axes[i,1].set_ylabel(f'Freq. (Hz)',fontsize=font)        
        axes[i,0].legend(loc='upper left',bbox_to_anchor=(0.8, 1.10), framealpha=1.0)
        
#         axes[i,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
#         axes[i,1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))


    axes[NN-1,0].set_xlabel(f"Time from {tr.stats.starttime.date} {str(tr.stats.starttime.time)[:8]} (s)",fontsize=font)     
    axes[NN-1,1].set_xlabel(f"Time from {tr.stats.starttime.date} {str(tr.stats.starttime.time)[:8]} (s)",fontsize=font)     

    return fig

## End of File