def __get_maximum_frequency(st):

    from numpy import argmax

    def __multitaper_psd(arr, dt, n_win=5, time_bandwidth=4.0):

        import multitaper as mt

        out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=2)

        _f, _psd = out_psd.rspec()

        f = _f.reshape(_f.size)
        psd = _psd.reshape(_psd.size)

        return f, psd

    fmax = {}
    for tr in st:
        _f, _psd = __multitaper_psd(tr.data, tr.stats.delta)
        fmax[tr.stats.channel] = round(_f[argmax(_psd)], 3)

    return fmax
