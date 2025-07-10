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
        shift, val = xcorr_max(ccf)
        cc.append(val)


    return array(times), array(cc)