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