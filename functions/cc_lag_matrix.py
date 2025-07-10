def __cc_lag_matrix(dat1, dat2, dt, twin_sec, T_lag_sec, dT_lag_sec, plot=False):

    import numpy as np
    import matplotlib.pyplot as plt
    from functions.cross_correlation_windows import __cross_correlation_windows

    L, dL = int(T_lag_sec/dt), int(dT_lag_sec/dt)

    Tlags = np.arange(-T_lag_sec, T_lag_sec+dT_lag_sec, dT_lag_sec)
    Nlags = np.arange(-L, L+dL, dL)

    for _k, lag in enumerate(Nlags):

        tt, cc = __cross_correlation_windows(dat1, dat2, dt, twin_sec, overlap=0.5, lag=lag, demean=True)

        if _k == 0:
            ccc = np.zeros((len(Nlags), len(cc)))

        ccc[_k] = cc

    if plot:
        plt.figure(figsize=(15, 5))

        cmap = plt.get_cmap("seismic")

        plt.pcolormesh(tt/86400, Tlags, ccc, vmin=-1, vmax=1, cmap=cmap)
        plt.ylabel("Lag Time (sec)")
        plt.xlabel("Time (days)")

        plt.show();

    return tt, Tlags, ccc