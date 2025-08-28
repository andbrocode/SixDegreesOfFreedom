def __compute_beamforming_pfo(tbeg, tend, submask, fmin=None, fmax=None, component="", bandpass=True, plot=False):

    import os
    import numpy as np
    import timeit
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import scipy.stats as sts

    from obspy import UTCDateTime, Stream
    from obspy.clients import fdsn
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.geodetics import locations2degrees
    from obspy.clients.fdsn import Client, RoutingClient
    from obspy.signal import array_analysis as AA
    from obspy.signal.util import util_geo_km
    from obspy.signal.rotate import rotate2zne
    from obspy.core.util import AttribDict
    from obspy.imaging.cm import obspy_sequential
    from obspy.signal.invsim import corn_freq_2_paz
    from obspy.signal.array_analysis import array_processing    
    from datetime import datetime
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    import warnings
    warnings.filterwarnings('ignore')

    ## _____________________________________________________

    def __get_data(config):

        config['subarray'] = []

        st = Stream()

        for k, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            if net == "II" and sta == "PFO":
                loc, cha = "10", "BH*"
            elif net == "PY" and sta == "PFOIX":
                loc, cha = "", "HH*"
            else:
                loc, cha = "", "BH*"

            # print(f" -> requesting {net}.{sta}.{loc}.{cha}")


            ## querry inventory data
            # try:
            inventory = config['fdsn_client'].get_stations(
                                                             network=net,
                                                             station=sta,
                                                             # channel=cha,
                                                             starttime=config['tbeg']-20,
                                                             endtime=config['tend']+20,
                                                             level="response"
                                                            )
            # except:
            #     print(f" -> {station}: Failed to load inventory!")
            #     inventory = None

            ## try to get waveform data
            try:
                stats = config['fdsn_client'].get_waveforms(
                                                            network=net,
                                                            station=sta,
                                                            location=loc,
                                                            channel=cha,
                                                            starttime=config['tbeg']-20,
                                                            endtime=config['tend']+20,
                                                            attach_response=True
                                                            )
            except Exception as E:
                print(E) if config['print_details'] else None
                print(f" -> geting waveforms failed for {net}.{sta}.{loc}.{cha} ...") if config['print_details'] else None
                continue


            ## merge if masked
            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3") if config['print_details'] else None
                stats.merge(method=1, fill_value="interpolate")


            ## sorting
            # stats.sort().reverse()


            ## remove response [ACC -> m/s/s | VEL -> m/s | DISP -> m]
            stats.remove_response(inventory=inventory, output="VEL")


            ## rotate to ZNE
            try:
                stats.rotate(method="->ZNE", inventory=inventory)
            except:
                print(" -> failed to rotate to ZNE")
                continue


            #correct mis-alignment
            # stats[0].data, stats[1].data, stats[2].data = rotate2zne(stats[0], 0, -90,
            #                                                          stats[1],config['subarray_misorientation'][config['subarray_stations'].index(station)],0,
            #                                                          stats[2],90+config['subarray_misorientation'][config['subarray_stations'].index(station)],0)


            ## trim to interval
            # stats.trim(config['tbeg'], config['tend'], nearest_sample=False)



            ## rename channels
            # if net == "II" and sta == "PFO":
            #     for tr in stats:
            #         if tr.stats.channel[-1] == "1":
            #             tr.stats.channel = str(tr.stats.channel).replace("1","E")
            #         if tr.stats.channel[-1] == "2":
            #             tr.stats.channel = str(tr.stats.channel).replace("2","N")

            if config['reference_station'] == "PY.PFOIX":
                stats = stats.resample(40)
                stats = stats.trim(config['tbeg']-20, config['tend']+20)


            if station == config['reference_station']:
                ref_station = stats.copy()

            st += stats


        print(st.__str__(extended=True)) if config['print_details'] else None

        ## update subarray stations if data could not be requested for all stations
        if len(st) < 3*len(config['subarray_stations']):
            config['subarray_stations'] = [f"{tr.stats.network}.{tr.stats.station}" for tr in st]
            config['subarray_stations'] = list(set(config['subarray_stations']))

        print(f" -> obtained: {int(len(st)/3)} of {len(config['subarray_stations'])} stations!")

        if len(st) == 0:
            return st, config
        else:
            return st, config


    def __add_coordinates(st, config):

        coo = []
        for i, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            if net == "II" and sta == "PFO":
                loc, cha = "10", "BH*"
            elif net == "PY" and sta == "PFOIX":
                loc, cha = "", "HH*"
            else:
                loc, cha = "", "BH*"

            try:
                inven = config['fdsn_client'].get_stations(network=net,
                                                           station=sta,
                                                           channel=cha,
                                                           starttime=config['tbeg'],
                                                           endtime=config['tend'],
                                                           level='response'
                                                          )
            except:
                print(f" -> cannot get inventory for {station}")

            l_lon =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])

            ## set coordinates of seismometer manually, since STATIONXML is wrong...
            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                l_lon, l_lat =  -116.455439, 33.610643


            for c in ["Z", "N", "E"]:
                st.select(station=sta, channel=f"*{c}")[0].stats.coordinates = AttribDict({
                                                                                          'latitude': l_lat,
                                                                                          'elevation': height/1000,
                                                                                          'longitude': l_lon
                                                                                           })

        return st

    ## _____________________________________________________

    ## start timer for runtime
    start_timer = timeit.default_timer()


    ## _____________________________________________________

    ## generate configuration object
    config = {}

    ## time period of event
    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    ## select the fdsn client for the stations
    config['fdsn_client'] = Client('IRIS')


    ## select stations to consider:
    ## all: [0,1,2,3,4,5,6,7,8,9,10,11,12] | optimal: [0,5,8,9,10,11,12] | inner: [0,1,2,3]
    if submask is not None:
        if submask == "inner":
            config['subarray_mask'] = [0,1,2,3,4]
            config['freq1'] = 1.0  ## 0.16  ## 0.00238*3700/100
            config['freq2'] = 6.0  ## 16.5 ## 0.25*3700/100
        elif submask == "mid":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8]
            config['freq1'] = 0.5 ## 0.03   ## 0.00238*3700/280
            config['freq2'] = 1.0 ## # 0.25*3700/280
        elif submask == "all":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            config['freq1'] = 0.1 ## 0.02   ## 0.00238*3700/700
            config['freq2'] = 0.5 ## 1.3 # 0.25*3700/700
    else:
        config['subarray_mask'] = [0,1,2,3,4]


    ## decide if information is printed while running the code
    config['print_details'] = False

    ## apply bandpass to data
    config['apply_bandpass'] = True


    ## _____________________
    ## PFO array information

    if config['tbeg'] > UTCDateTime("2023-04-01"):
        config['reference_station'] = 'PY.PFOIX' ## 'BPH01'  ## reference station

        config['array_stations'] = ['PY.PFOIX','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                    'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']
    else:
        config['reference_station'] = 'II.PFO' ## 'BPH01'  ## reference station

        config['array_stations'] = ['II.PFO','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                    'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']


    config['misorientations'] =  [0, 0. ,-1.375 ,0.25 ,0.125 ,-0.6875 ,-0.625 ,-1.9375 ,0.375
                                  ,-6.5625 ,0.3125 ,-1.125 ,-2.5625 ,0.1875]


    config['subarray_misorientation'] = [config['misorientations'][i] for i in config['subarray_mask']]
    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]

    ## ______________________________

    ## beamforming parameters
    config['slow_xmin'] = -0.5
    config['slow_xmax'] = 0.5
    config['slow_ymin'] = -0.5
    config['slow_ymax'] = 0.5
    config['slow_steps'] = 0.01

    config['win_length'] = 1/fmin # window length in seconds
    config['win_frac'] = 0.5  # fraction of window to use as steps

    config['freq_lower'] = fmin
    config['freq_upper'] = fmax
    config['prewhitening'] = 0  ## 0 or 1


    ## loading data
    st, config = __get_data(config)

    ## pre-pprocessing data
    st = st.detrend("demean")

    if config['apply_bandpass']:
        st = st.taper(0.1)
        st = st.filter("bandpass", freqmin=config['freq_lower'], freqmax=config['freq_upper'], corners=8, zerophase=True)

    ## add coordinates from inventories
    st = __add_coordinates(st, config)

    ## select only one component
    st = st.select(channel=f"*{component}")

    st = st.trim(config['tbeg']-0.1, config['tend']+0.1)

    ## define parameters for beamforming
    kwargs = dict(

        # slowness grid: X min, X max, Y min, Y max, Slow Step
        sll_x=config['slow_xmin'], slm_x=config['slow_xmax'],
        sll_y=config['slow_ymin'], slm_y=config['slow_ymax'],
        sl_s=config['slow_steps'],

        # sliding window properties
        win_len=config['win_length'], win_frac=config['win_frac'],

        # frequency properties
        frqlow=config['freq_lower'], frqhigh=config['freq_upper'], prewhiten=config['prewhitening'],

        # restrict output
        semb_thres=-1e9, vel_thres=-1e9, timestamp='mlabday',

        ## time period
        stime=config['tbeg'], etime=config['tend'],
        # stime=st[0].stats.starttime, etime=st[0].stats.endtime,
    )

    ## perform beamforming
    out = array_processing(st, **kwargs)

    st = st.trim(config['tbeg'], config['tend'])


    ## stop times
    stop_timer = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer - start_timer)/60,2)} minutes")

    ## ______________________________
    ## Plotting

    if plot:

        ## PLOT 1 -----------------------------------
        labels = ['rel.power', 'abs.power', 'baz', 'slow']

        out[:, 3][out[:, 3] < 0.0] += 360

        xlocator = mdates.AutoDateLocator()

        fig1, ax = plt.subplots(5,1, figsize=(15,10))

        Tsec = config['tend']-config['tbeg']
        times = (out[:, 0]-out[:, 0][0]) / max(out[:, 0]-out[:, 0][0]) * Tsec

        for i, lab in enumerate(labels):
            # ax[i].scatter(out[:, 0], out[:, i + 1], c=out[:, 1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
            # ax[i].scatter(times, out[:, i + 1], c=out[:, 1], alpha=0.6, edgecolors='none', cmap=obspy_sequential)
            ax[i].scatter(times, out[:, i + 1], c=out[:, 2], alpha=0.6, edgecolors='k', cmap=obspy_sequential)
            ax[i].set_ylabel(lab)
            # ax[i].set_xlim(out[0, 0], out[-1, 0])
            ax[i].set_ylim(out[:, i + 1].min(), out[:, i + 1].max())
            ax[i].xaxis.set_major_locator(xlocator)
            ax[i].xaxis.set_major_formatter(mdates.AutoDateFormatter(xlocator))

        ax[4].plot(st[0].times()/st[0].times()[-1]*out[:, 0][-1], st[0].data)
        ax[2].set_ylim(0, 360)

        fig1.autofmt_xdate()

        plt.show();



    ## PLOT 2 -----------------------------------
    cmap = obspy_sequential

    # make output human readable, adjust backazimuth to values between 0 and 360
    t, rel_power, abs_power, baz, slow = out.T
    baz[baz < 0.0] += 360

    # choose number of fractions in plot (desirably 360 degree/N is an integer!)
    N = 36
    N2 = 30
    abins = np.arange(N + 1) * 360. / N
    sbins = np.linspace(0, 3, N2 + 1)


    # sum rel power in bins given by abins and sbins
    # hist2d, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=rel_power)
    hist2d, baz_edges, sl_edges = np.histogram2d(baz, slow, bins=[abins, sbins], weights=abs_power)

    # transform to radian
    baz_edges = np.radians(baz_edges)

    if plot:

        # add polar and colorbar axes
        fig2 = plt.figure(figsize=(8, 8))

        cax = fig2.add_axes([0.85, 0.2, 0.05, 0.5])
        ax = fig2.add_axes([0.10, 0.1, 0.70, 0.7], polar=True)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")

        dh = abs(sl_edges[1] - sl_edges[0])
        dw = abs(baz_edges[1] - baz_edges[0])

        # circle through backazimuth
        for i, row in enumerate(hist2d):
            bars = ax.bar((i * dw) * np.ones(N2),
                          height=dh * np.ones(N2),
                          width=dw, bottom=dh * np.arange(N2),
                          color=cmap(row / hist2d.max()))

        ax.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        ax.set_xticklabels(['N', 'E', 'S', 'W'])

        # set slowness limits
        ax.set_ylim(0, config['slow_xmax'])
        [i.set_color('grey') for i in ax.get_yticklabels()]
        ColorbarBase(cax, cmap=cmap, norm=Normalize(vmin=hist2d.min(), vmax=hist2d.max()))

        plt.show();

    max_val = 0
    for i in range(hist2d.shape[0]):
        for j in range(hist2d.shape[1]):
            if hist2d[i,j] > max_val:
                max_val, slw_max, baz_max = hist2d[i,j], sbins[j], abins[i]

    ## prepare output
    baz = out[:, 3]
    baz[baz < 0.0] += 360

    ## compute statistics
    deltaa = 5
    angles = np.arange(0, 365, deltaa)

    baz_bf_no_nan = baz[~np.isnan(baz)]
    cc_bf_no_nan = out[:, 2][~np.isnan(out[:, 2])]

    hist = np.histogram(baz, bins=len(angles)-1, range=[min(angles), max(angles)], weights=out[:, 2], density=False)

    baz_bf_mean = round(np.average(baz_bf_no_nan, weights=cc_bf_no_nan), 0)
    baz_bf_std = np.sqrt(np.cov(baz_bf_no_nan, aweights=cc_bf_no_nan))

    kde = sts.gaussian_kde(baz_bf_no_nan, weights=cc_bf_no_nan)
    baz_bf_max = angles[np.argmax(kde.pdf(angles))] + deltaa/2


    ## prepare output dictionary
    output = {}
    output['t_win'] = out[:, 0]
    output['rel_pwr'] = out[:, 1]
    output['abs_pwr'] = out[:, 2]
    output['baz'] = baz
    output['slow'] = out[:, 4]
    output['baz_max_count'] = max_val
    output['baz_max'] = baz_max
    output['slw_max'] = slw_max
    output['baz_bf_mean'] = baz_bf_mean
    output['baz_bf_max'] = baz_bf_max
    output['baz_bf_std'] = baz_bf_std


    if plot:
        output['fig1'] = fig1
        output['fig2'] = fig2

    return output