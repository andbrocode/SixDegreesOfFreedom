import pandas as pd
import numpy as np
import pickle
from obspy.geodetics import locations2degrees, degrees2kilometers, gps2dist_azimuth
from typing import Dict, List, Tuple, Union, Optional, Any
from obspy import Stream

def get_default_config():
    """
    Returns default configuration settings for BSPF catalog analysis.
    
    Returns:
        dict: Dictionary containing default configuration parameters
    """
    config = {
        # Geographic bounds
        'minlatitude': 31,
        'maxlatitude': 35,
        'minlongitude': -119, 
        'maxlongitude': -114,
        
        # Station coordinates
        'BSPF_lon': [-116.455439],
        'BSPF_lat': [33.610643],
        
        # Magnitude threshold
        'minmagnitude': None,
        
        # Time bounds (will be set later)
        'tbeg': None,
        'tend': None,
        
        # File names
        'eventfile': None,  # Will be set based on dates
        'triggerfile': None,  # Will be set based on dates
        'gcmt_file': None,  # Will be set based on dates
        
        # Paths (will be populated based on environment)
        'path_to_data': None,
        'path_to_catalogs': None,
        'out_figures': None,
        'outpath': None
    }
    
    return config

def read_pickle(path, filename):
    """
    Read a pickle file from the specified path.
    
    Args:
        path (str): Path to the directory containing the pickle file
        filename (str): Name of the pickle file
        
    Returns:
        object: Contents of the pickle file
    """
    with open(path + filename, 'rb') as f:
        data = pickle.load(f)
    return data

def request_data(seed, tbeg, tend, bulk_download=True, translation_type="ACC"):

    from obspy.clients.fdsn import Client

    client = Client("IRIS")

    net, sta, loc, cha = seed.split(".")

    # querry inventory data
    try:
        inventory = client.get_stations(network=net,
                                        station=sta,
                                        location=loc,
                                        starttime=tbeg-60,
                                        endtime=tend+60,
                                        level="response",
                                        )
    except:
        print(" -> Failed to load inventory!")
        inventory = None

    # querry waveform data
    try:

        if bulk_download:
            bulk = [(net, sta, loc, cha, tbeg-60, tend+60)]
            waveform = client.get_waveforms_bulk(bulk, attach_response=True)
        else:
            waveform = client.get_waveforms(network=net,
                                           station=sta,
                                           location=loc,
                                           channel=cha, 
                                           starttime=tbeg-60,
                                           endtime=tend+60,
                                           attach_response=False
                                           )
    except:
        print(" -> Failed to load waveforms!")
        waveform = None

    # adjust channel names
    if cha[1] == "J" and waveform is not None:

        waveform.remove_sensitivity(inventory=inventory)
        print(" -> sensitivity removed!")

    # adjust channel names
    elif cha[1] == "H" and waveform is not None:
        waveform.remove_response(inventory=inventory, output=translation_type, plot=False)
        print(" -> response removed!")

    try:
        if waveform is not None:
            waveform.rotate(method="->ZNE", inventory=inventory)
            print(" -> rotated to ZNE")
    except:
        print(" -> failed to rotate to ZNE")

    if waveform is not None:
        waveform = waveform.trim(tbeg, tend)

    return waveform, inventory

def catalog_to_dataframe(catalog):
    """
    Convert ObsPy catalog to pandas DataFrame.
    
    Args:
        catalog: ObsPy catalog object
        
    Returns:
        pandas.DataFrame: DataFrame containing event information
    """
    data = []
    for event in catalog:
        try:
            origin = event.preferred_origin() or event.origins[0]
            magnitude = event.preferred_magnitude() or event.magnitudes[0]
            
            data.append({
                'timestamp': origin.time,
                'latitude': origin.latitude,
                'longitude': origin.longitude,
                'depth': origin.depth,
                'magnitude': magnitude.mag,
                'type': magnitude.magnitude_type
            })
        except:
            continue
            
    return pd.DataFrame(data)

def add_distances_and_backazimuth(station_lat, station_lon, events_df):
    """
    Add distance and backazimuth calculations to events DataFrame.
    
    Args:
        station_lat: Latitude of the station
        station_lon: Longitude of the station
        events_df: DataFrame containing event information
        
    Returns:
        pandas.DataFrame: DataFrame with added distance and backazimuth columns
    """
    distances_deg = []
    distances_km = []
    backazimuths = []
    
    for _, event in events_df.iterrows():
        # Calculate distance in degrees
        dist_deg = locations2degrees(station_lat[0], station_lon[0], 
                                   event.latitude, event.longitude)
        distances_deg.append(dist_deg)
        
        # Calculate distance in km and backazimuth
        dist_m, az, _ = gps2dist_azimuth(station_lat[0], station_lon[0],
                                        event.latitude, event.longitude)
        distances_km.append(dist_m/1000)  # Convert to km
        backazimuths.append(az)
    
    events_df['distances_deg'] = distances_deg
    events_df['distances_km'] = distances_km
    events_df['backazimuth'] = backazimuths
    
    return events_df 

def print_deviation_summary(analysis_results):
    """
    Print detailed summary of deviation analysis results
    
    Parameters:
    -----------
    analysis_results : dict
        Results from plot_backazimuth_deviation_analysis
    """
    import numpy as np
    
    deviations = analysis_results['deviations']
    theoretical_baz = analysis_results['theoretical_baz']
    center_freqs = analysis_results['center_frequencies']
    
    print("="*60)
    print("BACKAZIMUTH DEVIATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Theoretical Backazimuth: {theoretical_baz:.1f}°")
    print(f"Total Frequency Bands: {len(center_freqs)}")
    if 'bin_info' in analysis_results:
        print(f"Histogram binning: {analysis_results['bin_info']}")
    print()
    
    for wave_type, data in deviations.items():
        print(f"{wave_type.upper()} WAVES:")
        print(f"  Valid estimates: {len(data['deviation'])}/{len(center_freqs)}")
        print(f"  Frequency range: {data['frequencies'].min():.3f} - {data['frequencies'].max():.3f} Hz")
        print(f"  Mean deviation: {data['mean_deviation']:.2f}°")
        print(f"  Std deviation: {data['std_deviation']:.2f}°")
        print(f"  RMS deviation: {data['rms_deviation']:.2f}°")
        print(f"  Median deviation: {np.median(data['deviation']):.2f}°")
        print(f"  Max absolute deviation: {np.max(np.abs(data['deviation'])):.2f}°")
        print(f"  95th percentile: {np.percentile(np.abs(data['deviation']), 95):.2f}°")
        print()

def get_kde_stats(_baz, _ccc, _baz_steps=5, Ndegree=180, plot=False):
    """
    Get the statistics of the kde of the backazimuth and the cc values
    """
    import numpy as np
    import scipy.stats as sts
    import matplotlib.pyplot as plt

    # define angles for kde and histogram
    kde_angles = np.arange(0, 361, 1)
    hist_angles = np.arange(0, 365, 5)

    # get first kde estimate to determine the shift
    kde = sts.gaussian_kde(_baz, weights=_ccc, bw_method='scott')
    kde_max = np.argmax(kde.pdf(kde_angles))

    # determine the shift with respect to 180°
    shift = 180 - kde_max

    # shift the backazimuth array to the center of the x-axis
    _baz_shifted = (_baz + shift) % 360

    # get second kde estimate
    kde_shifted = sts.gaussian_kde(_baz_shifted, weights=_ccc, bw_method='scott')
    kde_max_shifted = np.argmax(kde_shifted.pdf(kde_angles))

    # get the estimate of the backazimuth corrected for the shift
    baz_estimate = kde_max_shifted - shift

    # shift new kde
    kde_angles_new = (kde_angles - shift) % 360
    kde_values_new = (kde_shifted.pdf(kde_angles)) % 360

    # resort the new kde
    idx = np.argsort(kde_angles_new)
    kde_angles_new = kde_angles_new[idx]
    kde_values_new = kde_values_new[idx]

    # get deviation
    dev = int(np.round(np.sqrt(np.cov(_baz_shifted, aweights=_ccc)), 0))


    if plot:
        plt.figure(figsize=(10, 5))

        plt.hist(_baz, bins=hist_angles, weights=_ccc, density=True, alpha=0.5)
        plt.hist(_baz_shifted, bins=hist_angles, weights=_ccc, density=True, alpha=0.5)

        plt.plot(kde_angles, kde.pdf(kde_angles), color='tab:blue')
        
        plt.plot(kde_angles, kde_shifted.pdf(kde_angles), color='tab:orange')


        plt.plot(kde_angles_new, kde_values_new, color='k')
        
        plt.scatter([kde_max], [max(kde.pdf(kde_angles))],
                    color='w', edgecolor='tab:blue', label=f'Max: {kde_max:.0f}°')
        plt.scatter([kde_max_shifted], [kde_shifted.pdf(kde_max_shifted)],
                    color='w', edgecolor='tab:orange', label=f'Max: {kde_max_shifted:.0f}° (shifted)')
        plt.scatter([baz_estimate], [max(kde_values_new)],
                    color='w', edgecolor='k', label=f'Estimate: {baz_estimate:.0f}°')
        
        # plot line between max and estimate
        plt.plot([kde_max, kde_max], [0, max(kde.pdf(kde_angles))], color='tab:blue', ls='--')
        plt.plot([kde_max_shifted, kde_max_shifted], [0, max(kde_values_new)], color='tab:orange', ls='--')
        plt.plot([baz_estimate, baz_estimate], [0, max(kde_values_new)], color='k', ls='--')

        plt.xlabel('Backazimuth (°)')   
        plt.ylabel('Density')
        plt.title('KDE of Backazimuth weighted by the CC value')

        plt.legend()

    # output
    out = {
        'baz_estimate': baz_estimate,
        'kde_max': kde_max_shifted,
        'shift': shift,
        'kde_values': kde_values_new,
        'kde_angles': kde_angles_new,
        'kde_dev': dev,
    }

    return out

def compute_adr_pfo(tbeg, tend, submask=None, status=False):

    import os
    import numpy as np
    import timeit
    import matplotlib.pyplot as plt
    import matplotlib.colors

    from obspy import UTCDateTime, Stream, read_inventory
    from obspy.clients import fdsn
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.geodetics import locations2degrees
    from obspy.clients.fdsn import Client, RoutingClient
    from obspy.signal import array_analysis as AA
    from obspy.signal.util import util_geo_km
    from obspy.signal.rotate import rotate2zne
    from datetime import datetime

    import warnings
    warnings.filterwarnings('ignore')

    if os.uname().nodename == 'lighthouse':
        root_path = '/home/andbro/'
        data_path = '/home/andbro/kilauea-data/'
        archive_path = '/home/andbro/freenas/'
        bay_path = '/home/andbro/bay200/'
    elif os.uname().nodename == 'kilauea':
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/bay200/'
    elif os.uname().nodename == 'lin-ffb-01':
        root_path = '/home/brotzer/'
        data_path = '/import/kilauea-data/'
        archive_path = '/import/freenas-ffb-01-data/'
        bay_path = '/bay200/'

    ## _____________________________________________________

    ## start timer for runtime
    start_timer = timeit.default_timer()


    ## _____________________________________________________

    ## generate configuration object
    config = {}

    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)

    ## select the fdsn client for the stations
    config['fdsn_client'] = Client('IRIS')


    ## select stations to consider: 
    if submask is not None:

        if submask == "inner":
            config['subarray_mask'] = [0,1,2,3,4]
            config['freq1'] = 1.0  ## 0.00238*3700/100
            config['freq2'] = 6.0 ## 0.25*3700/100 
        elif submask == "mid":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8]
            config['freq1'] = 0.5   ## 0.00238*3700/280
            config['freq2'] = 1.0   ## 0.25*3700/280
        elif submask == "all":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            config['freq1'] = 0.1   ## 0.00238*3700/700
            config['freq2'] = 0.5    ## 0.25*3700/700

    else:
        config['subarray_mask'] = [0,1,2,3,4]


    ## decide if information is printed while running the code
    config['print_details'] = False

    ## _____________________
    ## PFO array information

    if config['tbeg'] > UTCDateTime("2023-04-02"):
        config['reference_station'] = 'PY.PFOIX' ## 'BPH01'  ## reference station

        config['array_stations'] = ['PY.PFOIX','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                    'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']
    else:
        config['reference_station'] = 'II.PFO' ## 'BPH01'  ## reference station

        config['array_stations'] = ['II.PFO','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                    'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']


#     config['misorientations'] =  [0, 0. ,-1.375 ,0.25 ,0.125 ,-0.6875 ,-0.625 ,-1.9375 ,0.375 
#                                   ,-6.5625 ,0.3125 ,-1.125 ,-2.5625 ,0.1875]

#     config['subarray_misorientation'] = [config['misorientations'][i] for i in config['subarray_mask']]

    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]
    config['subarray_sta'] = config['subarray_stations']

    ## ______________________________
    ## parameter for array-derivation

    #config['prefilt'] = (0.001, 0.01, 5, 10)
    config['apply_bandpass'] = True


    # adr parameters
    config['vp'] = 6200 #6264. #1700
    config['vs'] = 3700 #3751. #1000
    config['sigmau'] = 1e-7 # 0.0001


    ## _____________________________________________________


    def __get_inventory_and_distances(config):

        coo = []
        for i, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            if net == "II" and sta == "PFO":
                loc, cha = "10", "BH*"
            elif net == "II" and sta == "XPFO":
                loc, cha = "30", "BH*"
            elif net == "PY" and sta == "PFOIX":
                loc, cha = "", "HH*"
            else:
                loc, cha = "", "BH*"

            try:
                ## load local version
                inven = read_inventory(data_path+f"BSPF/data/stationxml/{net}.{sta}.xml")
            except:
                inven = config['fdsn_client'].get_stations(network=net,
                                                           station=sta,
                                                           channel=cha,
                                                           location=loc,
                                                           starttime=config['tbeg'],
                                                           endtime=config['tend'],
                                                           level='response'
                                                          )

            l_lon =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat =  float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])


            ## set coordinates of seismometer manually, since STATIONXML is wrong...
            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                l_lon, l_lat =  -116.455439, 33.610643


            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon, o_lat, l_lon, l_lat)

            coo.append([lon*1000, lat*1000, height-o_height])  ## convert unit from km to m

        return inven, np.array(coo)


    def __check_samples_in_stream(st, config):

        for tr in st:
            if tr.stats.npts != config['samples']:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {config['samples']})")
                st.remove(tr)

        return st


    def __get_data(config):


        config['subarray'] = []

        st = Stream()

        for k, station in enumerate(config['subarray_stations']):

            net, sta = station.split(".")

            if net == "II" and sta == "PFO":
                loc, cha = "10", "BH*"
            elif net == "II" and sta == "XPFO":
                loc, cha = "30", "BH*"
            elif net == "PY" and sta == "PFOIX":
                loc, cha = "", "HH*"
            else:
                loc, cha = "", "BH*"

            print(f" -> requesting {net}.{sta}.{loc}.{cha}") if config['print_details'] else None


            ## querry inventory data
            try:
                try:
                    ## load local version
                    inventory = read_inventory(data_path+f"BSPF/data/stationxml/{net}.{sta}.xml")
                except:
                    inventory = config['fdsn_client'].get_stations(
                                                                network=net,
                                                                station=sta,
                                                                location=loc,
                                                                channel=cha,
                                                                starttime=config['tbeg']-30,
                                                                endtime=config['tend']+30,
                                                                level="response"
                                                                )
            except:
                print(f" -> {sta} Failed to load inventory!")
                inventory = None


            ## try to get waveform data
            try:
                stats = config['fdsn_client'].get_waveforms(
                                                            network=net,
                                                            station=sta,
                                                            location=loc,
                                                            channel=cha,
                                                            starttime=config['tbeg']-30,
                                                            endtime=config['tend']+30,
                                                            attach_response=True,
                                                            )
            except Exception as E:
                print(E) if config['print_details'] else None
                print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...")
                config['stations_loaded'][k] = 0
                continue

            ## merge if masked
            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3") if config['print_details'] else None
                stats.merge(method=1, fill_value="interpolate")



            ## remove response [VEL -> rad/s | DISP -> rad]
            # stats = stats.remove_sensitivity(inventory)
            stats.remove_response(inventory, output="VEL", water_level=60)


            #correct mis-alignment
            # stats[0].data, stats[1].data, stats[2].data = rotate2zne(stats[0],0,-90,
            #                                                          stats[1],config['subarray_misorientation'][config['subarray_stations'].index(station)],0, 
            #                                                          stats[2],90+config['subarray_misorientation'][config['subarray_stations'].index(station)],0)



            ## rotate to ZNE
            try:
                stats = stats.rotate(method="->ZNE", inventory=inventory)
            except:
                print(f" -> {sta} failed to rotate to ZNE")
                continue

            ## resampling using decitmate
            # stats = stats.detrend("linear");
            # stats = stats.taper(0.01);
            # stats = stats.filter("lowpass", freq=18, corners=4, zerophase=True);
            # if station == "PY.PFOIX":
            #     stats = stats.decimate(5, no_filter=True); ## 200 Hz -> 40 Hz
            # else:
            #     stats = stats.decimate(2, no_filter=True); ## 40 Hz -> 20 Hz

            ## resample all to 40 Hz
            stats = stats.resample(40, no_filter=False)

            if station == config['reference_station']:
                # ref_station = stats.copy().resample(40, no_filter=False)
                ref_station = stats.copy()

            st += stats
            config['subarray'].append(f"{stats[0].stats.network}.{stats[0].stats.station}")

        ## trim to interval
        # stats.trim(config['tbeg'], config['tend'], nearest_sample=False)

        st = st.sort()


        config['subarray_stations'] = config['subarray']

        print(f" -> obtained: {len(st)/3} of {len(config['subarray_stations'])} stations!") if config['print_details'] else None

        if len(st) == 0:
            return st, Stream(), config
        else:
            return st, ref_station, config


    def __compute_ADR(tse, tsn, tsz, config, ref_station):

        ## make sure input is array type
        tse, tsn, tsz = np.array(tse), np.array(tsn), np.array(tsz)

        ## define array for subarray stations with linear numbering
        substations = np.arange(len(config['subarray_stations']))

        try:
            result = AA.array_rotation_strain(substations,
                                              np.transpose(tse),
                                              np.transpose(tsn),
                                              np.transpose(tsz),
                                              config['vp'],
                                              config['vs'],
                                              config['coo'],
                                              config['sigmau'],
                                             )
        except Exception as E:
            print(E)
            print("\n -> failed to compute ADR...")
            return None

        ## create rotation stream and add data
        rotsa = ref_station.copy()

        rotsa[0].data = result['ts_w3']
        rotsa[1].data = result['ts_w2']
        rotsa[2].data = result['ts_w1']

        rotsa[0].stats.channel='BJZ'
        rotsa[1].stats.channel='BJN'
        rotsa[2].stats.channel='BJE'

        rotsa[0].stats.station='RPFO'
        rotsa[1].stats.station='RPFO'
        rotsa[2].stats.station='RPFO'

        rotsa = rotsa.detrend('linear')

        return rotsa

    ## __________________________________________________________
    ## MAIN ##

    ## launch a times
    start_timer1 = timeit.default_timer()

    ## status of stations loaded
    config['stations_loaded'] = np.ones(len(config['subarray_stations']))

    ## request data for pfo array
    st, ref_station, config = __get_data(config)


    ## check if enough stations for ADR are available otherwise continue
    if len(st) < 9:
        print(" -> not enough stations (< 3) for ADR computation!")
        return
    else:
        print(f" -> continue computing ADR for {int(len(st)/3)} of {len(config['subarray_mask'])} stations ...")

    ## get inventory and coordinates/distances
    inv, config['coo'] = __get_inventory_and_distances(config)

    ## processing
    st.detrend("demean")

    if config['apply_bandpass']:
        st.taper(0.01)
        st.filter('bandpass', freqmin=config['freq1'], freqmax=config['freq2'], corners=4, zerophase=True)
        print(f" -> bandpass: {config['freq1']} - {config['freq2']} Hz")


    ## plot station coordinates for check up
    # import matplotlib.pyplot as plt
    # plt.figure()
    # for c in config['coo']:
    #     print(c)
    #     plt.scatter(c[0], c[1])


    ## prepare data arrays
    tsz, tsn, tse = [], [], []
    for tr in st:
        try:
            if "Z" in tr.stats.channel:
                tsz.append(tr.data)
            elif "N" in tr.stats.channel:
                tsn.append(tr.data)
            elif "E" in tr.stats.channel:
                tse.append(tr.data)
        except:
            print(" -> stream data could not be appended!")

    ## compute array derived rotation (ADR)
    rot = __compute_ADR(tse, tsn, tsz, config, ref_station)


    ## get mean starttime
    tstart = [tr.stats.starttime - tbeg for tr in st]
    for tr in rot:
        tr.stats.starttime = tbeg + np.mean(tstart)


    ## trim to requested interval
    rot = rot.trim(config['tbeg'], config['tend'])


    ## plot status of data retrieval for waveforms of array stations
    if status:

        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])

        ax.pcolormesh(np.array([config['stations_loaded'], np.ones(len(config['stations_loaded']))*0.5]).T, cmap=cmap, edgecolors="k", lw=0.5)

        ax.set_yticks(np.arange(0, len(config['subarray_sta']))+0.5, labels=config['subarray_sta'])

        # ax.set_xlabel("Event No.",fontsize=12)
        ax.set_xticks([])
        ax.set_xlim(0, 1)

        plt.show();


    ## stop times
    stop_timer1 = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer1 - start_timer1)/60, 2)} minutes\n")

    if status:
        return rot, config['stations_loaded']
    else:
        return rot

def compute_beamforming_pfo(tbeg, tend, submask, fmin: Optional[float] = None, fmax: Optional[float] = None, component: str = "", bandpass: bool = True, plot: bool = False) -> Tuple[Stream, List[int]]:

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

def compute_frequency_dependent_backazimuth(st, params, plot=False):
    """
    Compute frequency-dependent backazimuth analysis using array processing.
    
    Parameters:
    -----------
    st : obspy.Stream
        Stream containing array data
    params : dict
        Dictionary containing frequency bands and processing parameters
    plot : bool, optional
        Whether to plot the results (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing the analysis results
    """
    import numpy as np
    from obspy.signal.array_analysis import array_processing
    import matplotlib.pyplot as plt
    
    results = {
        'times': [],
        'frequency': {'center': [], 'min': [], 'max': []},
        'backazimuth': {'optimal': [], 'mean': [], 'std': []},
        'velocity': [],
        'cross_correlation': {'optimal': [], 'mean': [], 'std': []}
    }
    
    # Process each frequency band
    for fmin, fmax in zip(params['freq_min'], params['freq_max']):
        
        # Configure array processing parameters with explicit types
        kwargs = {
            'sll_x': float(-params['slowness_max']),
            'slm_x': float(params['slowness_max']),
            'sll_y': float(-params['slowness_max']), 
            'slm_y': float(params['slowness_max']),
            'sl_s': float(params['slowness_step']),
            'win_len': float(params['window_length']),
            'win_frac': float(params['window_fraction']),
            'frqlow': float(fmin),
            'frqhigh': float(fmax),
            'prewhiten': int(params['prewhitening']),
            'semb_thres': -1e9,
            'vel_thres': -1e9,
            'timestamp': 'mlabday',
            'stime': st[0].stats.starttime,
            'etime': st[0].stats.endtime,
            'method': 0,  # Explicitly set method
            'coordsys': 'lonlat',  # Explicitly set coordinate system
            'verbose': False  # Explicitly set verbose
        }

        # Perform array processing
        out = array_processing(st, **kwargs)
        
        # Extract results
        times = out[:, 0]
        rel_power = out[:, 1]
        abs_power = out[:, 2] 
        baz = out[:, 3]
        slowness = out[:, 4]
        
        # Fix backazimuth values
        baz[baz < 0.0] += 360
        
        # Calculate velocities
        velocity = 1.0 / slowness
        
        # Store results
        results['times'].extend(times)
        results['frequency']['center'].extend([np.mean([fmin, fmax])] * len(times))
        results['frequency']['min'].extend([fmin] * len(times))
        results['frequency']['max'].extend([fmax] * len(times))
        results['backazimuth']['optimal'].extend(baz)
        results['velocity'].extend(velocity)
        results['cross_correlation']['optimal'].extend(abs_power)
        
        # Calculate statistics
        results['backazimuth']['mean'].append(np.mean(baz))
        results['backazimuth']['std'].append(np.std(baz))
        results['cross_correlation']['mean'].append(np.mean(abs_power))
        results['cross_correlation']['std'].append(np.std(abs_power))

    if plot:
        fig = plot_frequency_backazimuth_analysis(results)
        return results, fig
    
    return results

def plot_tangent_method_comparison(results_rot, results_acc, event_info=None, figsize=(12, 6)):
    """
    Simple comparison plot of rotation vs acceleration tangent methods
    
    Parameters:
    -----------
    results_rot : dict
        Results from tangent method using rotation components
    results_acc : dict  
        Results from tangent method using acceleration components
    event_info : dict, optional
        Event information with theoretical backazimuth
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Simple comparison plot figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    
    # Extract data
    baz_rot = results_rot['cc_max_y'] 
    cc_rot = results_rot['cc_max']
    
    baz_acc = results_acc['cc_max_y']
    cc_acc = results_acc['cc_max']
    
    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins every 10 degrees
    bins = np.arange(0, 361, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = 10
    
    # Plot histograms with offset positioning
    bar_width = bin_width * 0.35
    
    # Rotation histogram (left side of bins)
    counts_rot, _ = np.histogram(baz_rot, bins=bins, density=True)
    ax.bar(bin_centers - bar_width/2, counts_rot, width=bar_width, 
           alpha=0.6, color='tab:blue', edgecolor='darkblue', linewidth=1.5,
           label=f'Rotation (N={len(baz_rot)})')
    
    # Acceleration histogram (right side of bins)
    counts_acc, _ = np.histogram(baz_acc, bins=bins, density=True)
    ax.bar(bin_centers + bar_width/2, counts_acc, width=bar_width, 
           alpha=0.6, color='tab:orange', edgecolor='brown', linewidth=1.5,
           label=f'Acceleration (N={len(baz_acc)})')
    
    # Add KDE curves
    if len(baz_rot) > 1:
        # get kde stats (pad)
        kde_stats_rot = get_kde_stats(baz_rot, cc_rot, _baz_steps=0.5, Ndegree=60)
        baz_estimate_rot = kde_stats_rot['baz_estimate']
        baz_std_rot = kde_stats_rot['kde_dev']
        kde_max_rot = max(kde_stats_rot['kde_values'])

        ax.plot(kde_stats_rot['kde_angles'], 
                kde_stats_rot['kde_values'],
                color='darkblue', linewidth=2.5,
                alpha=0.9, label='Rotation KDE'
                )

    if len(baz_acc) > 1:
        # get kde stats (pad)
        kde_stats_acc = get_kde_stats(baz_acc, cc_acc, _baz_steps=0.5, Ndegree=60)
        baz_estimate_acc = kde_stats_acc['baz_estimate']
        baz_std_acc = kde_stats_acc['kde_dev']
        kde_max_acc = max(kde_stats_acc['kde_values'])

        ax.plot(kde_stats_acc['kde_angles'], 
                kde_stats_acc['kde_values'],
                color='brown', linewidth=2.5,
                alpha=0.9, label='Acceleration KDE'
                )

    # Add theoretical backazimuth if available
    if event_info and 'backazimuth' in event_info:
        theo_baz = event_info['backazimuth']
        ax.axvline(theo_baz, color='green', linestyle='--', 
                   linewidth=3, label=f'Theoretical: {theo_baz:.1f}°')
    
    # get estatimates from results_rot and results_acc
    rot_baz_estimate = round(results_rot['baz_estimate'], 0)
    acc_baz_estimate = round(results_acc['baz_estimate'], 0)
    rot_baz_estimate_std = round(results_rot['baz_estimate_std'], 0)
    acc_baz_estimate_std = round(results_acc['baz_estimate_std'], 0)

    print(f"rot_baz_estimate: {rot_baz_estimate}, acc_baz_estimate: {acc_baz_estimate}")
    
    # Add statistics text
    stats_text = f"Rotation: {rot_baz_estimate}° ± {rot_baz_estimate_std}°\n"
    stats_text += f"Acceleration: {acc_baz_estimate}° ± {acc_baz_estimate_std}°\n"
    
    # Calculate difference
    diff = abs(rot_baz_estimate - acc_baz_estimate)
    if diff > 180:
        diff = 360 - diff
    # stats_text += f"Difference: {diff:.1f}°"
    
    # Add deviations if theoretical available
    if event_info and 'backazimuth' in event_info:
        dev_rot = abs(rot_baz_estimate - theo_baz)
        if dev_rot > 180:
            dev_rot = 360 - dev_rot
        dev_acc = abs(acc_baz_estimate - theo_baz)
        if dev_acc > 180:
            dev_acc = 360 - dev_acc
        stats_text += f"\nRot. Dev.: {dev_rot}°\nAcc. Dev.: {dev_acc}°"
    
    # add max_rot and max_acc as vertical lines between 0 and max value
    ax.plot([rot_baz_estimate, rot_baz_estimate], [0, kde_max_rot],
            color='darkblue', linestyle='--', linewidth=2,
            label=f'Rotation Max: {rot_baz_estimate} ± {rot_baz_estimate_std}°'
            )
    ax.plot([acc_baz_estimate, acc_baz_estimate], [0, kde_max_acc],
            color='darkred', linestyle='--', linewidth=2,
            label=f'Acceleration Max: {acc_baz_estimate} ± {acc_baz_estimate_std}°'
            )

    # Position statistics text
    # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
    #         verticalalignment='top', fontsize=11, fontfamily='monospace',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Configure plot
    ax.set_xlabel('Backazimuth (°)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Tangent Method Comparison: Rotation vs Acceleration Components', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.minorticks_on()

    # Remove 0.00 tick label from density axis
    yticks = ax.get_yticks()
    yticks_filtered = yticks[yticks > 0.001]
    if len(yticks_filtered) > 0:
        ax.set_yticks(yticks_filtered)
    
    plt.tight_layout()
    return fig

def plot_backazimuth_deviation_analysis(results, event_info, figsize=(15, 8), bin_step=None):
    """
    Plot deviation analysis between estimated and theoretical backazimuth
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_dependent_backazimuth
    event_info : dict
        Event information with 'backazimuth' key for theoretical comparison
    figsize : tuple
        Figure size (width, height)
    bin_step : float, optional
        Bin spacing in degrees (e.g., 5 for bins every 5 degrees). 
        If None, uses automatic binning with 20 bins.
        
    Returns:
    --------
    tuple : (figure, analysis_results)
        Figure object and dictionary with deviation analysis results
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from scipy import stats
    
    if 'backazimuth' not in event_info:
        print("No theoretical backazimuth available in event_info")
        return None, {}
    
    theoretical_baz = event_info['backazimuth']
    wave_types = list(results['wave_types'].keys())
    n_wave_types = len(wave_types)
    
    if n_wave_types == 0:
        print("No wave type results to analyze")
        return None, {}
    
    # Calculate deviations for each wave type
    deviations = {}
    center_freqs = results['frequency_bands']['center']
    
    for wave_type in wave_types:
        peak_baz = results['wave_types'][wave_type]['peak_baz']
        valid_mask = ~np.isnan(peak_baz)
        
        if np.any(valid_mask):
            # Calculate angular deviation (considering circular nature of angles)
            deviation = peak_baz[valid_mask] - theoretical_baz
            # Wrap to [-180, 180] range
            deviation = ((deviation + 180) % 360) - 180
            
            deviations[wave_type] = {
                'deviation': deviation,
                'frequencies': center_freqs[valid_mask],
                'all_deviation': np.full_like(peak_baz, np.nan),
                'mean_deviation': np.mean(deviation),
                'std_deviation': np.std(deviation),
                'rms_deviation': np.sqrt(np.mean(deviation**2))
            }
            
            # Store all deviations (including NaN for missing estimates)
            all_dev = np.full_like(peak_baz, np.nan)
            dev_calc = peak_baz - theoretical_baz
            dev_calc = ((dev_calc + 180) % 360) - 180
            all_dev[valid_mask] = deviation
            deviations[wave_type]['all_deviation'] = dev_calc
        else:
            print(f"No valid estimates for {wave_type}")
            continue
    
    if not deviations:
        print("No valid deviations to plot")
        return None, {}
    
    # Create figure with layout: main plot + single merged histogram
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[3, 0.8, 0.1, 0.3], 
                 hspace=0.0, wspace=0.0)
    ax_freq = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_freq)
    
    # Colors for different wave types
    colors = {'love': 'blue', 'rayleigh': 'red'}
    
    # Plot 1: Deviation vs Frequency with lines to zero (no regression)
    for i, (wave_type, data) in enumerate(deviations.items()):
        color = colors.get(wave_type, f'C{i}')
        
        # Plot vertical lines from zero to each point
        for freq, dev in zip(data['frequencies'], data['deviation']):
            ax_freq.plot([freq, freq], [0, dev], color=color, alpha=0.3, linewidth=1)
        
        # Plot deviation vs frequency markers
        ax_freq.semilogx(data['frequencies'], data['deviation'], 
                        'o', color=color, alpha=0.8, markersize=8,
                        label=f'{wave_type.upper()} waves', markeredgecolor='black',
                        markeredgewidth=0.5)
    
    # Reference line at zero deviation
    ax_freq.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax_freq.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_freq.set_ylabel('Deviation from Theoretical BAZ (°)', fontsize=12)
    ax_freq.set_title('Backazimuth Deviation vs Frequency', fontsize=14, fontweight='bold')
    
    # Add grid and subgrid
    ax_freq.grid(True, which='major', alpha=0.5, linewidth=1)
    ax_freq.grid(True, which='minor', alpha=0.3, linewidth=0.5)
    ax_freq.minorticks_on()
    
    ax_freq.legend(loc='upper left')
    
    # Determine binning strategy
    all_deviations = np.concatenate([data['deviation'] for data in deviations.values()])
    
    if bin_step is not None:
        # Use fixed degree spacing
        data_min, data_max = np.min(all_deviations), np.max(all_deviations)
        
        # Extend range to nearest bin_step boundaries
        bin_min = np.floor(data_min / bin_step) * bin_step
        bin_max = np.ceil(data_max / bin_step) * bin_step
        
        # Create bins every bin_step degrees
        common_bins = np.arange(bin_min, bin_max + bin_step, bin_step)
        bin_info = f"(bins every {bin_step}°)"
    else:
        # Use automatic binning
        n_bins = 20
        bin_range = (np.min(all_deviations), np.max(all_deviations))
        common_bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
        bin_info = f"({len(common_bins)-1} bins)"
    
    # Plot 2: Optimized histogram with KDE overlay
    if n_wave_types == 1:
        # Single wave type
        wave_type = list(deviations.keys())[0]
        data = deviations[wave_type]
        color = colors.get(wave_type, 'blue')
        
        # Create histogram
        counts, bins, patches = ax_hist.hist(data['deviation'], bins=common_bins, alpha=0.6, color=color, 
                                           edgecolor='black', density=True, orientation='horizontal')
        
        # Add KDE overlay
        if len(data['deviation']) > 1:
            kde = stats.gaussian_kde(data['deviation'])
            y_kde = np.linspace(data['deviation'].min(), data['deviation'].max(), 100)
            kde_values = kde(y_kde)
            ax_hist.plot(kde_values, y_kde, color=color, linewidth=2, alpha=0.8, label='KDE')
    
    else:
        # Multiple wave types - optimized layout with bars left/right of bin centers
        bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
        bin_width = np.diff(common_bins)[0]
        
        # Calculate bar positioning
        bar_width = bin_width * 0.35  # Narrower bars
        positions = [-bar_width/2, bar_width/2] if n_wave_types == 2 else [0]  # Left/right positioning
        
        kde_curves = {}  # Store KDE curves for overlay
        
        for i, (wave_type, data) in enumerate(deviations.items()):
            color = colors.get(wave_type, f'C{i}')
            
            # Calculate histogram counts
            counts, _ = np.histogram(data['deviation'], bins=common_bins, density=True)
            
            # Position bars left/right of bin centers
            if n_wave_types > 1:
                offset_bins = bin_centers + positions[i]
            else:
                offset_bins = bin_centers
            
            # Plot bars
            bars = ax_hist.barh(offset_bins, counts, height=bar_width, 
                              color=color, alpha=0.6, edgecolor='black', linewidth=0.5,
                              label=f'{wave_type.upper()}')
            
            # Calculate and store KDE for overlay
            if len(data['deviation']) > 1:
                kde = stats.gaussian_kde(data['deviation'])
                y_kde = np.linspace(common_bins[0], common_bins[-1], 100)
                kde_values = kde(y_kde)
                kde_curves[wave_type] = {'y': y_kde, 'kde': kde_values, 'color': color}
        
        # Overlay KDE curves
        for wave_type, kde_data in kde_curves.items():
            ax_hist.plot(kde_data['kde'], kde_data['y'], 
                        color=kde_data['color'], linewidth=2.5, alpha=0.9,
                        linestyle='-', label=f'{wave_type.upper()} KDE')
    
    # Zero reference line in histogram
    ax_hist.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    ax_hist.set_xlabel('Density', fontsize=10)
    ax_hist.set_title(f'Distribution {bin_info}', fontsize=11)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.tick_params(labelleft=False)  # Remove y-axis labels
    
    # Remove 0.00 tick label from density axis
    xticks = ax_hist.get_xticks()
    xticks_filtered = xticks[xticks > 0.001]  # Remove ticks close to zero
    if len(xticks_filtered) > 0:
        ax_hist.set_xticks(xticks_filtered)
    
    # Add legend to histogram
    if n_wave_types > 1:
        ax_hist.legend(loc='upper right', fontsize=8)
    elif n_wave_types == 1 and len(list(deviations.values())[0]['deviation']) > 1:
        ax_hist.legend(loc='upper right', fontsize=8)
    
    # Overall title
    plt.suptitle('Backazimuth Estimation Deviation Analysis', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Return results for further analysis (without printing summary)
    analysis_results = {
        'deviations': deviations,
        'theoretical_baz': theoretical_baz,
        'center_frequencies': center_freqs,
        'bin_info': bin_info
    }
    
    return fig, analysis_results

def plot_frequency_backazimuth_analysis(results, event_info=None, vmax_percentile=95,
                                       figsize=(12, 10), show_peak_line=True):
    """
    Plot frequency-dependent backazimuth analysis results
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_dependent_backazimuth
    event_info : dict, optional
        Event information with 'backazimuth' key for theoretical comparison
    vmax_percentile : float
        Percentile for color scale maximum (to avoid outliers)
    figsize : tuple
        Figure size (width, height)
    show_peak_line : bool
        Whether to show line connecting peak estimates
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    
    wave_types = list(results['wave_types'].keys())
    n_wave_types = len(wave_types)
    
    if n_wave_types == 0:
        print("No wave type results to plot")
        return None
    
    # Create figure
    fig, axes = plt.subplots(n_wave_types, 1, figsize=figsize, sharex=True)
    if n_wave_types == 1:
        axes = [axes]
    
    # Get frequency data
    center_freqs = results['frequency_bands']['center']
    baz_grid = results['baz_grid']
    
    # Create meshgrid for pcolormesh
    freq_edges = np.logspace(np.log10(center_freqs.min()), np.log10(center_freqs.max()), len(center_freqs) + 1)
    baz_edges = np.arange(0, 361, np.diff(baz_grid)[0])
    
    colors = {'love': 'Blues', 'rayleigh': 'Reds'}
    
    for i, wave_type in enumerate(wave_types):
        ax = axes[i]
        data = results['wave_types'][wave_type]
        
        # Get KDE values and normalize for better visualization
        kde_matrix = data['kde_values'].T  # Transpose for correct orientation
        
        # Set colormap limits
        kde_nonzero = kde_matrix[kde_matrix > 0]
        if len(kde_nonzero) > 0:
            vmax = np.percentile(kde_nonzero, vmax_percentile)
            vmin = np.percentile(kde_nonzero, 5)
        else:
            vmax = 1.0
            vmin = 0.01
        
        # Create pcolormesh plot
        colormap = colors.get(wave_type, 'viridis')
        im = ax.pcolormesh(center_freqs, baz_grid, kde_matrix, 
                          cmap=colormap, shading='auto',
                          vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='KDE Density', pad=0.02)
        
        # Plot peak line if requested
        if show_peak_line:
            valid_peaks = ~np.isnan(data['peak_baz'])
            if np.any(valid_peaks):
                ax.plot(center_freqs[valid_peaks], data['peak_baz'][valid_peaks], 
                       'k-', linewidth=2, alpha=0.8, label='Peak BAZ')
                ax.scatter(center_freqs[valid_peaks], data['peak_baz'][valid_peaks], 
                       color='k', marker='o', alpha=0.8, facecolor='white', zorder=3)
        
        # Plot theoretical backazimuth if available
        if event_info and 'backazimuth' in event_info:
            ax.axhline(y=event_info['backazimuth'], color='grey', 
                      linestyle='--', linewidth=2, alpha=0.9, label='Theoretical BAZ')
        
        # Customize axes
        ax.set_xscale('log')
        ax.set_ylabel('Backazimuth (°)')
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(0, 361, 60))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{wave_type.upper()} Wave Backazimuth vs Frequency', 
                    fontsize=12, fontweight='bold')
        
        # Add legend if there are lines to show
        if show_peak_line or (event_info and 'backazimuth' in event_info):
            ax.legend(loc='upper right')
        
        # Add statistics text
        n_bands_with_data = np.sum(data['n_estimates'] > 0)
        stats_text = f'Bands with data: {n_bands_with_data}/{len(center_freqs)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               verticalalignment='top', fontsize=10)
    
    # Set x-label only for bottom subplot
    axes[-1].set_xlabel('Frequency (Hz)')
    
    # Main title
    octave_frac = results['parameters']['octave_fraction']
    plt.suptitle(f'Frequency-Dependent Backazimuth Analysis (1/{octave_frac} Octave Bands)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_backazimuth_map(results, event_info=None, map_projection='orthographic', 
                        bin_step=5, figsize=(15, 8), debug=False):
    """
    Plot backazimuth estimation results from compute_backazimuth_simple
    
    Parameters:
    -----------
    results : dict
        Results from compute_backazimuth_simple()
    event_info : dict, optional
        Event information for comparison
    map_projection : str
        Map projection type ('orthographic' or 'platecarree')
    bin_step : float
        Bin spacing in degrees for histograms
    figsize : tuple, optional
        Figure size (width, height). If None, auto-determined.
    debug : bool
        Enable debugging output
        
    Returns:
    --------
    matplotlib.figure.Figure
        Backazimuth analysis plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as sts
    from matplotlib.gridspec import GridSpec
    
    if debug:
        print("\n" + "="*60)
        print("BACKAZIMUTH MAP PLOTTING DEBUG")
        print("="*60)

    def _plot_spherical_map_backazimuth(ax, event_info, baz_estimates, station_lat, station_lon, 
                                        projection='orthographic'):
        """Plot 2D spherical map with backazimuth information"""
        import numpy as np
        
        if debug:
            print(f"\n--- MAP PLOTTING ---")
            print(f"Station: ({station_lat:.6f}, {station_lon:.6f})")
            print(f"Projection: {projection}")
            print(f"Estimates: {baz_estimates}")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            use_cartopy = True
            if debug:
                print("✓ Using Cartopy")
        except ImportError:
            use_cartopy = False
            if debug:
                print("⚠ Using matplotlib fallback")
        
        # Set up map features
        if use_cartopy:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.7)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.6)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.8)
            
            if projection == 'orthographic':
                ax.gridlines(alpha=0.5)
                ax.set_global()
            else:
                gl = ax.gridlines(draw_labels=True, alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
            
            transform = ccrs.PlateCarree()
        else:
            if projection == 'orthographic':
                theta = np.linspace(0, 2*np.pi, 100)
                ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
                ax.axis('off')
            else:
                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
                ax.grid(True, alpha=0.5)
            transform = None
        
        # Validate and normalize coordinates
        if not (-90 <= station_lat <= 90):
            print(f"ERROR: Invalid station latitude: {station_lat}")
            return
        station_lon_norm = ((station_lon + 180) % 360) - 180
        
        if debug and abs(station_lon_norm - station_lon) > 1e-6:
            print(f"Normalized station longitude: {station_lon} → {station_lon_norm}")
        
        # Plot station
        if use_cartopy:
            ax.plot(station_lon_norm, station_lat, marker='^', color='red', markersize=15,
                    label='Station', markeredgecolor='black', markeredgewidth=2,
                    transform=transform, zorder=5)
            if debug:
                print(f"✓ Station plotted at ({station_lon_norm}, {station_lat})")
        else:
            if projection == 'orthographic':
                x_st, y_st = _project_to_sphere(station_lat, station_lon_norm, station_lat, station_lon_norm)
                ax.plot(x_st, y_st, marker='^', color='red', markersize=15,
                        label='Station', markeredgecolor='black', markeredgewidth=2, zorder=5)
                if debug:
                    print(f"✓ Station plotted at sphere coords ({x_st:.3f}, {y_st:.3f})")
            else:
                ax.plot(station_lon_norm, station_lat, marker='^', color='red', markersize=15,
                        label='Station', markeredgecolor='black', markeredgewidth=2, zorder=5)
        
        # Plot event if available
        if event_info and 'latitude' in event_info and 'longitude' in event_info:
            event_lat = event_info['latitude']
            event_lon = ((event_info['longitude'] + 180) % 360) - 180
            
            if debug:
                print(f"Event: ({event_lat:.6f}, {event_lon:.6f})")
            
            if use_cartopy:
                ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=20,
                        label='Event', markeredgecolor='black', markeredgewidth=2,
                        transform=transform, zorder=5)
            else:
                if projection == 'orthographic':
                    x_ev, y_ev = _project_to_sphere(event_lat, event_lon, station_lat, station_lon_norm)
                    ax.plot(x_ev, y_ev, marker='*', color='yellow', markersize=20,
                            label='Event', markeredgecolor='black', markeredgewidth=2, zorder=5)
                else:
                    ax.plot(event_lon, event_lat, marker='*', color='yellow', markersize=20,
                            label='Event', markeredgecolor='black', markeredgewidth=2, zorder=5)
        
        # Plot great circles
        colors = {'love': 'blue', 'rayleigh': 'red', 'tangent': 'purple'}
        
        # Theoretical great circle first
        if event_info and 'backazimuth' in event_info:
            theo_baz = event_info['backazimuth']
            if debug:
                print(f"\nTheoretical BAZ: {theo_baz:.1f}°")
            
            try:
                if use_cartopy:
                    gc_lons, gc_lats = _great_circle_path_2d(
                        station_lat, station_lon_norm, theo_baz)
                    
                    # CRITICAL FIX: Check that great circle starts at station
                    if debug:
                        lat_diff = abs(gc_lats[0] - station_lat)
                        lon_diff = abs(gc_lons[0] - station_lon_norm)
                        print(f"  Theoretical GC start: ({gc_lats[0]:.6f}, {gc_lons[0]:.6f})")
                        print(f"  Station coords:       ({station_lat:.6f}, {station_lon_norm:.6f})")
                        print(f"  Difference:           ({lat_diff:.2e}, {lon_diff:.2e})")
                    
                    ax.plot(gc_lons, gc_lats, color='green', linewidth=4, 
                        linestyle=':', label=f'Theoretical: {theo_baz:.f}°', alpha=0.9,
                        transform=transform, zorder=3)
                else:
                    _plot_great_circle_basic(ax, station_lat, station_lon_norm, 
                                                theo_baz, 'green', f'Theoretical: {theo_baz:.1f}°', 
                                                projection, linestyle=':')
            except Exception as e:
                print(f"ERROR plotting theoretical great circle: {e}")
        
        # Estimated great circles
        for wave_type, baz_deg in baz_estimates.items():
            if debug:
                print(f"\n{wave_type.upper()} BAZ: {baz_deg:.1f}°")
            
            try:
                color = colors.get(wave_type, 'purple')
                
                if use_cartopy:
                    gc_lons, gc_lats = _great_circle_path_2d(
                        station_lat, station_lon_norm, baz_deg)
                    
                    # CRITICAL FIX: Verify great circle starts at station
                    if debug:
                        lat_diff = abs(gc_lats[0] - station_lat)
                        lon_diff = abs(gc_lons[0] - station_lon_norm)
                        print(f"  {wave_type} GC start: ({gc_lats[0]:.6f}, {gc_lons[0]:.6f})")
                        print(f"  Difference:           ({lat_diff:.2e}, {lon_diff:.2e})")
                        if lat_diff > 1e-6 or lon_diff > 1e-6:
                            print(f"  ⚠ WARNING: Great circle doesn't start at station!")
                    
                    ax.plot(gc_lons, gc_lats, color=color, linewidth=3, 
                        label=f'{wave_type.upper()}: {baz_deg:.0f}°', alpha=0.8,
                        transform=transform, zorder=4)
                else:
                    _plot_great_circle_basic(ax, station_lat, station_lon_norm, 
                                                baz_deg, color, f'{wave_type.upper()}: {baz_deg:.1f}°', 
                                                projection)
            except Exception as e:
                print(f"ERROR plotting {wave_type} great circle: {e}")
        
        # Set title and legend
        # ax.set_title('Geographic View', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(0.75, 1.1), loc='upper left')

    def _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=120, num_points=100):
        """
        FIXED: Calculate great circle path points ensuring start at station
        """
        import numpy as np
        
        if debug:
            print(f"    Computing great circle: lat0={lat0:.6f}, lon0={lon0:.6f}, az={azimuth:.1f}°")
        
        # Validate inputs
        if not (-90 <= lat0 <= 90):
            raise ValueError(f"Invalid latitude: {lat0}")
        
        # Normalize azimuth to [0, 360)
        azimuth = azimuth % 360
        
        # Convert to radians
        lat0_rad = np.radians(lat0)
        lon0_rad = np.radians(lon0)
        azimuth_rad = np.radians(azimuth)
        
        # CRITICAL FIX: Ensure distances array starts exactly at 0
        distances = np.linspace(0.0, np.radians(max_distance_deg), num_points)
        
        # Calculate great circle points using spherical trigonometry
        # Using the standard great circle formulas
        lats_rad = np.arcsin(
            np.sin(lat0_rad) * np.cos(distances) + 
            np.cos(lat0_rad) * np.sin(distances) * np.cos(azimuth_rad)
        )
        
        # Calculate longitude differences
        dlon = np.arctan2(
            np.sin(azimuth_rad) * np.sin(distances) * np.cos(lat0_rad),
            np.cos(distances) - np.sin(lat0_rad) * np.sin(lats_rad)
        )
        
        lons_rad = lon0_rad + dlon
        
        # Convert back to degrees
        lats_deg = np.degrees(lats_rad)
        lons_deg = np.degrees(lons_rad)
        
        # Normalize longitude to [-180, 180]
        lons_deg = ((lons_deg + 180) % 360) - 180
        
        # CRITICAL VERIFICATION: First point must exactly match input
        if debug:
            lat_error = abs(lats_deg[0] - lat0)
            lon_error = abs(lons_deg[0] - lon0)
            print(f"    First point check: lat_error={lat_error:.2e}, lon_error={lon_error:.2e}")
            if lat_error > 1e-10 or lon_error > 1e-10:
                print(f"    ⚠ WARNING: First point doesn't match input!")
                print(f"    Input:  ({lat0:.10f}, {lon0:.10f})")
                print(f"    Output: ({lats_deg[0]:.10f}, {lons_deg[0]:.10f})")
        
        # FORCE exact match for first point to eliminate numerical errors
        lats_deg[0] = lat0
        lons_deg[0] = lon0
        
        return lons_deg, lats_deg

    def _plot_great_circle_basic(ax, lat0, lon0, azimuth, color, label, projection, linestyle='-'):
        """Plot great circle for basic matplotlib (non-cartopy) plots"""
        import numpy as np
        
        try:
            if projection == 'orthographic':
                gc_lats, gc_lons = _great_circle_path_2d(lat0, lon0, azimuth, max_distance_deg=90)
                x_coords, y_coords = _project_to_sphere(gc_lats, gc_lons, lat0, lon0)
                
                # Only plot points that are on the visible hemisphere
                visible = (x_coords**2 + y_coords**2 <= 1) & ~np.isnan(x_coords) & ~np.isnan(y_coords)
                if np.any(visible):
                    ax.plot(x_coords[visible], y_coords[visible], color=color, linewidth=3, 
                        label=label, alpha=0.8, linestyle=linestyle)
                    if debug:
                        print(f"    Plotted {np.sum(visible)}/{len(visible)} visible points on sphere")
            else:
                gc_lats, gc_lons = _great_circle_path_2d(lat0, lon0, azimuth)
                ax.plot(gc_lons, gc_lats, color=color, linewidth=3, 
                    label=label, alpha=0.8, linestyle=linestyle)
                if debug:
                    print(f"    Plotted {len(gc_lats)} points on flat projection")
        except Exception as e:
            print(f"ERROR in _plot_great_circle_basic: {e}")

    def _project_to_sphere(lat, lon, center_lat, center_lon):
        """FIXED: Project lat/lon to sphere coordinates for orthographic-like view"""
        import numpy as np
        
        # Convert to arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        center_lat_rad = np.radians(center_lat)
        center_lon_rad = np.radians(center_lon)
        
        # Calculate angular distance from center
        cos_c = (np.sin(center_lat_rad) * np.sin(lat_rad) + 
                 np.cos(center_lat_rad) * np.cos(lat_rad) * np.cos(lon_rad - center_lon_rad))
        
        # Orthographic projection (only visible hemisphere)
        x = np.full_like(lat, np.nan, dtype=float)
        y = np.full_like(lat, np.nan, dtype=float)
        
        # Only project points on visible hemisphere (cos_c >= 0)
        visible = cos_c >= 0
        
        if np.any(visible):
            x[visible] = np.cos(lat_rad[visible]) * np.sin(lon_rad[visible] - center_lon_rad)
            y[visible] = (np.cos(center_lat_rad) * np.sin(lat_rad[visible]) - 
                          np.sin(center_lat_rad) * np.cos(lat_rad[visible]) * 
                          np.cos(lon_rad[visible] - center_lon_rad))
        
        return x, y

    def _create_map_subplot(fig, gridspec, projection, station_coords=None, event_info=None):
        """Create map subplot with appropriate projection"""
        try:
            import cartopy.crs as ccrs
            import numpy as np
            
            if projection == 'orthographic':
                # Calculate optimal center point
                center_lon = station_coords.get('longitude', 0) if station_coords else 0
                center_lat = station_coords.get('latitude', 0) if station_coords else 0
                
                # Ensure coordinates are valid numbers
                if not (np.isfinite(center_lon) and np.isfinite(center_lat)):
                    center_lon, center_lat = 0, 0
                
                if event_info and 'latitude' in event_info and 'longitude' in event_info:
                    # Validate event coordinates
                    event_lat = event_info['latitude']
                    event_lon = event_info['longitude']
                    
                    if not (np.isfinite(event_lat) and np.isfinite(event_lon)):
                        # If event coordinates are invalid, center on station
                        proj = ccrs.Orthographic(center_lon, center_lat)
                        ax = fig.add_subplot(gridspec, projection=proj)
                        return ax
                    
                    # Normalize longitudes to [-180, 180]
                    event_lon = ((event_lon + 180) % 360) - 180
                    station_lon = ((center_lon + 180) % 360) - 180
                    
                    # Convert to radians for spherical geometry calculation
                    lat1, lon1 = np.radians(center_lat), np.radians(station_lon)
                    lat2, lon2 = np.radians(event_lat), np.radians(event_lon)
                    
                    try:
                        # Calculate midpoint using spherical geometry
                        Bx = np.cos(lat2) * np.cos(lon2 - lon1)
                        By = np.cos(lat2) * np.sin(lon2 - lon1)
                        
                        # Calculate midpoint
                        center_lat = np.degrees(np.arctan2(np.sin(lat1) + np.sin(lat2),
                                                        np.sqrt((np.cos(lat1) + Bx)**2 + By**2)))
                        
                        # Calculate central meridian that contains both points
                        dlon = lon2 - lon1
                        if abs(dlon) > np.pi:
                            dlon = -(2*np.pi - abs(dlon)) * np.sign(dlon)
                            
                        center_lon = np.degrees(lon1 + dlon/2)
                        center_lon = ((center_lon + 180) % 360) - 180
                        
                        # Calculate angular distance between points
                        angular_dist = np.degrees(np.arccos(np.sin(lat1) * np.sin(lat2) + 
                                                          np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)))
                        
                        # Add rotation to ensure both points are visible
                        # Rotation angle depends on angular distance between points
                        rotation_angle = min(30, max(10, angular_dist / 4))  # Scale rotation with distance
                        center_lon = center_lon + rotation_angle * np.sign(dlon)
                        
                    except (ValueError, RuntimeWarning) as e:
                        # If calculations fail, use simple midpoint
                        center_lat = (center_lat + event_lat) / 2
                        center_lon = (station_lon + event_lon) / 2
                    
                    # Ensure final center coordinates are valid
                    if not (np.isfinite(center_lon) and np.isfinite(center_lat)):
                        center_lon, center_lat = 0, 0
                    
                    # Create the projection with the calculated center
                    proj = ccrs.Orthographic(center_lon, center_lat)
                    ax = fig.add_subplot(gridspec, projection=proj)
                    
                    # Set map bounds to ensure visibility
                    if event_info and 'latitude' in event_info and 'longitude' in event_info:
                        try:
                            # Add padding around the points
                            padding = max(20, min(90, angular_dist / 2))  # Dynamic padding with limits
                            
                            # Ensure all values are finite
                            bounds = [
                                min(station_lon, event_lon) - padding,
                                max(station_lon, event_lon) + padding,
                                min(center_lat, event_info['latitude']) - padding,
                                max(center_lat, event_info['latitude']) + padding
                            ]
                            
                            if all(np.isfinite(b) for b in bounds):
                                ax.set_extent(bounds, crs=ccrs.PlateCarree())
                        except (ValueError, RuntimeWarning):
                            # If setting extent fails, let cartopy handle the bounds
                            pass
                    
                else:
                    ax = fig.add_subplot(gridspec, projection=ccrs.PlateCarree())
                    
                return ax
        except ImportError:
            return fig.add_subplot(gridspec)

    # Rest of the plot_backazimuth_map function remains the same...
    if not results or 'detailed_results' not in results:
        print("No results to plot")
        return None
    detailed_results = results['detailed_results']
    baz_estimates = results.get('estimates', {})
    station_coords = results.get('station_coordinates', {})
    
    num_wave_types = len(detailed_results)
    if num_wave_types == 0:
        print("No wave type results to plot")
        return None
    
    # Auto-determine figure size if not provided
    if figsize is None:
        if num_wave_types == 1:
            figsize = (16, 8)
        else:
            figsize = (16, 10)
    
    # Create figure layout
    fig = plt.figure(figsize=figsize)
    
    if num_wave_types == 1:
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[0, 0], map_projection, station_coords, event_info)
        ax_hist = fig.add_subplot(gs[0, 1])
        hist_axes = [ax_hist]
    elif num_wave_types == 2:
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[:, 0], map_projection, station_coords, event_info)
        ax_hist1 = fig.add_subplot(gs[0, 1])
        ax_hist2 = fig.add_subplot(gs[1, 1])
        hist_axes = [ax_hist1, ax_hist2]
    else:  # 3 wave types
        gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax_map = _create_map_subplot(fig, gs[:, 0], map_projection, station_coords, event_info)
        ax_hist1 = fig.add_subplot(gs[0, 1])
        ax_hist2 = fig.add_subplot(gs[1, 1])
        ax_hist3 = fig.add_subplot(gs[2, 1])
        hist_axes = [ax_hist1, ax_hist2, ax_hist3]
    
    # Plot the map
    if event_info:
        _plot_spherical_map_backazimuth(
            ax_map, event_info, baz_estimates, 
            station_coords.get('latitude', 0),
            station_coords.get('longitude', 0),
            map_projection
        )
    
    # Plot histograms
    colors = {'love': 'blue', 'rayleigh': 'red', 'tangent': 'purple'}
    angles = np.arange(0, 361, bin_step)
    angle_fine = np.arange(0, 360, 1)
    
    wave_types_list = list(detailed_results.keys())
    
    for i, wave_type in enumerate(wave_types_list):
        data = detailed_results[wave_type]
        baz = data['baz']
        cc = data['cc']
        
        ax = hist_axes[i]
        color = colors.get(wave_type, f'C{i}')
        
        # Compute statistics
        # baz_mean = np.average(baz, weights=cc)
        # baz_std = np.sqrt(np.average((baz - baz_mean)**2, weights=cc))
        # baz_max = baz_estimates.get(wave_type, baz_mean)

        # Plot histogram
        counts, _ = np.histogram(baz, bins=angles, density=True)
        bin_centers = (angles[:-1] + angles[1:]) / 2
        ax.bar(bin_centers, counts, width=bin_step*0.8, 
            alpha=0.7, color=color, edgecolor='black', linewidth=0.5, 
            label=f'N={len(baz)}')
        
        # Plot KDE overlay
        if len(baz) > 1:
            kde = sts.gaussian_kde(baz, weights=cc)
            kde_values = kde.pdf(angle_fine)
            ax.plot(angle_fine, kde_values, color='black', linewidth=2, label='KDE')
        
        # Compute the maximum of the KDE and index
        kde_max = kde_values.max()
        kde_max_idx = np.argmax(kde_values)
        baz_max = kde_max_idx

        # Mark estimated maximum
        ax.plot([kde_max_idx, kde_max_idx], [0, kde_max], color='black', linestyle='--', linewidth=2, 
                label=f'Est: {kde_max_idx:.0f}°')
        
        # Mark theoretical BAZ if available
        if event_info and 'backazimuth' in event_info:
            ax.axvline(event_info['backazimuth'], color='green', 
                    linestyle=':', linewidth=3, label=f"Theo: {event_info['backazimuth']:.0f}°")
            
            # Calculate deviation
            dev = abs(kde_max_idx - event_info['backazimuth'])
            if dev > 180:
                dev = 360 - dev
            
            # Add statistics text
            stats_text = (f"Max: {kde_max_idx}°\n"
                          f"Deviation: {round(dev, 0)}°")
        else:
            stats_text = (f"Max: {kde_max_idx}°")
        
        # Add statistics text box
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #         verticalalignment='top', fontsize=10,
        #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Configure histogram axis
        ax.set_title(f'{wave_type.upper()} Wave Backazimuth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Remove 0.00 tick label from density axis
        yticks = ax.get_yticks()
        yticks_filtered = yticks[yticks > 0.001]
        if len(yticks_filtered) > 0:
            ax.set_yticks(yticks_filtered)
    ax.set_xlabel('Backazimuth (°)')
    
    # Overall title
    title = f"Analysis"
    try:
        title += f" | T = {results['parameters']['baz_win_sec']}s ({results['parameters']['baz_win_sec_overlap']*100:.0f}%)"
        title += f" | CC > {results['parameters']['cc_threshold']}"
    except:
        pass
    
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    return fig

def print_dict_tree(d, indent=0, prefix=""):
    """
    Print a dictionary's keys in a tree-like structure.
    
    Args:
        d (dict): The dictionary to display
        indent (int): Current indentation level
        prefix (str): Prefix for the current line
    """
    for i, (key, value) in enumerate(d.items()):
        is_last = i == len(d) - 1
        current_prefix = "└── " if is_last else "├── "
        print(" " * indent + prefix + current_prefix + str(key))
        
        if isinstance(value, dict):
            next_prefix = "    " if is_last else "│   "
            print_dict_tree(value, indent + 4, prefix + next_prefix)

def compute_frequency_backazimuth_adaptive(sd_object, wave_type='love', fmin=0.01, fmax=0.5, 
                                          octave_fraction=3, baz_step=1, 
                                          window_factor=1.0, overlap_fraction=0.5,
                                          baz_win_sec_overlap=0.5, verbose=True,
                                          cc_threshold=None):
    """
    Compute backazimuth for octave frequency bands with adaptive time windows (1/fc)
    
    Parameters:
    -----------
    sd_object : sixdegrees object
    wave_type : str
        'love', 'rayleigh', or 'tangent'
    fmin, fmax : float
        Frequency range in Hz
    octave_fraction : int
        Octave fraction (3 for 1/3 octave)
    window_factor : float
        Multiplier for 1/fc to determine time window length (default: 1.0)
    overlap_fraction : float
        Overlap fraction between time windows (0-1)
    cc_threshold : float, optional
        Correlation coefficient threshold for filtering results
    Other parameters passed to compute_backazimuth
    
    Returns:
    --------
    dict : Results with frequency bands, adaptive time windows, correlations, backazimuths,
           and statistical estimates including histogram, KDE, and uncertainty
    """
    import numpy as np
    import gc
    from acoustics.octave import Octave
    from obspy import Stream
    
    if verbose:
        print(f"Computing {wave_type} backazimuth with adaptive time windows (factor={window_factor})...")
    
    # Generate octave bands
    octave = Octave(fraction=octave_fraction, fmin=fmin, fmax=fmax)
    center_freqs = octave.center
    lower_freqs = octave.lower
    upper_freqs = octave.upper
    
    # Store original stream
    original_stream = sd_object.get_stream('all', raw=True).copy()
    total_duration = original_stream[0].stats.endtime - original_stream[0].stats.starttime
    
    # Initialize results
    results = {
        'frequency_bands': center_freqs,
        'frequency_lower': lower_freqs,
        'frequency_upper': upper_freqs,
        'backazimuth_data': [],
        'correlation_data': [],
        'time_windows': [],  # Will be list of arrays, one per frequency
        'adaptive_windows': True,
        'window_factor': window_factor,
    }
    
    # Process each frequency band with adaptive time windows
    for i, (fl, fu, fc) in enumerate(zip(lower_freqs, upper_freqs, center_freqs)):
        # Calculate adaptive time window length
        time_window_sec = max(int(window_factor / fc), 1)
   
        if verbose:
            print(f"  Processing {fc:.3f} Hz ({fl:.3f}-{fu:.3f} Hz), window={time_window_sec:.1f}s")
        
        try:
            # Filter data for this frequency band
            filtered_stream = original_stream.copy()
            filtered_stream.filter('bandpass', freqmin=fl, freqmax=fu, corners=4, zerophase=True)
            
            # Temporarily replace stream in sd_object
            sd_object.st = filtered_stream
            
            # Compute backazimuth for this frequency band with adaptive window
            results_baz = sd_object.compute_backazimuth(
                wave_type=wave_type,
                baz_step=baz_step,
                baz_win_sec=time_window_sec,
                baz_win_sec_overlap=overlap_fraction,
                verbose=False,
                out=True
            )
 
            if results_baz and 'cc_max_y' in results_baz:
                # Store time windows for this frequency
                results['time_windows'].append(results_baz['cc_max_t'])
                
                # Store backazimuth and correlation data
                results['backazimuth_data'].append(results_baz['cc_max_y'])
                results['correlation_data'].append(results_baz['cc_max'])
            else:
                # Fill with NaN if no results
                # Create dummy time windows based on expected length
                n_windows = max(1, int(total_duration / time_window_sec * (1 - overlap_fraction) + 1))
                dummy_times = np.linspace(time_window_sec/2, total_duration - time_window_sec/2, n_windows)
                
                results['time_windows'].append(dummy_times)
                results['backazimuth_data'].append(np.full(len(dummy_times), np.nan))
                results['correlation_data'].append(np.full(len(dummy_times), np.nan))
                
        except Exception as e:
            if verbose:
                print(f"    Error processing {fc:.3f} Hz: {e}")
            
            # Create dummy data for failed processing
            time_window_sec = window_factor / fc
            n_windows = max(1, int(total_duration / time_window_sec * (1 - overlap_fraction) + 1))
            dummy_times = np.linspace(time_window_sec/2, total_duration - time_window_sec/2, n_windows)
            
            results['time_windows'].append(dummy_times)
            results['backazimuth_data'].append(np.full(len(dummy_times), np.nan))
            results['correlation_data'].append(np.full(len(dummy_times), np.nan))
        
        finally:
            # Restore original stream
            sd_object.st = original_stream
        
        gc.collect()
    
    if verbose:
        total_points = sum(len(baz_data) for baz_data in results['backazimuth_data'])
        valid_points = sum(np.sum(~np.isnan(baz_data)) for baz_data in results['backazimuth_data'])
        coverage = valid_points / total_points * 100 if total_points > 0 else 0
        print(f"Completed: {coverage:.1f}% coverage ({valid_points}/{total_points} points)")

    # Compute statistical estimates
    if verbose:
        print("Computing statistical estimates...")
    
    # Flatten all backazimuth and correlation data
    all_baz = np.concatenate(results['backazimuth_data'])
    all_cc = np.concatenate(results['correlation_data'])

    # Apply correlation threshold if specified
    if cc_threshold is not None:
        mask = all_cc >= cc_threshold
        all_baz = all_baz[mask]
        all_cc = all_cc[mask]

    # Remove NaN values
    valid_mask = ~np.isnan(all_baz)
    all_baz = all_baz[valid_mask]
    all_cc = all_cc[valid_mask]

    if len(all_baz) > 5:

        # get kde stats for backazimuth
        kde_stats = get_kde_stats(all_baz, all_cc, _baz_steps=0.5, Ndegree=60)
        baz_estimate = kde_stats['baz_estimate']
        baz_std = kde_stats['kde_dev']


        # Store statistical results
        results.update({
            'baz_estimate': baz_estimate,
            'baz_std': baz_std,
            'n_measurements': len(all_baz),
        })
        
        if verbose:
            print(f"Estimated backazimuth: {baz_estimate:.1f}° ± {baz_std:.1f}°")
            print(f"Based on {len(all_baz)} measurements with mean CC: {np.mean(all_cc):.3f}")
    else:
        if verbose:
            print("Warning: No valid measurements for statistical estimation")
        results.update({
            'baz_estimate': np.nan,
            'baz_std': np.nan,
            'n_measurements': 0,
        })
    
    gc.collect()
    return results

def plot_frequency_time_map_adaptive(results, plot_type='backazimuth', event_info=None, 
                                     figsize=(12, 8), vmin=None, vmax=None):
    """
    Plot frequency vs time map for adaptive time windows using grid-based approach
    
    Parameters:
    -----------
    results : dict
        Results from compute_frequency_backazimuth_adaptive
    plot_type : str
        'backazimuth' (shows deviation from theoretical) or 'correlation'
    event_info : dict, optional
        Event info with theoretical 'backazimuth' for comparison
    figsize : tuple
        Figure size
    vmin, vmax : float, optional
        Color scale limits
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if this is adaptive window data
    if not results.get('adaptive_windows', False):
        print("Warning: This function is designed for adaptive window results")
    
    freq_bands = results['frequency_bands']
    
    # Find the smallest time window to determine grid resolution
    min_window_length = float('inf')
    all_times = []
    
    for freq_idx, freq in enumerate(freq_bands):
        time_windows = results['time_windows'][freq_idx]
        all_times.extend(time_windows)
        
        # Calculate window length for this frequency
        window_length = results['window_factor'] / freq
        min_window_length = min(min_window_length, window_length)
    
    # Create time grid based on smallest window
    all_times = np.array(all_times)
    time_min, time_max = np.min(all_times), np.max(all_times)
    
    # Grid resolution: use half the minimum window length for fine resolution
    grid_time_step = min_window_length / 2
    n_time_bins = int((time_max - time_min) / grid_time_step) + 1
    time_grid = np.linspace(time_min, time_max, n_time_bins)
    
    # Create frequency grid (log scale)
    freq_grid = freq_bands
    
    # Create meshgrid
    TIME_GRID, FREQ_GRID = np.meshgrid(time_grid, freq_grid)
    
    # Initialize data grid with NaN
    data_grid = np.full(TIME_GRID.shape, np.nan)
    
    # Fill grid with data from each frequency band
    for freq_idx, freq in enumerate(freq_bands):
        time_windows = results['time_windows'][freq_idx]
        
        if plot_type == 'backazimuth':
            baz_data = results['backazimuth_data'][freq_idx]
            
            if event_info and 'backazimuth' in event_info:
                theoretical_baz = event_info['backazimuth']
                # Calculate deviation
                deviation = baz_data - theoretical_baz
                deviation = ((deviation + 180) % 360) - 180
                values = deviation
            else:
                values = baz_data
        else:  # correlation
            values = results['correlation_data'][freq_idx]
        
        # Ensure arrays have same length
        min_length = min(len(time_windows), len(values))
        time_windows_trimmed = time_windows[:min_length]
        values_trimmed = values[:min_length]
        
        # Calculate window length for this frequency
        window_length = results['window_factor'] / freq
        half_window = window_length / 2
        
        # Fill grid cells for each time window
        for t_center, value in zip(time_windows_trimmed, values_trimmed):
            if np.isnan(value):
                continue
                
            # Find time range for this window
            t_start = t_center - half_window
            t_end = t_center + half_window
            
            # Find grid indices that fall within this time window
            time_mask = (time_grid >= t_start) & (time_grid <= t_end)
            time_indices = np.where(time_mask)[0]
            
            # Fill all grid cells within this time window
            for t_idx in time_indices:
                data_grid[freq_idx, t_idx] = value
    
    # Set up plot parameters
    if plot_type == 'backazimuth':
        if event_info and 'backazimuth' in event_info:
            label = 'Backazimuth Deviation (°)'
            cmap = 'RdBu_r'
            if vmin is None and vmax is None:
                valid_data = data_grid[~np.isnan(data_grid)]
                if len(valid_data) > 0:
                    max_abs_dev = np.max(np.abs(valid_data))
                    vmin, vmax = -max_abs_dev, max_abs_dev
                else:
                    vmin, vmax = -10, 10
        else:
            label = 'Backazimuth (°)'
            cmap = 'hsv'
            if vmin is None: vmin = 0
            if vmax is None: vmax = 360
    else:
        label = 'Cross-Correlation'
        cmap = 'viridis'
        if vmin is None: vmin = 0
        if vmax is None: 
            valid_data = data_grid[~np.isnan(data_grid)]
            vmax = np.max(valid_data) if len(valid_data) > 0 else 1.0
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap using pcolormesh
    im = ax.pcolormesh(time_grid, freq_bands, data_grid, 
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=label, pad=0.02)
    
    # Show adaptive window boundaries
    try:
        for freq_idx, freq in enumerate(freq_bands):
            time_windows = results['time_windows'][freq_idx]
            window_length = results['window_factor'] / freq
            
            # Plot window boundaries as horizontal lines
            # for t in time_windows:
            #     # Draw window extent
            #     ax.plot([t - window_length/2, t + window_length/2], [freq, freq], 
            #            'k-', alpha=0.4, linewidth=1)
            #     # Mark window center
            #     ax.plot([t, t], [freq * 0.95, freq * 1.05], 'k-', alpha=0.6, linewidth=1)
    except Exception as e:
        print(f"Warning: Could not plot window boundaries: {e}")
    
    # Formatting
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_yscale('log')
    
    title = f'Frequency-Time Map: {plot_type.title()} (Adaptive Windows: {results["window_factor"]:.1f}/fc)'
    if plot_type == 'backazimuth' and event_info and 'backazimuth' in event_info:
        title += f'\n(Theoretical BAZ: {event_info["backazimuth"]:.1f}°)'
    
    ax.set_title(title, fontsize=14)
    # ax.grid(True, alpha=0.3)
    
    # Calculate statistics
    all_valid_values = data_grid[~np.isnan(data_grid)]

    
    plt.tight_layout()
    return fig

def animate_waveforms(sd, time_step: float = 0.5, duration: float = None,
                     save_path: str = None, dpi: int = 150, show_arrivals: bool = False,
                     rotate_zrt: bool = False, tail_duration: float = 50.0, baz: float = None,
                     n_frames: int = None):
    """
    Create an animation of waveforms and particle motion.
    
    Parameters:
    -----------
    sd : sixdegrees object
        The sixdegrees object containing the waveform data
    time_step : float
        Time step between frames in seconds (default: 0.5)
    duration : float, optional
        Duration of the animation in seconds. If None, uses full stream length
    save_path : str, optional
        Path to save the animation (e.g., 'animation.mp4'). If None, displays animation
    dpi : int
        DPI for the saved animation (default: 150)
    show_arrivals : bool
        Whether to show theoretical P and S wave arrival times (default: False)
    rotate_zrt : bool
        Whether to rotate horizontal components to radial and transverse (default: False)
    tail_duration : float
        Duration of particle motion tail in seconds (default: 5.0)
    baz : float, optional
        Backazimuth in degrees. If provided, overrides event-based backazimuth
    n_frames : int, optional
        Number of frames for the animation. If provided, adjusts duration accordingly
    
    Returns:
    --------
    anim : matplotlib.animation.FuncAnimation
        The animation object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from obspy import Stream, UTCDateTime
    import matplotlib.gridspec as gridspec
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    from obspy.signal.rotate import rotate_ne_rt

    def normalize_trace(data):
        """Normalize trace data to [-1, 1]."""
        max_val = np.max(np.abs(data))
        if max_val > 0:
            return data / max_val
        return data

    # Get streams
    trans_st = sd.get_stream(stream_type="translation")
    rot_st = sd.get_stream(stream_type="rotation")
    
    # Check if streams are empty
    if not trans_st or not rot_st:
        raise ValueError("Translation or rotation stream is empty")
    
    # define components
    components = ['Z', 'N', 'E']

    # define time delta
    dt = trans_st[0].stats.delta

    #define phases
    phases = ['P', 'S']

    # Get sampling rate and adjust time_step if needed
    sampling_rate = trans_st[0].stats.sampling_rate
    min_time_step = 1.0 / sampling_rate
    
    if time_step < min_time_step:
        print(f"Warning: time_step {time_step}s is smaller than minimum {min_time_step}s. Adjusting.")
        time_step = min_time_step

    # Rotate to ZRT if requested
    if rotate_zrt:
        try:
            # Use provided backazimuth or get from event
            if baz is None:
                event_info = sd.get_event_info(sd.get_stream()[0].stats.starttime, 
                                            base_catalog="USGS",
                                            )
                baz = event_info.get('backazimuth')

            if baz is not None:
                print(f"Using backazimuth: {baz}°")

                HRdata, HTdata = rotate_ne_rt(
                    trans_st.select(component='N')[0].data,
                    trans_st.select(component='E')[0].data,
                    baz
                )
                JRdata, JTdata = rotate_ne_rt(
                    rot_st.select(component='N')[0].data,
                    rot_st.select(component='E')[0].data,
                    baz
                )

                HZ = trans_st.select(component='Z')[0]
                JZ = rot_st.select(component='Z')[0]

                # set R for N
                JR = rot_st.select(component='N')[0].copy()
                JR.data = JRdata
                JR.stats.channel = JR.stats.channel[:-1] + 'R'

                HR = trans_st.select(component='N')[0].copy()
                HR.data = HRdata
                HR.stats.channel = HR.stats.channel[:-1] + 'R'

                # set T for E
                JT = rot_st.select(component='E')[0].copy()
                JT.data = JTdata
                JT.stats.channel = JT.stats.channel[:-1] + 'T'

                HT = rot_st.select(component='E')[0].copy()
                HT.data = HTdata
                HT.stats.channel = HT.stats.channel[:-1] + 'T'

                trans_st = Stream([HZ, HR, HT])
                rot_st = Stream([JZ, JR, JT])

                components = ['Z', 'R', 'T']
            else:
                print("Warning: Could not get backazimuth, using ZNE components")
        except Exception as e:
            print(f"Warning: Error rotating to ZRT: {e}, using ZNE components")

    print(rot_st, trans_st)

    # Verify all components exist in both streams
    for comp in components:
        if not trans_st.select(component=comp) or not rot_st.select(component=comp):
            raise ValueError(f"Component {comp} not found in both streams")
    
    # Get reference time array and calculate duration
    ref_trace = trans_st.select(component=components[0])[0]
    t_trans = ref_trace.times()
    total_duration = t_trans[-1]
    
    # Calculate time limits and frames
    if n_frames is not None:
        duration = n_frames * time_step
    elif duration is None:
        duration = total_duration
    else:
        duration = min(duration, total_duration)
    
    # Calculate number of frames based on duration and time_step
    n_frames = int(duration / time_step)
    
    # Adjust time_step to match exact duration
    time_step = duration / n_frames
    
    print(f"Animation parameters:")
    print(f"Duration: {duration:.2f}s")
    print(f"Time step: {time_step:.3f}s")
    print(f"Number of frames: {n_frames}")
    print(f"Frame rate: {1/time_step:.1f} fps")
    
    # Normalize all traces
    for comp in components:
        trans_st.select(component=comp)[0].data = normalize_trace(trans_st.select(component=comp)[0].data)
        rot_st.select(component=comp)[0].data = normalize_trace(rot_st.select(component=comp)[0].data)
    
    # Set up the figure with two rows
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5])  # Adjusted height ratio
    
    # First row: single panel for all waveforms
    ax_waves = fig.add_subplot(gs[0, :])
    
    # Second row: two panels for particle motion
    ax_love = fig.add_subplot(gs[1, 0])
    ax_rayleigh = fig.add_subplot(gs[1, 1])
    
    # Setup waveform plot with closer vertical offsets
    offsets = np.arange(6) * 1.5  # Reduced spacing from 2.5 to 1.5
    trace_pairs = list(zip(components * 2, ['Translation'] * 3 + ['Rotation'] * 3, offsets))
    
    # Initialize lines for each trace
    wave_lines_past = []
    wave_lines_future = []
    
    for comp, trace_type, offset in trace_pairs:
        tr = trans_st.select(component=comp)[0] if trace_type == 'Translation' else rot_st.select(component=comp)[0]
        color = 'k' if trace_type == 'Translation' else 'darkred'
        
        # Plot future data in grey
        line_future, = ax_waves.plot(t_trans, tr.data + offset, color='lightgrey', alpha=0.5)
        # Initialize past data line
        line_past, = ax_waves.plot([], [], color=color)
        
        # Add channel name as label on the left side
        ax_waves.text(-0.01, offset, tr.stats.channel, 
                     transform=ax_waves.get_yaxis_transform(),
                     verticalalignment='center',
                     horizontalalignment='right')
        
        wave_lines_past.append(line_past)
        wave_lines_future.append(line_future)
    
    ax_waves.set_ylim(-1, 9)  # Adjusted for closer spacing
    ax_waves.set_xlim(0, duration)
    ax_waves.set_xlabel('Time (s)')
    ax_waves.set_yticks([])  # Remove y-axis ticks
    
    # Remove top and right spines
    ax_waves.spines['top'].set_visible(False)
    ax_waves.spines['right'].set_visible(False)
    
    # Initialize particle motion plots
    ax_love.set_aspect('equal')
    ax_rayleigh.set_aspect('equal')
    
    # Set titles and labels
    ax_love.set_title('Love Wave Particle Motion')
    ax_rayleigh.set_title('Rayleigh Wave Particle Motion')
    
    ax_love.set_xlabel(f'HT')
    ax_rayleigh.set_xlabel(f'JT')
    ax_love.set_ylabel(f'JZ')
    ax_rayleigh.set_ylabel(f'HZ')

    # Set equal limits for particle motion plots based on normalized data
    pm_lim = 1.2  # slightly larger than normalized range (-1, 1)
    ax_love.set_xlim(-pm_lim, pm_lim)
    ax_love.set_ylim(-pm_lim, pm_lim)
    ax_rayleigh.set_xlim(-pm_lim, pm_lim)
    ax_rayleigh.set_ylim(-pm_lim, pm_lim)
    
    # Add grid to particle motion plots
    ax_love.grid(True, ls='--', zorder=0)
    ax_rayleigh.grid(True, ls='--', zorder=0)
    
    # Create red fade colormap for particle motion trails
    cmap = plt.cm.Blues

    # amount of trail samples
    tail_samples = int(tail_duration / dt)

    # Initialize particle motion lines and points
    love_trail = ax_love.scatter([], [], c=[], cmap=cmap, s=10, vmin=0, vmax=tail_samples, zorder=10)
    rayleigh_trail = ax_rayleigh.scatter([], [], c=[], cmap=cmap, s=10, vmin=0, vmax=tail_samples, zorder=10)
    love_point = ax_love.scatter([], [], color='darkblue', s=50, zorder=10)
    rayleigh_point = ax_rayleigh.scatter([], [], color='darkblue', s=50, zorder=10)
    
    # Add cursor line and trail region
    cursor_line = ax_waves.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
    
    # Define global trail region variable
    global trail_region
    # Create initial shaded region for trail duration (will be updated in animation)
    trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), 0, 0,
                                       color='lightblue', alpha=0.2, zorder=0)

    # Add P and S wave arrival lines if requested
    starttime = sd.get_stream()[0].stats.starttime
    if show_arrivals:
        try:
            p_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='P')) - starttime
        except:
            # replace P with Pdiff
            p_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='Pdiff')) - starttime
            phases = ['Pdiff', 'S']

        s_arrival = UTCDateTime(sd.get_theoretical_arrival(phase='S')) - starttime
        if p_arrival is not None:
            ax_waves.axvline(x=p_arrival, color='black', linestyle='-', alpha=0.5)
            ax_waves.text(p_arrival, ax_waves.get_ylim()[1], phases[0], 
                         horizontalalignment='right', verticalalignment='bottom')
        if s_arrival is not None:
            ax_waves.axvline(x=s_arrival, color='black', linestyle='-', alpha=0.5)
            ax_waves.text(s_arrival, ax_waves.get_ylim()[1], phases[1], 
                         horizontalalignment='right', verticalalignment='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    def init():
        """Initialize animation"""
        global trail_region
        for line in wave_lines_past:
            line.set_data([], [])
        love_trail.set_offsets(np.c_[[], []])
        rayleigh_trail.set_offsets(np.c_[[], []])
        love_point.set_offsets(np.c_[[], []])
        rayleigh_point.set_offsets(np.c_[[], []])
        trail_region.remove()  # Remove old region
        # Create new empty region
        trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), 0, 0,
                                           color='lightblue', alpha=0.2, zorder=0)
        return wave_lines_past + [love_trail, rayleigh_trail, love_point, rayleigh_point, cursor_line, trail_region]
    
    def animate(frame):
        """Animation function"""
        global trail_region
        current_time = frame * time_step
        
        # Ensure we don't exceed the data length
        current_time = min(current_time, duration)
        
        # Update cursor position
        cursor_line.set_xdata([current_time, current_time])
        
        # Update trail region by removing old and creating new
        trail_region.remove()
        start_time = max(0, current_time - tail_duration)
        trail_region = ax_waves.fill_betweenx(np.array([-1, 9]), start_time, current_time,
                                           color='lightblue', alpha=0.2, zorder=0)
        
        # Update waveform lines
        for i, (comp, trace_type, offset) in enumerate(trace_pairs):
            tr = trans_st.select(component=comp)[0] if trace_type == 'Translation' else rot_st.select(component=comp)[0]
            mask = t_trans <= current_time
            wave_lines_past[i].set_data(t_trans[mask], tr.data[mask] + offset)
        
        try:
            # Update particle motion plots
            tail_samples = int(tail_duration / dt)
            current_idx = min(int(current_time * sampling_rate), len(t_trans) - 1)
            start_idx = max(0, current_idx - tail_samples)
            
            if rotate_zrt:  # Only show particle motion in ZRT coordinates
                # Get relevant components for Love waves (HT and RZ)
                rz = rot_st.select(component='Z')[0].data[start_idx:current_idx+1]
                ht = trans_st.select(component='Z')[0].data[start_idx:current_idx+1]
                
                # Get relevant components for Rayleigh waves (RT and HZ)
                hz = trans_st.select(component='Z')[0].data[start_idx:current_idx+1]
                rt = rot_st.select(component='T')[0].data[start_idx:current_idx+1]
                
                # Create fade effect - scale from 0 to tail_samples
                n_points = current_idx + 1 - start_idx
                # Create fade values that increase from oldest to newest points
                fade_values = np.arange(n_points) if n_points > 0 else np.array([])
                
                # Update Love wave plot
                if len(ht) > 0:
                    love_trail.set_offsets(np.c_[ht, rz])
                    love_trail.set_array(fade_values)
                    love_point.set_offsets([[ht[-1], rz[-1]]])  # Current point is at the end
                
                # Update Rayleigh wave plot
                if len(rt) > 0:
                    rayleigh_trail.set_offsets(np.c_[rt, hz])
                    rayleigh_trail.set_array(fade_values)
                    rayleigh_point.set_offsets([[rt[-1], hz[-1]]])  # Current point is at the end
            
        except Exception as e:
            print(f"Warning: Error updating particle motion: {e}")
        
        return wave_lines_past + [love_trail, rayleigh_trail, love_point, rayleigh_point, cursor_line, trail_region]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=time_step*10, blit=True)
    
    # Save or display animation
    if save_path:
        anim.save(save_path, writer='ffmpeg', dpi=dpi)
        plt.close()
    else:
        plt.show()
    
    return anim 
