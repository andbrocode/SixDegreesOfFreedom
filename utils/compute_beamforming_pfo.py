"""
Functions for computing beamforming analysis at PFO.
"""
import os
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as sts
import warnings
from typing import Tuple, List, Optional
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from obspy.signal import array_analysis as AA
from obspy.signal.util import util_geo_km
from obspy.signal.rotate import rotate2zne
from obspy.core.util import AttribDict
from obspy.imaging.cm import obspy_sequential
from obspy.signal.array_analysis import array_processing
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

warnings.filterwarnings('ignore')

def compute_beamforming_pfo(tbeg, tend, submask, fmin: Optional[float] = None, fmax: Optional[float] = None, 
                          component: str = "", bandpass: bool = True, plot: bool = False) -> Tuple[dict, List[int]]:
    """
    Compute beamforming analysis for the PFO array.
    
    Args:
        tbeg (UTCDateTime): Start time
        tend (UTCDateTime): End time
        submask (str): Subarray configuration ('inner', 'mid', 'all')
        fmin (float, optional): Minimum frequency for analysis
        fmax (float, optional): Maximum frequency for analysis
        component (str): Component to analyze (Z, N, E)
        bandpass (bool): Whether to apply bandpass filter
        plot (bool): Whether to plot results
        
    Returns:
        dict: Dictionary containing beamforming results including:
            - t_win: Time windows
            - rel_pwr: Relative power
            - abs_pwr: Absolute power
            - baz: Backazimuth values
            - slow: Slowness values
            And additional statistics
    """
    def __get_data(config):
        """Helper function to get seismic data."""
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

            inventory = config['fdsn_client'].get_stations(
                network=net,
                station=sta,
                starttime=config['tbeg']-20,
                endtime=config['tend']+20,
                level="response"
            )

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
                print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...") if config['print_details'] else None
                continue

            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3") if config['print_details'] else None
                stats.merge(method=1, fill_value="interpolate")

            stats.remove_response(inventory=inventory, output="VEL")

            try:
                stats.rotate(method="->ZNE", inventory=inventory)
            except:
                print(" -> failed to rotate to ZNE")
                continue

            if config['reference_station'] == "PY.PFOIX":
                stats = stats.resample(40)
                stats = stats.trim(config['tbeg']-20, config['tend']+20)

            if station == config['reference_station']:
                ref_station = stats.copy()

            st += stats

        if len(st) < 3*len(config['subarray_stations']):
            config['subarray_stations'] = [f"{tr.stats.network}.{tr.stats.station}" for tr in st]
            config['subarray_stations'] = list(set(config['subarray_stations']))

        print(f" -> obtained: {int(len(st)/3)} of {len(config['subarray_stations'])} stations!")

        if len(st) == 0:
            return st, config
        else:
            return st, config

    def __add_coordinates(st, config):
        """Helper function to add coordinates to stream."""
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

            l_lon = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])

            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                l_lon, l_lat = -116.455439, 33.610643

            for c in ["Z", "N", "E"]:
                st.select(station=sta, channel=f"*{c}")[0].stats.coordinates = AttribDict({
                    'latitude': l_lat,
                    'elevation': height/1000,
                    'longitude': l_lon
                })

        return st

    # Set up configuration
    start_timer = timeit.default_timer()
    config = {}
    config['tbeg'] = UTCDateTime(tbeg)
    config['tend'] = UTCDateTime(tend)
    config['fdsn_client'] = Client('IRIS')

    # Configure subarray based on mask
    if submask is not None:
        if submask == "inner":
            config['subarray_mask'] = [0,1,2,3,4]
            config['freq1'] = 1.0
            config['freq2'] = 6.0
        elif submask == "mid":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8]
            config['freq1'] = 0.5
            config['freq2'] = 1.0
        elif submask == "all":
            config['subarray_mask'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
            config['freq1'] = 0.1
            config['freq2'] = 0.5
    else:
        config['subarray_mask'] = [0,1,2,3,4]

    config['print_details'] = False

    # Set reference station based on time
    if config['tbeg'] > UTCDateTime("2023-04-01"):
        config['reference_station'] = 'PY.PFOIX'
        config['array_stations'] = ['PY.PFOIX','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']
    else:
        config['reference_station'] = 'II.PFO'
        config['array_stations'] = ['II.PFO','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']

    config['misorientations'] = [0, 0., -1.375, 0.25, 0.125, -0.6875, -0.625, -1.9375, 0.375,
                              -6.5625, 0.3125, -1.125, -2.5625, 0.1875]

    config['subarray_misorientation'] = [config['misorientations'][i] for i in config['subarray_mask']]
    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]

    # Beamforming parameters
    config['slow_xmin'] = -0.5
    config['slow_xmax'] = 0.5
    config['slow_ymin'] = -0.5
    config['slow_ymax'] = 0.5
    config['slow_steps'] = 0.01

    config['win_length'] = 1/fmin if fmin else 1.0
    config['win_frac'] = 0.5

    config['freq_lower'] = fmin if fmin else config['freq1']
    config['freq_upper'] = fmax if fmax else config['freq2']
    config['prewhitening'] = 0

    # Get and process data
    st, config = __get_data(config)
    st = st.detrend("demean")

    if bandpass:
        st = st.taper(0.1)
        st = st.filter("bandpass", freqmin=config['freq_lower'], freqmax=config['freq_upper'],
                      corners=8, zerophase=True)

    st = __add_coordinates(st, config)
    st = st.select(channel=f"*{component}")
    st = st.trim(config['tbeg']-0.1, config['tend']+0.1)

    # Configure beamforming parameters
    kwargs = {
        'sll_x': config['slow_xmin'], 'slm_x': config['slow_xmax'],
        'sll_y': config['slow_ymin'], 'slm_y': config['slow_ymax'],
        'sl_s': config['slow_steps'],
        'win_len': config['win_length'], 'win_frac': config['win_frac'],
        'frqlow': config['freq_lower'], 'frqhigh': config['freq_upper'],
        'prewhiten': config['prewhitening'],
        'semb_thres': -1e9, 'vel_thres': -1e9,
        'timestamp': 'mlabday',
        'stime': config['tbeg'], 'etime': config['tend'],
    }

    # Perform beamforming
    out = array_processing(st, **kwargs)
    st = st.trim(config['tbeg'], config['tend'])

    # Process results
    baz = out[:, 3]
    baz[baz < 0.0] += 360

    # Compute statistics
    deltaa = 5
    angles = np.arange(0, 365, deltaa)
    baz_bf_no_nan = baz[~np.isnan(baz)]
    cc_bf_no_nan = out[:, 2][~np.isnan(out[:, 2])]
    hist = np.histogram(baz, bins=len(angles)-1, range=[min(angles), max(angles)],
                       weights=out[:, 2], density=False)
    baz_bf_mean = round(np.average(baz_bf_no_nan, weights=cc_bf_no_nan), 0)
    baz_bf_std = np.sqrt(np.cov(baz_bf_no_nan, aweights=cc_bf_no_nan))
    kde = sts.gaussian_kde(baz_bf_no_nan, weights=cc_bf_no_nan)
    baz_bf_max = angles[np.argmax(kde.pdf(angles))] + deltaa/2

    # Prepare output
    output = {
        't_win': out[:, 0],
        'rel_pwr': out[:, 1],
        'abs_pwr': out[:, 2],
        'baz': baz,
        'slow': out[:, 4],
        'baz_max_count': np.max(hist[0]),
        'baz_max': angles[np.argmax(hist[0])],
        'slw_max': np.mean(out[:, 4]),
        'baz_bf_mean': baz_bf_mean,
        'baz_bf_max': baz_bf_max,
        'baz_bf_std': baz_bf_std
    }

    if plot:
        # Create plots (implementation of plotting code would go here)
        pass

    stop_timer = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer - start_timer)/60,2)} minutes")

    return output
