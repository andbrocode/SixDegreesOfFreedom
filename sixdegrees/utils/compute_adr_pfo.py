"""
Functions for computing array-derived rotation at PFO.
"""
import os
import numpy as np
import timeit
import matplotlib.pyplot as plt
import matplotlib.colors
import warnings
from obspy import UTCDateTime, Stream, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from obspy.signal import array_analysis as AA
from obspy.signal.util import util_geo_km
from obspy.signal.rotate import rotate2zne

warnings.filterwarnings('ignore')

def compute_adr_pfo(tbeg, tend, submask=None, status=False):
    """
    Compute array-derived rotation (ADR) for the PFO array.
    
    Args:
        tbeg (UTCDateTime): Start time
        tend (UTCDateTime): End time
        submask (str, optional): Subarray configuration ('inner', 'mid', 'all')
        status (bool): Whether to plot status information
        
    Returns:
        Stream or tuple: If status=False returns rotational Stream, if True returns (Stream, status_array)
    """
    def __get_inventory_and_distances(config):
        """Helper function to get inventory and calculate distances."""
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
                inven = read_inventory(config['data_path']+f"BSPF/data/stationxml/{net}.{sta}.xml")
            except:
                inven = config['fdsn_client'].get_stations(network=net,
                                                       station=sta,
                                                       channel=cha,
                                                       location=loc,
                                                       starttime=config['tbeg'],
                                                       endtime=config['tend'],
                                                       level='response'
                                                      )

            l_lon = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['longitude'])
            l_lat = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['latitude'])
            height = float(inven.get_coordinates('%s.%s.%s.%sZ'%(net,sta,loc,cha[:2]))['elevation'])

            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                l_lon, l_lat = -116.455439, 33.610643

            if sta == "XPFO" or sta == "PFO" or sta == "PFOIX":
                o_lon, o_lat, o_height = l_lon, l_lat, height

            lon, lat = util_geo_km(o_lon, o_lat, l_lon, l_lat)

            coo.append([lon*1000, lat*1000, height-o_height])  # convert unit from km to m

        return inven, np.array(coo)

    def __check_samples_in_stream(st, config):
        """Helper function to check samples in stream."""
        for tr in st:
            if tr.stats.npts != config['samples']:
                print(f" -> removing {tr.stats.station} due to improper number of samples ({tr.stats.npts} not {config['samples']})")
                st.remove(tr)
        return st

    def __get_data(config):
        """Helper function to get seismic data."""
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

            try:
                try:
                    inventory = read_inventory(config['data_path']+f"BSPF/data/stationxml/{net}.{sta}.xml")
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

            if len(stats) > 3:
                print(f" -> merging stream. Length: {len(stats)} -> 3") if config['print_details'] else None
                stats.merge(method=1, fill_value="interpolate")

            stats.remove_response(inventory=inventory, output="VEL", water_level=60)

            try:
                stats = stats.rotate(method="->ZNE", inventory=inventory)
            except:
                print(f" -> {sta} failed to rotate to ZNE")
                continue

            stats = stats.resample(40, no_filter=False)

            if station == config['reference_station']:
                ref_station = stats.copy()

            st += stats
            config['subarray'].append(f"{stats[0].stats.network}.{stats[0].stats.station}")

        st = st.sort()
        config['subarray_stations'] = config['subarray']

        print(f" -> obtained: {len(st)/3} of {len(config['subarray_stations'])} stations!") if config['print_details'] else None

        if len(st) == 0:
            return st, Stream(), config
        else:
            return st, ref_station, config

    def __compute_ADR(tse, tsn, tsz, config, ref_station):
        """Helper function to compute array-derived rotation."""
        tse, tsn, tsz = np.array(tse), np.array(tsn), np.array(tsz)
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

    # Set up configuration
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
    if config['tbeg'] > UTCDateTime("2023-04-02"):
        config['reference_station'] = 'PY.PFOIX'
        config['array_stations'] = ['PY.PFOIX','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']
    else:
        config['reference_station'] = 'II.PFO'
        config['array_stations'] = ['II.PFO','PY.BPH01','PY.BPH02','PY.BPH03','PY.BPH04','PY.BPH05','PY.BPH06','PY.BPH07',
                                'PY.BPH08','PY.BPH09','PY.BPH10','PY.BPH11','PY.BPH12','PY.BPH13']

    config['subarray_stations'] = [config['array_stations'][i] for i in config['subarray_mask']]
    config['subarray_sta'] = config['subarray_stations']

    # ADR parameters
    config['apply_bandpass'] = True
    config['vp'] = 6200
    config['vs'] = 3700
    config['sigmau'] = 1e-7

    # Set data paths based on hostname
    hostname = os.uname().nodename
    if hostname == 'lighthouse':
        config['data_path'] = '/home/andbro/'
    elif hostname == 'kilauea':
        config['data_path'] = '/import/kilauea-data/'
    elif hostname == 'lin-ffb-01':
        config['data_path'] = '/import/kilauea-data/'

    # Get data and compute ADR
    start_timer = timeit.default_timer()
    config['stations_loaded'] = np.ones(len(config['subarray_stations']))
    st, ref_station, config = __get_data(config)

    if len(st) < 9:
        print(" -> not enough stations (< 3) for ADR computation!")
        return

    print(f" -> continue computing ADR for {int(len(st)/3)} of {len(config['subarray_mask'])} stations ...")

    inv, config['coo'] = __get_inventory_and_distances(config)
    st.detrend("demean")

    if config['apply_bandpass']:
        st.taper(0.01)
        st.filter('bandpass', freqmin=config['freq1'], freqmax=config['freq2'], corners=4, zerophase=True)
        print(f" -> bandpass: {config['freq1']} - {config['freq2']} Hz")

    # Prepare data arrays
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

    # Compute ADR
    rot = __compute_ADR(tse, tsn, tsz, config, ref_station)

    # Adjust timing
    tstart = [tr.stats.starttime - tbeg for tr in st]
    for tr in rot:
        tr.stats.starttime = tbeg + np.mean(tstart)

    rot = rot.trim(config['tbeg'], config['tend'])

    # Plot status if requested
    if status:
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        cmap = matplotlib.colors.ListedColormap(['darkred', 'green'])
        ax.pcolormesh(np.array([config['stations_loaded'], np.ones(len(config['stations_loaded']))*0.5]).T,
                     cmap=cmap, edgecolors="k", lw=0.5)
        ax.set_yticks(np.arange(0, len(config['subarray_sta']))+0.5, labels=config['subarray_sta'])
        ax.set_xticks([])
        ax.set_xlim(0, 1)
        plt.show()

    stop_timer = timeit.default_timer()
    print(f"\n -> Runtime: {round((stop_timer - start_timer)/60, 2)} minutes\n")

    if status:
        return rot, config['stations_loaded']
    else:
        return rot
