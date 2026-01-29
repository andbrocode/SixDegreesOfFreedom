'''
@package sixdegrees
@copyright:
    Andreas Brotzer (rotzleffel@tutanota.com)
@license:
    GNU General Public License, Version 3
    (http://www.gnu.org/licenses/gpl-3.0.html)
'''

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Tuple, Union, Optional, Any
from obspy import UTCDateTime, Stream, Inventory, read, read_inventory
from obspy.clients.filesystem.sds import Client
from obspy.clients.fdsn import Client as FDSNClient
from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
from obspy.signal.cross_correlation import correlate, xcorr_max
from obspy.signal.rotate import rotate_ne_rt
from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
from numpy import ones, arange, linspace, asarray, array, meshgrid, nan, shape, ndarray
from numpy import arctan, pi, linspace, cov, argsort, corrcoef, correlate, zeros
from numpy.linalg import eigh
from acoustics.octave import Octave
from numpy import isnan, interp
from obspy.signal.cross_correlation import correlate, xcorr_max
from numpy.typing import NDArray
from matplotlib.colors import LogNorm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from scipy.signal import hilbert


class sixdegrees():

    def __init__(self, conf: Dict={}):
        """
        Initialize sixdegrees object with configuration dictionary.

        Args:
            conf (Dict): Configuration dictionary containing parameters for data processing.
                        Defaults to empty dictionary if not provided.
        """
        # define configurations
        self.config = conf

        # predefine varibales
        self.data_source = ""
        self.fdsn_client_rot = "IRIS"
        self.fdsn_client_tra = "IRIS"
        self.tra_inv = None
        self.rot_inv = None
        self.rot_inv_file = None
        self.tra_inv_file = None
        self.fmin = None
        self.fmax = None
        # predefine results for backazimuth estimation
        self.baz_results = {}

        # predefine theoretical baz
        self.baz_theo = None
        
        # define data source (local SDS or online FDSN)
        self.data_source = conf.get('data_source', 'fdsn')  # 'sds' or 'fdsn'

        # get startime and convert to UTCDateTime
        self.tbeg = conf.get('tbeg', None)
        if self.tbeg is not None:
            self.tbeg = UTCDateTime(self.tbeg)

        # get endtime and convert to UTCDateTime
        self.tend = conf.get('tend', None)
        if self.tend is not None:
            self.tend = UTCDateTime(self.tend)

        # set verbose (print information)
        self.verbose = conf.get('verbose', False)

        # seed id of stream
        self.net, self.sta, self.loc, self.cha = conf.get('seed', "XX.XXXX..").split('.')

        # seed id of rotation stream
        self.rot_seed = conf.get('rot_seed', None)

        # seed id of translation stream
        self.tra_seed = conf.get('tra_seed', None)

        # station coordinates
        self.station_longitude = conf.get('station_lon', None)
        self.station_latitude = conf.get('station_lat', None)

        # define project name
        self.project = conf.get('project', "test")
        
        # define working directory
        self.workdir = conf.get('workdir', os.path.normpath("./"))

        # define directory for output data
        self.path_to_data_out = conf.get('path_to_data_out', os.path.normpath(os.path.join(self.workdir, "output")))

        # define directory for figure output
        self.path_to_figs_out = conf.get('path_to_figs_out', os.path.normpath(os.path.join(self.workdir, "figures")))

        # path to rotation station inventory
        self.rot_inv_file = conf.get('path_to_inv_rot', None)
        # path to translation station inventory
        self.tra_inv_file = conf.get('path_to_inv_tra', None)

        # path to SDS file structure for rotation data
        self.rot_sds = conf.get('path_to_sds_rot', None)
        # path to SDS file structure for translation data
        self.tra_sds = conf.get('path_to_sds_tra', None)

        # path to FDSN client for rotation data
        self.fdsn_client_rot = conf.get('fdsn_client_rot', None)
        # path to FDSN client for translation data
        self.fdsn_client_tra = conf.get('fdsn_client_tra', None)

        # path to mseed file if using direct file input
        self.mseed_file = conf.get('path_to_mseed_file', False)

        # rotate_zne
        self.rotate_zne = conf.get('rotate_zne', False)

        # remove_response_tra
        self.tra_remove_response = conf.get('tra_remove_response', False)

        # remove_response_rot
        self.rot_remove_response = conf.get('rot_remove_response', False)

        # output type for remove response
        self.tra_output = conf.get('tra_output', "ACC")

        # add dummy trace
        self.dummy_trace = conf.get('dummy_trace', False)

        # units
        self.runit = conf.get('runit', "rad/s")
        self.tunit = conf.get('tunit', r"m/s$^2$")
        
        # define mu symbol
        self.mu = "$\mu$"

        # polarity dictionary
        self.pol_applied = False
        self.pol_dict = {}

        # Add new attributes
        self.rot_components = None  # Components to rotate from (e.g., 'ZUV')
        self.rot_target = 'ZNE'     # Target components (e.g., 'ZNE')

        # Add ROMY rotation options
        self.use_romy_zne = conf.get('use_romy_zne', False)
        self.keep_z = conf.get('keep_z', True)

        # check attributes
        checks = self.check_attributes()

        if not checks['passed']:
            for note in checks['notes']:
                print(note)
            raise ValueError("Required attributes are not set. Please check the configuration.")

    # ____________________________________________________

    def check_attributes(self):
        """
        Check if all required attributes are set
        """
        checks = {'passed': True, 'notes': []}
        if self.data_source == 'sds':
            if self.rot_sds is None:
                checks['notes'].append("WARNING: no path to SDS file structure for rotation data given!")
                checks['passed'] = False
            if self.tra_sds is None:
                checks['notes'].append("WARNING: no path to SDS file structure for translation data given!")
                checks['passed'] = False
        elif self.data_source == 'fdsn':
            if self.fdsn_client_rot is None:
                checks['notes'].append("WARNING: no FDSN client for rotation data given!")
                checks['passed'] = False
            if self.fdsn_client_tra is None:
                checks['notes'].append("WARNING: no FDSN client for translation data given!")
                checks['passed'] = False
        elif self.data_source == 'mseed_file':
            if self.mseed_file is None:
                checks['notes'].append("WARNING: no path to mseed file given!")
                checks['passed'] = False
        if self.data_source != 'fdsn':
            if self.station_longitude is None:
                checks['notes'].append("WARNING: no station longitude given!")
                checks['passed'] = False
            if self.station_latitude is None:
                checks['notes'].append("WARNING: no station latitude given!")
                checks['passed'] = False
        if self.rot_seed is None:
            checks['notes'].append("WARNING: no rotation seed id given!")
            checks['passed'] = False
        if self.tra_seed is None:
            checks['notes'].append("WARNING: no translation seed id given!")
            checks['passed'] = False
        if self.rot_inv_file is None:
            checks['notes'].append("INFO: no path to rotation station inventory given!")
        if self.tra_inv_file is None:
            checks['notes'].append("INFO: no path to translation station inventory given!")
        if self.tbeg is None:
            checks['notes'].append("WARNING: no starttime given!")
            checks['passed'] = False
        if self.tend is None:
            checks['notes'].append("WARNING: no endtime given!")
            checks['passed'] = False

        return checks

    def copy(self):
        """
        Create a deep copy of the sixdegrees object.
        
        Returns:
            sixdegrees: A new sixdegrees instance with copied attributes
        """
        import copy
        
        # Create a new instance with the same configuration
        new_instance = sixdegrees(self.config)
        
        # Copy all attributes that may have been modified during the object's lifecycle
        attributes_to_copy = [
            'data_source', 'fdsn_client_rot', 'fdsn_client_tra', 'tra_inv', 'rot_inv',
            'rot_inv_file', 'tra_inv_file', 'fmin', 'fmax', 'baz_results', 'baz_theo',
            'tbeg', 'tend', 'verbose', 'net', 'sta', 'loc', 'cha', 'rot_seed', 'tra_seed',
            'station_longitude', 'station_latitude', 'project', 'workdir', 'path_to_data_out',
            'path_to_figs_out', 'rot_sds', 'tra_sds', 'mseed_file', 'dummy_trace',
            'rotate_zne', 'tra_remove_response', 'rot_remove_response', 'tra_output',
            'runit', 'tunit', 'mu', 'pol_applied', 'pol_dict', 'rot_components',
            'rot_target', 'use_romy_zne', 'keep_z', 'base_catalog', 'event_info',
            'time_intervals', 'f_center', 'f_lower', 'f_upper', 'st0', 'st', 'sampling_rate',
            'baz_step', 'baz_win_sec', 'baz_win_overlap', 'spectra'
        ]
        
        for attr in attributes_to_copy:
            if hasattr(self, attr):
                original_value = getattr(self, attr)
                if original_value is not None:
                    # Deep copy for complex objects, shallow copy for simple ones
                    if isinstance(original_value, (dict, list, Stream, Inventory)):
                        setattr(new_instance, attr, copy.deepcopy(original_value))
                    else:
                        setattr(new_instance, attr, copy.copy(original_value))
        
        return new_instance

    def attributes(self) -> List[str]:
        """
        Get list of instance attributes
        
        Returns:
        --------
        List[str]
            List of attribute names
        """
        return [a for a in dir(self) if not a.startswith('__')]

    def check_path(self, dir_to_check):
        """Check if directory exists and create if not"""
        if not os.path.isdir(dir_to_check):
            os.makedirs(dir_to_check)
            # Fix: Remove extra self parameter
            self.check_path(dir_to_check)
        else:
            print(f" -> {dir_to_check} exists")

    def get_stream(self, stream_type: str="all", raw: bool=False) -> Stream:
        """
        Get stream data based on specified type and processing level.

        Args:
            stream_type (str): Type of stream to return. Options: "rotation", "translation", or "all".
                              Defaults to "all".
            raw (bool): If True, returns raw unprocessed data. If False, returns processed data.
                       Defaults to False.

        Returns:
            Stream: ObsPy Stream object containing the requested data.
        """
        if stream_type == "rotation":
            if raw:
                return self.st0.select(channel="*J*").copy()
            else:
                return self.st.select(channel="*J*").copy()
        elif stream_type == "translation":
            if raw:
                # return only traces with H in central channel position
                stx = Stream()
                for tr in self.st0.select(channel="*H*"):
                    if tr.stats.channel[1] == "H":
                        stx.append(tr)
                return stx
            else:
                stx = Stream()
                for tr in self.st.select(channel="*H*"):
                    if tr.stats.channel[1] == "H":
                        stx.append(tr)
                return stx
        elif stream_type == "all":
            if raw:
                return self.st0.copy()
            else:
                return self.st.copy()
        else:
            raise ValueError(f"Invalid stream type: {stream_type}. Use 'rotation' or 'translation'.")

    def get_event_info(self, origin_time: Union[str, UTCDateTime], time_margin: float=60.0, base_catalog: str="ISC", magnitude: float=6.0) -> Dict:
        """
        Get earthquake event information from catalog based on origin time
        
        Parameters:
        -----------
        origin_time : str or UTCDateTime
            Origin time of the earthquake
        time_margin : float, optional
            Time margin in seconds to search around origin time
        base_catalog : str, optional
            Base catalog to use for event search (default: "ISC")
        magnitude : float, optional
            Reference magnitude to search around (will search ±1 magnitude unit)
            
        Returns:
        --------
        Dict : Event information including location, magnitude, etc.
        """
        
        from obspy.clients.fdsn import Client
        from obspy.geodetics.base import gps2dist_azimuth, kilometers2degrees
        
        # Convert time to UTCDateTime if string
        try:
            t0 = UTCDateTime(origin_time)
        except Exception as e:
            print(f"Error converting origin time: {e}")
            return {}

        self.base_catalog = base_catalog
        try:
            # Initialize FDSN client for chosen catalog
            client = Client(self.base_catalog)

            # Check if station coordinates are available
            if not hasattr(self, 'station_latitude') or not hasattr(self, 'station_longitude'):
                print("Station coordinates not set. Please set station_latitude and station_longitude or load data and inventory first..")
                return {}
            
            # Search for events
            catalog = client.get_events(
                starttime=t0 - time_margin,
                endtime=t0 + time_margin,
                minmagnitude=float(magnitude)-1,
                maxmagnitude=float(magnitude)+1,
                orderby='time'
            )
            
            if len(catalog) == 0:
                print(f"No events found in {self.base_catalog} catalog for given time.")
                return {}
                
            # Get closest event in time
            event = catalog[0]
            origin = event.origins[0]
            magnitude_obj = event.magnitudes[0]
            
            # Calculate distance and backazimuth
            distance, az, baz = gps2dist_azimuth(
                float(origin.latitude),
                float(origin.longitude),
                float(self.station_latitude),
                float(self.station_longitude),
            )
            
            # Store event information
            self.event_info = {
                'origin_time': origin.time,
                'latitude': float(origin.latitude),
                'longitude': float(origin.longitude),
                'depth_km': float(origin.depth) / 1000 if origin.depth else None,
                'magnitude': float(magnitude_obj.mag),
                'magnitude_type': str(magnitude_obj.magnitude_type),
                'distance_km': round(float(distance) / 1000, 2),
                'distance_deg': kilometers2degrees(float(distance) / 1000),
                'azimuth': round(float(az), 2),
                'backazimuth': round(float(baz), 2),
                'catalog': str(self.base_catalog),
                'event_id': str(event.resource_id.id)
            }
            
            if self.verbose:
                print(f"Found event:")
                print(f"Origin time: {self.event_info['origin_time']}")
                print(f"Magnitude: {self.event_info['magnitude']} {self.event_info['magnitude_type']}")
                print(f"Location: {self.event_info['latitude']:.3f}°N, {self.event_info['longitude']:.3f}°E")
                print(f"Depth: {self.event_info['depth_km']:.1f} km")
                print(f"Epicentral Distance: {self.event_info['distance_km']:.2f} km")
                print(f"Epicentral Distance: {self.event_info['distance_deg']:.1f}°")
                print(f"Backazimuth: {self.event_info['backazimuth']:.2f}°")
            
            return self.event_info
            
        except Exception as e:
            print(f"Error getting event information from {self.base_catalog}:")
            if self.verbose:
                print(e)

            # prepare dummy event info
            self.event_info = {
                'origin_time': t0,
                'latitude': 0,
                'longitude': 0,
                'depth_km': 0,
                'magnitude': 0,
                'backazimuth': 0,
                'distance_km': 0,
                'distance_deg': 0,
                'azimuth': 0,
            }
            print(f"No event found in {self.base_catalog} catalog for given time. Using dummy event info.")
            return self.event_info

    def get_theoretical_arrival(self, phase: str='P') -> float:
        """
        Calculate theoretical arrival time for given phase
        
        Parameters:
        -----------
        phase : str
            Seismic phase name (e.g., 'P', 'S', 'PP', etc.)
            
        Returns:
        --------
        float : Theoretical arrival time as UTCDateTime
        """
        try:
            from obspy.taup import TauPyModel
            model = TauPyModel(model="iasp91")
            
            if not hasattr(self, 'event_info'):
                print("No event information available. Run get_event_info first.")
                return None
                
            # Get arrivals
            arrivals = model.get_travel_times(
                source_depth_in_km=self.event_info['depth_km'],
                distance_in_degree=self.event_info['distance_deg'],
                phase_list=[phase]
            )
            
            if len(arrivals) == 0:
                print(f"No theoretical arrivals found for phase {phase}")
                return None
                
            arrival_time = self.event_info['origin_time'] + arrivals[0].time
            
            if self.verbose:
                print(f"Theoretical {phase}-arrival: {arrival_time}")
                
            return arrival_time
            
        except Exception as e:
            print(f"Error calculating theoretical arrival time:")
            if self.verbose:
                print(e)
            return None

    def get_time_intervals(self, tbeg: Union[None, str, UTCDateTime]=None, tend: Union[None, str, UTCDateTime]=None, interval_seconds: int=3600, interval_overlap: int=0) -> List[Tuple[UTCDateTime, UTCDateTime]]:
        '''
        Obtain time intervals
        '''

        from obspy import UTCDateTime

        if tbeg is None:
            tbeg = self.tbeg
        else:
            tbeg = UTCDateTime(tbeg)

        if tend is None:
            tend = self.tend
        else:
            tend = UTCDateTime(tend)

        times = []
        t1, t2 = tbeg, tbeg + interval_seconds

        while t2 <= tend:
            times.append((t1, t2))
            t1 = t1 + interval_seconds - interval_overlap
            t2 = t2 + interval_seconds - interval_overlap

        self.time_intervals = times
        return times

    def get_octave_bands(self, fmin: Union[None, float]=None, fmax: Union[None, float]=None, faction_of_octave: int=1, plot: bool=False) -> Tuple[array, array, array]:
        '''
        Computing octave frequency bands

        Arguments:
            - fmin:    (float) minimum center frequency
            - fmax:    (float) maximum center frequency
            - fraction_of_octave:    (int) octave fraction (e.g. [1] = octaves, 3 = third octaves, 12 = 12th octaves)
            - plot:    (bool) show frequency bands

        Example:

        >>> flower, fupper, fcenter = get_octave_bands(f_min, f_max, fband_type="octave", plot=False)
        '''

        import matplotlib.pyplot as plt
        from acoustics.octave import Octave
        from numpy import array

        if fmin is None:
            fmin = self.fmin
        if fmax is None:
            fmax = self.fmax

        # avoid fmin = zero
        if fmin == 0:
            print(f" -> set fmin to 1e-10")
            fmin = 1e-10

        f_lower, f_upper, f_center = [], [], []

        # compute f-bands
        _octaves = Octave(fraction=faction_of_octave,
                          interval=None,
                          fmin=fmin,
                          fmax=fmax,
                          unique=False,
                          reference=1000.0
                         )

        # store for object
        self.f_center = array(_octaves.center)
        self.f_lower = array(_octaves.lower)
        self.f_upper = array(_octaves.upper)

        # checkup plot
        if plot:
            plt.figure(figsize=(15, 5))
            for fl, fc, fu in zip(self.f_lower, self.f_center, self.f_upper):
                plt.axvline(fu, color="r")
                plt.axvline(fl, color="r")
                plt.axvline(fc, ls="--")
                plt.axvline(fmin, color="g")
                plt.axvline(fmax, color="g")
                plt.xscale("log")
            plt.show();

        return self.f_lower, self.f_upper, self.f_center

    def add_dummy_trace(self, stream, template_seed: str="XX.XXXX..XXX"):
        """
        Add a dummy trace to the stream based on a template seed.

        Args:
            stream: ObsPy Stream object to add the dummy trace to.
            template_seed (str): Template seed ID in format "NET.STA.LOC.CHA".
                               Defaults to "XX.XXXX..XXX".

        Returns:
            Stream: Stream with the added dummy trace.
        """
        from obspy import Trace
        from numpy import zeros
        # get template seed
        _net, _sta, _loc, _cha = template_seed.split('.')

        # copy trace
        tmp = stream[0].copy()

        # adjust trace
        tmp.data = np.zeros(tmp.stats.npts)
        tmp.stats.channel = _cha
        tmp.stats.location = _loc
        tmp.stats.station = _sta
        tmp.stats.network = _net

        stream += tmp
        return stream

    def apply_dummy_traces(self, stream: Stream=None) -> Stream:
        """
        Apply dummy traces to the loaded stream based on configuration.
        This method should be called after data is loaded.

        Args:
            stream (Stream, optional): Stream to apply dummy traces to. 
                                      If None, uses self.st. Defaults to None.

        Returns:
            Stream: Stream with dummy traces added.
        """
        if stream is None:
            stream = self.st
        
        if not self.dummy_trace:
            return stream
        
        # Apply dummy traces based on channel type
        for dummy in self.dummy_trace:
            # Check if dummy trace is for rotation (J) or translation (H)
            if len(dummy) >= 2 and dummy[-2] in ["J", "H"]:
                # Check if this dummy trace already exists in the stream
                dummy_exists = any(
                    tr.id == dummy for tr in stream
                )
                
                if not dummy_exists:
                    if self.verbose:
                        print(f"-> adding dummy trace: {dummy}")
                    stream = self.add_dummy_trace(stream, dummy)
                elif self.verbose:
                    print(f"-> dummy trace {dummy} already exists, skipping")
        
        return stream

    def get_component_lag(self, normalize: bool=True, baz: float=None, correct_traces: bool=True, raw: bool=False) -> Dict:
        """
        Get lag between rotation and translation components
        
        Parameters:
        -----------
        component : str
            Component to analyze ('Z', 'N', 'E', or 'T')
        normalize : bool
            Normalize cross-correlation
        baz : float
            Backazimuth
        correct_traces : bool
            Correct time lag for traces
        raw : bool
            Use raw stream
        Returns:
        --------
        Dict with lag information:
            'lag_samples': lag in samples
            'lag_time': lag in seconds
            'cc_max': maximum correlation coefficient
        """
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from obspy.signal.rotate import rotate_ne_rt
        from numpy import roll

        if baz is None:
            if hasattr(self, 'event_info') and 'backazimuth' in self.event_info:
                baz = self.event_info['backazimuth']
            else:
                raise ValueError("No backazimuth provided or available")
        else:
            baz = float(baz)

        # Get components
        rot = self.get_stream("rotation", raw=raw).copy()
        tra = self.get_stream("translation", raw=raw).copy()
        
        # rotate components
        rot_r, rot_t = rotate_ne_rt(
            rot.select(channel="*N")[0].data,
            rot.select(channel="*E")[0].data,
            baz
        )
        tra_r, tra_t = rotate_ne_rt(
            tra.select(channel="*N")[0].data,
            tra.select(channel="*E")[0].data,
            baz
        )

        tra_z = tra.select(channel="*Z")[0].data
        rot_z = rot.select(channel="*Z")[0].data

        # Compute cross-correlation
        cc0 = correlate(tra_z, rot_t, 0, normalize=normalize)
        _, cc0_max_h = xcorr_max(cc0)

        cc = correlate(tra_z, rot_t, len(rot_t), normalize=normalize)
        lag_samples_h, cc_max_h = xcorr_max(cc)
        
        # Convert to time
        lag_time_h = lag_samples_h / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-T & ACC-Z:  cc_zero: {cc0_max_h:.2f}, lag_time: {lag_time_h} s, lag_samples: {lag_samples_h}, cc_max: {cc_max_h:.2f}")

        # Compute cross-correlation
        cc0 = correlate(tra_t, rot_z, 0, normalize=normalize)
        _, cc0_max_z = xcorr_max(cc0)

        cc = correlate(tra_t, rot_z, len(rot_z), normalize=normalize)
        lag_samples_z, cc_max_z = xcorr_max(cc)
        
        # Convert to time
        lag_time_z = lag_samples_z / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-Z & ACC-T:  cc_zero: {cc0_max_z:.2f}, lag_time: {lag_time_z} s, lag_samples: {lag_samples_z}, cc_max: {cc_max_z:.2f}")
       
        # shift rotataion waveforms
        if correct_traces:
            for tr in rot:
                if tr.stats.channel.endswith("Z"):
                    print(f" -> shifting rotation Z waveform by {lag_time_z} seconds")
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_z
            
                if tr.stats.channel.endswith("N") or tr.stats.channel.endswith("E"):
                    print(f" -> shifting rotation N/E waveform by {lag_time_h} seconds")
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_h

        # shift rotataion waveforms for raw stream
        rot_raw = self.get_stream("rotation", raw=True).copy()
        if correct_traces:
            for tr in rot_raw:
                if tr.stats.channel.endswith("Z"):
                    print(f" -> shifting rotation Z waveform by {lag_time_z} seconds")
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_z
            
                if tr.stats.channel.endswith("N") or tr.stats.channel.endswith("E"):
                    print(f" -> shifting rotation N/E waveform by {lag_time_h} seconds")
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_h
              
        # update and trim raw stream
        if correct_traces:
            # reassign raw stream
            self.st0 = rot_raw + self.get_stream("translation", raw=True).copy()
            # trim stream
            self.st0.trim(self.tbeg, self.tend)
            # avoid overlaps
            self.st0 = self.st0.merge(method=1)

            # reassign stream
            self.st = rot + tra
            # trim stream
            self.st.trim(self.tbeg, self.tend)
            # avoid overlaps
            self.st = self.st.merge(method=1)

    def write_to_sds(self, stream, sds_path: str, format: str="MSEED") -> None:
        """
        Write stream to SDS file structure
        
        Parameters:
        -----------
        stream : obspy.Stream
            Stream to write
        sds_path : str
            Root path of SDS archive
        format : str, optional
            Data format (default: "MSEED")
        """
        from obspy.clients.filesystem.sds import Client
        import os
        
        try:
            # Initialize SDS client
            client = Client(sds_path)
            
            # Write each trace to SDS structure
            for tr in stream:
                # Get time info
                year = str(tr.stats.starttime.year)
                julday = "%03d" % tr.stats.starttime.julday
                
                # Create directory structure
                net_dir = os.path.join(sds_path, year, tr.stats.network)
                sta_dir = os.path.join(net_dir, tr.stats.station)
                cha_dir = os.path.join(sta_dir, f"{tr.stats.channel}.D")
                
                # Create directories if they don't exist
                for directory in [net_dir, sta_dir, cha_dir]:
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                
                # Write trace
                stream.write(tr, format=format)
                
                if self.verbose:
                    print(f"Written: {tr.id} for {year}-{julday}")
                    
        except Exception as e:
            print(f"Error writing to SDS: {e}")#

    def sort_channels(self, stream, components: str):
        """
        Sort stream channels by specified components.

        Args:
            stream: ObsPy Stream object to sort.
            components (str): String of component letters to select and sort by.

        Returns:
            Stream: New stream with channels sorted by the specified components.
        """
        stsort = Stream()
        for c in components:
            stsort += stream.select(component=c)
        return stsort

    def update_seeds(self):
        '''
        Update seeds with master seed
        '''
        new_rot_seeds = []
        for x in self.rot_seed:
            out = x.split('.')
            out[0], out[1] = self.net, self.sta
            new_rot_seeds.append(".".join(out))

        new_tra_seeds = []
        for x in self.tra_seed:
            out = x.split('.')
            out[0], out[1] = self.net, self.sta
            new_tra_seeds.append(".".join(out))
            
        self.rot_seed = new_rot_seeds
        self.tra_seed = new_tra_seeds

    def trim(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime]):
        '''
        Trim stream
        '''
        self.st.trim(t1, t2)
        self.st0.trim(t1, t2)
        self.tbeg = t1
        self.tend = t2

    def load_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime], resample_rate: Optional[float]=None, merging: bool=False):
        '''
        Load data for translation and rotaion as obspy stream

        @param t1: starttime
        @param t2: endtime  
        @param resample_rate: resample rate in Hz
        @param merging: merge stream if True
        @type t1: str or UTCDateTime
        @type t2: str or UTCDateTime
        @type resample_rate: float or None
        @type merging: bool
        '''

        from obspy import Stream, UTCDateTime, read
        from obspy.clients.fdsn import Client as FDSNClient

        st0 = Stream()

        t1, t2 = UTCDateTime(t1), UTCDateTime(t2)

        if len(self.tra_seed) > 0:
            # add to stream
            st0 += self._load_translation_data(t1, t2, merging=merging)

        if len(self.rot_seed) > 0:
            # add to stream
            st0 += self._load_rotation_data(t1, t2, merging=merging)

        # check if stream has correct length
        if len(st0) < (len(self.tra_seed) + len(self.rot_seed)):
            print(f" -> missing stream data")

       # check if merging is required
        if len(st0) > (len(self.tra_seed) + len(self.rot_seed)):
            st0 = st0.merge(method=1, fill_value=0)

        # apply dummy traces after data is loaded
        st0 = self.apply_dummy_traces(st0)

        # resample stream
        if resample_rate is not None:
            if self.verbose:
                print(f"-> resampling stream to {resample_rate} Hz")
            for tr in st0:
                tr = tr.detrend("demean")
                tr = tr.detrend("linear")
                tr = tr.detrend("simple")
                # tr = tr.taper(max_percentage=0.05, type='cosine')
                # tr = tr.filter("highpass", freq=0.001, corners=2, zerophase=True)
                tr = tr.filter("lowpass", freq=resample_rate/4, corners=2, zerophase=True)
                tr = tr.resample(resample_rate, no_filter=True)
                tr = tr.detrend("demean")

                # adjust channel code
                if resample_rate >= 100:
                    tr.stats.channel = "H"+tr.stats.channel[1:]
                elif resample_rate >= 10 and resample_rate < 100:
                    tr.stats.channel = "B"+tr.stats.channel[1:]
                elif resample_rate < 10:
                    tr.stats.channel = "L"+tr.stats.channel[1:]

            if self.verbose:
                print(st0)

        # Update stream IDs
        for tr in st0:
            tr.stats.network = self.net
            tr.stats.station = self.sta
            tr.stats.location = self.loc

        # assign stream as raw stream
        self.st0 = st0

        # assign stream to object
        self.st = st0

        # write polarity dictionary for all traces
        self.pol_dict.update({tr.stats.channel[1:]: 1 for tr in self.st})

        # Check if all traces have the same sampling rate and add as attribute
        if len(self.st) > 0:
            sampling_rates = set(tr.stats.sampling_rate for tr in self.st)
            if len(sampling_rates) == 1:
                self.sampling_rate = sampling_rates.pop()
            else:
                print(" WARNING: Not all traces have the same sampling rate!")
                if self.verbose:
                    print(f"Sampling rates found: {sampling_rates}")

    def _load_translation_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime], merging: bool=False):
        '''
        Load translation data

        @param t1: starttime
        @param t2: endtime
        @param merging: merge stream if True
        @type t1: str or UTCDateTime
        @type t2: str or UTCDateTime
        @type merging: bool
        '''

        # Load translation data
        tra = Stream()

        # initialize FDSN client
        client = None
        if self.data_source.lower() == 'fdsn':
            try:
                client = FDSNClient(self.fdsn_client_tra)
            except Exception as e:
                print(f"WARNING: failed to initialize FDSN client: {str(e)}")

        for tseed in self.tra_seed:

            net, sta, loc, cha = tseed.split('.')

            try:
                if self.data_source.lower() == 'sds':
                    if self.verbose:
                        print(f"-> fetching {tseed} data from SDS")
                    # read from local SDS archive
                    tra += self.read_from_sds(self.tra_sds, tseed, t1-1, t2+1)
                
                elif self.data_source.lower() == 'fdsn':
                    # read from FDSN web service
                    if self.verbose:
                        print(f"-> fetching {tseed} data from FDSN")
                    tra += client.get_waveforms(network=net, station=sta,
                                                location=loc, channel=cha,
                                                starttime=t1-1, endtime=t2+1)
                    
                elif self.data_source.lower() == 'mseed_file':
                    # check if mseed file exists
                    if not os.path.exists(self.mseed_file):
                        raise ValueError(f"Mseed file {self.mseed_file} does not exist!")
                    if self.verbose:
                        print(f"-> fetching {tseed} data from mseed file")

                    tra0 = read(self.mseed_file)
                    tra += tra0.select(network=net, station=sta, location=loc, channel=cha)
                    tra = tra.trim(t1-1, t2+1)
                
                else:
                    raise ValueError(f"Unknown data source: {self.data_source}. Use 'sds' or 'fdsn'.")

            except Exception as e:
                print(f" -> loading translational data failed!")
                if self.verbose:
                    print(e)

        # merge stream if required
        if merging:
            tra = tra.merge(method=1, fill_value=0)

        # get inventory
        if self.tra_inv_file is not None:
            if self.verbose:
                print(f"-> translation inventory provided: {self.tra_inv_file}")
            self.tra_inv = read_inventory(self.tra_inv_file)

        else:
            try:
                if client is not None:
                    # get inventory from FDSN
                    self.tra_inv = client.get_stations(network=net, station=sta,
                                                        starttime=t1, endtime=t2,
                                                        level="response",
                                                        )
                    if self.verbose:
                        print(f"-> translation inventory requested: {self.tra_inv}")
            except Exception as e:
                self.tra_inv = None
                print(f"WARNING: failed to get inventory: {str(e)}")

        # Get station coordinates
        if self.tra_inv is not None:
            if self.station_latitude is None and self.station_longitude is None:
                try:
                    coords = self.tra_inv.get_coordinates(self.tra_seed[0])
                    self.station_latitude = coords['latitude']
                    self.station_longitude = coords['longitude']
                except Exception as e:
                    print(f"WARNING: failed to get station coordinates: {str(e)}")

        # Remove response of translation data
        if self.tra_remove_response and self.tra_inv is not None:
            try:
                # remove response
                tra = tra.remove_response(self.tra_inv, output=self.tra_output)
                if self.verbose:
                    print(f"-> removing response: {self.tra_output}")
                # detrend
                tra = tra.detrend("linear")
            except Exception as e:
                print(f"WARNING: failed to remove response: {str(e)}")

        # rotate to ZNE
        if self.rotate_zne and self.tra_inv is not None:
            channels = [tr.stats.channel[-1] for tr in tra]

            if self.verbose:
                print(f"-> rotating translational data {''.join(channels)} to ZNE")

            # tra = tra.rotate(method="->ZNE", inventory=self.tra_inv)

            if 'Z' in channels and 'X' in channels and 'Y' in channels:
                tra = self.sort_channels(tra, ['Z', 'X', 'Y'])
                tra = tra._rotate_to_zne(self.tra_inv, components='ZXY')
            elif 'Z' in channels and 'N' in channels and 'E' in channels:
                tra = self.sort_channels(tra, ['Z', 'N', 'E'])
                tra = tra._rotate_to_zne(self.tra_inv, components='ZNE')
            elif 'U' in channels and 'V' in channels and 'W' in channels:
                tra = tra._rotate_to_zne(self.tra_inv, components='UVW')
            elif 'Z' in channels and '1' in channels and '2' in channels:
                tra = tra._rotate_to_zne(self.tra_inv, components='Z12')
            else:
                print(f"WARNING: unknown rotation components: {channels}")

        if self.verbose:
            print(tra)
        
        return tra

    def _load_rotation_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime], merging: bool=False):
        '''
        Load rotation data

        @param t1: starttime
        @param t2: endtime
        @param merging: merge stream if True
        @type t1: str or UTCDateTime
        @type t2: str or UTCDateTime
        @type merging: bool
        '''
        # Load rotation data
        rot = Stream()

        # initialize FDSN client
        client = None
        if self.data_source.lower() == 'fdsn':
            try:
                client = FDSNClient(self.fdsn_client_rot)
            except Exception as e:
                print(f"WARNING: failed to initialize FDSN client: {str(e)}")

        # raw channel order
        channel_raw = {"Z": "3", "N": "2", "E": "1"}

        for rseed in self.rot_seed:
            net, sta, loc, cha = rseed.split('.')

            try:
                if self.data_source.lower() == 'sds':
                    # read from local SDS archive
                    if self.verbose:
                        print(f"-> fetching {rseed} data from SDS")
                    rot += self.read_from_sds(self.rot_sds, rseed, t1-1, t2+1)

                elif self.data_source.lower() == 'fdsn':
                    # read from FDSN web service
                    if self.verbose:
                        print(f"-> fetching {rseed} data from FDSN")
                    if sta == "BSPF":
                        rot += client.get_waveforms(net, sta, loc, cha[:2]+channel_raw[cha[2]], t1-1, t2+1)
                    elif sta == "ROMY":
                        cl = FDSNClient(base_url=self.fdsn_client_rot)
                        rot += cl.get_waveforms(net, sta, loc, cha, t1-1, t2+1)
                    else:
                        rot += client.get_waveforms(net, sta, loc, cha, t1-1, t2+1)

                elif self.data_source.lower() == 'mseed_file':
                    # check if mseed file exists
                    if not os.path.exists(self.mseed_file):
                        raise ValueError(f"Mseed file {self.mseed_file} does not exist!")
                    # read directly from mseed file
                    if self.verbose:
                        print(f"-> fetching {rseed} data from mseed file")

                    rot0 = read(self.mseed_file)
                    rot += rot0.select(
                        network=net,
                        station=sta,
                        location=loc,
                        channel=cha
                    )
                    rot = rot.trim(t1-1, t2+1)
                else:
                    raise ValueError(f"Unknown data source: {self.data_source}. Use 'sds' or 'fdsn'.")
            except Exception as e:
                print(f" -> loading rotational data failed!")
                if self.verbose:
                    print(e)

        # merge stream if required
        if merging:
            rot = rot.merge(method=1, fill_value=0)

        # get inventory
        if self.rot_inv_file is not None:
            if self.verbose:
                print(f"-> rotation inventory provided: {self.rot_inv_file}")
            self.rot_inv = read_inventory(self.rot_inv_file)

        elif self.data_source.lower() == 'fdsn':
            try:
                # get inventory from FDSN
                self.rot_inv = client.get_stations(network=net, station=sta,
                                                    starttime=t1, endtime=t2,
                                                    level="response",
                                                    )
            except Exception as e:
                self.rot_inv = None
                print(f"WARNING: failed to get inventory: {str(e)}")

        # assign station coordinates
        if self.rot_inv is not None:
            if self.station_latitude is None and self.station_longitude is None:
                if self.rot_inv is not None:
                    coords = self.rot_inv.get_coordinates(self.rot_seed[0])
                    self.station_latitude = coords['latitude']
                    self.station_longitude = coords['longitude']
                else:
                    print(f"WARNING: Cannot assign station coordinates! \nProvide a rotation inventory file or set station_latitude and station_longitude in configuration!")

        # remove sensitivity
        if self.rot_remove_response and self.rot_inv is not None:
            try:
                # remove sensitivity
                rot = rot.remove_sensitivity(self.rot_inv)
                # detrend
                rot = rot.detrend("linear")

                if self.verbose:
                    print("-> removing sensitivity")
    
            except Exception as e:
                print(f"WARNING: failed to remove sensitivity: {str(e)}")

        # rotate to ZNE
        if self.rotate_zne:
            # check if ROMY data is present
            if any("ROMY" in rseed for rseed in self.rot_seed):
                # Add option to use rotate_romy_zne for ROMY data
                if self.use_romy_zne and self.rot_inv:
                    try:
                        if self.verbose:
                            print(f"-> rotated ROMY data using rotate_romy_zne (keep_z={self.keep_z})")
                        # get components
                        components = [tr.stats.channel[-1] for tr in rot]
                        # if differenz npts then equalize npts
                        sizes = [tr.stats.npts for tr in rot]
                        if len(set(sizes)) > 1:
                            for tr in rot:
                                tr.data = tr.data[:min(sizes)]
                        # rotate
                        rot = self.rotate_romy_zne(
                            rot, 
                            self.rot_inv,
                            use_components=components,
                            keep_z=self.keep_z
                        )
                    except Exception as e:
                        print(f"WARNING: ROMY ZNE rotation failed: {str(e)}")
            else:
                channels = [tr.stats.channel[-1] for tr in rot]
                if self.verbose:
                    print(f"-> rotating rotational data {''.join(channels)} to ZNE") 
                # rot = rot.rotate(method="->ZNE", inventory=self.rot_inv)
                if 'Z' in channels and 'N' in channels and 'E' in channels:
                    rot = rot._rotate_to_zne(self.rot_inv, components='ZNE')
                elif 'Z' in channels and 'X' in channels and 'Y' in channels:
                    rot = self.sort_channels(rot, ['Z', 'Y', 'X'])
                    rot = rot._rotate_to_zne(self.rot_inv, components='ZYX')
                elif 'U' in channels and 'V' in channels and 'W' in channels:
                    rot = rot._rotate_to_zne(self.rot_inv, components='UVW')
                elif 'Z' in channels and '1' in channels and '2' in channels:
                    rot = rot._rotate_to_zne(self.rot_inv, components='Z12')
                elif '1' in channels and '2' in channels and '3' in channels:
                    rot = rot._rotate_to_zne(self.rot_inv, components='321')
                else:
                    print(f"WARNING: unknown rotation components: {channels}")

        # assign station coordinates
        if self.station_latitude is None and self.station_longitude is None:
            if self.rot_inv is not None:
                coords = self.rot_inv.get_coordinates(self.rot_seed[0])
                self.station_latitude = coords['latitude']
                self.station_longitude = coords['longitude']
            else:
                print(f"WARNING: Cannot assign station coordinates! \nProvide a rotation inventory file or set station_latitude and station_longitude in configuration!")

        if self.verbose:
            print(rot)
        
        return rot

    def filter_data(self, fmin: Optional[float]=None, fmax: Optional[float]=None, output: bool=False):
        """
        Apply bandpass filter to the data stream.

        Args:
            fmin (Optional[float]): Minimum frequency for bandpass filter in Hz. If None, no lower limit.
            fmax (Optional[float]): Maximum frequency for bandpass filter in Hz. If None, no upper limit.
            output (bool): Whether to return the filtered stream. Defaults to False.

        Returns:
            Stream: Filtered stream if output=True, otherwise None.
        """
        if fmin is None and fmax is None:
            print("WARNING: No frequencies specified. Returning original stream.")

        # reset stream to raw stream
        self.st = self.st0.copy()

        # set fmin and fmax
        if fmin is not None:
            self.fmin = fmin
        if fmax is not None:
            self.fmax = fmax

        # detrend and filter
        self.st = self.st.detrend("linear")
        self.st = self.st.detrend("demean")
        self.st = self.st.taper(0.05, type='cosine')

        if fmin is not None and fmax is not None:
            self.st = self.st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True) 
        elif fmin is not None:
            self.st = self.st.filter("lowpass", freq=fmin, corners=4, zerophase=True)
        elif fmax is not None:
            self.st = self.st.filter("highpass", freq=fmax, corners=4, zerophase=True)

        # return stream if output is True
        if output:
            return self.st

    def trim_stream(self, set_common: bool=True, set_interpolate: bool=False):
        '''
        Trim a stream to common start and end times (and interpolate to common times)
        '''

        from numpy import interp, arange

        def _get_size(st0: Stream) -> List[int]:
            return [tr.stats.npts for tr in st0]

        # get size of traces
        n_samples = _get_size(self.st)

        # check if all traces have same amount of samples
        if not all(x == n_samples[0] for x in n_samples):
            if self.verbose:
                print(f" -> stream size inconsistent: {n_samples}")

            # if difference not larger than one -> adjust
            if any([abs(x-n_samples[0]) > 1 for x in n_samples]):

                # set to common minimum interval
                if set_common:
                    _tbeg = max([tr.stats.starttime for tr in self.st])
                    _tend = min([tr.stats.endtime for tr in self.st])
                    self.st = self.st.trim(_tbeg, _tend, nearest_sample=True)
                    if self.verbose:
                        print(f"  -> adjusted: {_get_size(self.st)}")

                    if set_interpolate:
                        _times = arange(0, min(_get_size(self.st)), self.st[0].stats.delta)
                        for tr in self.st:
                            tr.data = interp(_times, tr.times(reftime=_tbeg), tr.data)
            else:
                # adjust for difference of one sample
                for tr in self.st:
                    tr.data = tr.data[:min(n_samples)]
                if self.verbose:
                    print(f"  -> adjusted: {_get_size(self.st)}")

    def correct_tilt(self, g: float=9.81, raw: bool=False):
        '''
        Correct tilt of horizontal translation components

        Args:
            g (float): Acceleration due to gravity in m/s^2. Defaults to 9.81 m/s^2.
            raw (bool): If True, correct the tilt of the raw stream. Defaults to False.

        Returns:
            None: Modifies the stream in place.

        Examples:
            >>> # Correct tilt of horizontal translation components
            >>> sd.correct_tilt(g=9.81, raw=False)

            >>> # Correct tilt of horizontal translation components for raw stream
            >>> sd.correct_tilt(g=9.81, raw=True)
        '''
        self.st.select(channel="*HN")[0].data -= -g*self.st.select(channel="*JE")[0].data
        self.st.select(channel="*HE")[0].data -= g*self.st.select(channel="*JN")[0].data

        if raw:
            self.st0.select(channel="*HN")[0].data -= -g*self.st0.select(channel="*JE")[0].data
            self.st0.select(channel="*HE")[0].data -= g*self.st0.select(channel="*JN")[0].data

    def polarity_stream(self, pol_dict: Dict={}, raw: bool=False):
        '''
        Modify polarity of data
        '''
        
        same_dict = [True if v == self.pol_dict[k] else False for k, v in pol_dict.items()]
        if all(same_dict):
            print("-> polarity already applied. Exiting...")
            return
        else:
            self.pol_applied = True
            for k, v in pol_dict.items():
                if k not in self.pol_dict:
                    raise ValueError(f"Channel {k} not found in polarity dictionary")
                elif self.pol_dict[k] != v:
                    print(f"-> polarity for channel {k} changed from {self.pol_dict[k]} to {v}")
                    self.pol_dict[k] = v

                    # apply polarity to data
                    for tr in self.st:
                        if k in tr.stats.channel:
                            tr.data *= pol_dict[k]
                    if raw:
                        for tr in self.st0:
                            if k in tr.stats.channel:
                                tr.data *= pol_dict[k]

    def replace_values_in_window(self, tbeg: Union[str, UTCDateTime], 
                                  tend: Union[str, UTCDateTime], 
                                  value: float=0.0, 
                                  channel_list: Optional[List[str]]=None,
                                  raw: bool=False) -> None:
        """
        Replace values in a time window for specified channels with a given value.
        
        This method replaces all data values within the specified time window
        for traces matching the given channel patterns with the specified value.
        Useful for masking out bad data, removing glitches, or zeroing specific
        time periods.
        
        Args:
            tbeg (Union[str, UTCDateTime]): Start time of the window to replace.
                                                 Can be a string or UTCDateTime object.
            tend (Union[str, UTCDateTime]): End time of the window to replace.
                                               Can be a string or UTCDateTime object.
            value (float): Value to replace the data with. Defaults to 0.0.
            channel_list (List[str]): List of channel patterns to match (e.g., ['BJZ', 'BJN']).
                                     Patterns can be full channel names or wildcards (e.g., ['*JZ']).
                                     If None, applies to all traces in the stream. Defaults to None.
            raw (bool): If True, also applies the replacement to the raw stream (self.st0).
                       If False, only modifies the processed stream (self.st). Defaults to False.
        
        Returns:
            None: Modifies the stream in place.
        
        Examples:
            >>> # Replace values for BJZ channel between two times with 0
            >>> sd.replace_values_in_window(tbeg='2025-12-08 14:20:00', tend='2025-12-08 14:25:00', 
            ...                             value=0.0, channel_list=['BJZ'])
            
            >>> # Replace values for multiple channels with NaN
            >>> sd.replace_values_in_window(tbeg='2025-12-08 14:20:00', tend='2025-12-08 14:25:00',
            ...                             value=np.nan, channel_list=['BJZ', 'BJN', 'BJE'])
            
            >>> # Replace values for all rotation channels using wildcard
            >>> sd.replace_values_in_window(tbeg='2025-12-08 14:20:00', tend='2025-12-08 14:25:00',
            ...                             value=0.0, channel_list=['*J*'])
        """
        from fnmatch import fnmatch
        
        # Convert times to UTCDateTime if needed
        tbeg = UTCDateTime(tbeg)
        tend = UTCDateTime(tend)
        
        # Validate time window
        if tbeg >= tend:
            raise ValueError(f"tbeg ({tbeg}) must be before tend ({tend})")
        
        # Determine which stream to modify
        stream = self.st0 if raw else self.st
        
        if stream is None or len(stream) == 0:
            raise ValueError("Stream is empty. Please load data first.")
        
        # Track which traces were modified
        modified_traces = []
        
        # Helper function to check if a channel matches a pattern
        def channel_matches(channel: str, pattern: str) -> bool:
            """
            Check if a channel matches a pattern.
            - Single characters match channels ending with that character (e.g., 'Z' matches 'BJZ', 'BHZ')
            - Patterns with wildcards use fnmatch (e.g., '*Z', '*J*')
            - Other patterns match if contained in channel or exact match
            """
            # If pattern is a single character, match channels ending with it
            # This handles cases like 'Z' matching 'BJZ', 'BHZ', etc.
            if len(pattern) == 1:
                return channel.endswith(pattern) or pattern in channel
            # If pattern contains wildcards, use fnmatch
            elif '*' in pattern or '?' in pattern:
                return fnmatch(channel, pattern)
            # Otherwise, check if pattern is contained in channel or exact match
            else:
                return pattern in channel or channel == pattern
        
        # Iterate through traces in the stream
        for tr in stream:
            # Check if this trace matches any channel pattern
            if channel_list is None:
                # Apply to all traces if no channel list specified
                match = True
            else:
                # Check if trace channel matches any pattern in channel_list
                match = any(channel_matches(tr.stats.channel, pattern) for pattern in channel_list)
            
            if match:
                # Check if time window overlaps with trace time range
                trace_start = tr.stats.starttime
                trace_end = tr.stats.endtime
                
                # Skip if time window doesn't overlap with trace
                if tend < trace_start or tbeg > trace_end:
                    if self.verbose:
                        print(f"  -> Skipping {tr.id}: time window doesn't overlap with trace")
                    continue
                
                # Calculate indices for the time window
                # Clamp window to trace boundaries
                window_start = max(tbeg, trace_start) # get the maximum of the start time of the window and the start time of the trace
                window_end = min(tend, trace_end)
                
                # Get sample indices
                start_idx = int((window_start - trace_start) / tr.stats.delta)
                end_idx = int((window_end - trace_start) / tr.stats.delta) + 1
                
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, len(tr.data)))
                end_idx = max(0, min(end_idx, len(tr.data)))
                
                if start_idx < end_idx:
                    # Replace values in the window
                    tr.data[start_idx:end_idx] = value
                    modified_traces.append(tr.id)
                    
                    if self.verbose:
                        print(f"  -> Replaced {end_idx - start_idx} samples in {tr.id} "
                              f"from {window_start} to {window_end} with value {value}")

        if self.verbose and len(modified_traces) == 0:
            print(f"  -> No traces matched the channel patterns: {channel_list}")
        elif self.verbose:
            print(f"  -> Modified {len(modified_traces)} trace(s): {modified_traces}")

    @staticmethod
    def load_from_yaml(name: str):
        """
        Load an object from a yaml file.
        """
        import os, yaml
        # if file does not end with yml add yml
        if not name.endswith(".yml"):
            name += ".yml"
        # check if file exists
        if not os.path.isfile(name):
            raise FileNotFoundError(f"File {name} not found")
        # load file
        with open(name, 'r') as f:
            obj = yaml.load(f, Loader=yaml.FullLoader)

        return obj

    @staticmethod
    def load_from_pickle(name: str):
        """
        Load an object from a pickle file.
        """
        import os, pickle
        # if file does not end with pkl add pkl
        if not name.endswith(".pkl"):
            name += ".pkl"
        # check if file exists
        if not os.path.isfile(name):
            raise FileNotFoundError(f"File {name} not found")
        # load file
        with open(name, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def store_as_yaml(obj: object, name: str):
        """
        Store an object as a yaml file.
        """
        import os, yaml
        # if file does not end with yml add yml
        if not name.endswith(".yml"):
            name += ".yml"
        # check if file exists
        if os.path.isfile(name):
            print(f" -> file {name} already exists. Aborting...")
            return
        # store file
        ofile = open(name, 'w')
        yaml.dump(obj, ofile)
        ofile.close()
        # check if file is stored
        if os.path.isfile(name+".yml"):
            print(f" -> stored: {name}.yml")

    @staticmethod
    def store_as_pickle(obj: object, name: str):
        """
        Store an object as a pickle file.
        """
        import os, pickle
        # if file does not end with pkl add pkl
        if not name.endswith(".pkl"):
            name += ".pkl"
        # store file
        ofile = open(name, 'wb')
        pickle.dump(obj, ofile)
        # check if file is stored
        if os.path.isfile(name+".pkl"):
            print(f" -> stored: {name}.pkl")

    def optimize_parameters(self, wave_type: str='love', overlap: float=0.5, twin_min: float=1,
                          fbands: Dict=None, baz_step: int=1, bandwidth_factor: float=6) -> Dict:
        """
        Optimize parameters for wave analysis by maximizing cross-correlation
        """
        import numpy as np
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from obspy.signal.rotate import rotate_ne_rt
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        def filter_stream(stream: Stream, fmin: Optional[float]=None, fmax: Optional[float]=None) -> Stream:
            if fmin is None and fmax is None:
                print("WARNING: No frequencies specified. Returning original stream.")
                return stream
            stream_copy = stream.copy()
            stream_copy.detrend('linear')
            stream_copy.taper(max_percentage=0.01, type='cosine')
            stream_copy.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
            stream_copy.detrend('linear')
            return stream_copy

        def process_frequency_band(freq_params, rot, acc, df, n_samples, baz_values, wave_type, overlap, twin_min):
            fl, fu, fc = freq_params
            
            # Calculate window parameters
            f_bandwidth = fu - fl
            win_time_s_freq = max(twin_min, bandwidth_factor/f_bandwidth)
            win_samples = int(win_time_s_freq * df)
            step = int(win_samples * (1 - overlap))
            n_windows = (n_samples - win_samples) // step + 1
            
            if n_windows < 1:
                return None
                
            # Filter data once for this frequency band
            rot_filt = filter_stream(rot, fl, fu)
            acc_filt = filter_stream(acc, fl, fu)
            
            # Pre-extract all needed components
            rot_z = rot_filt.select(channel='*Z')[0].data
            acc_z = acc_filt.select(channel='*Z')[0].data
            rot_n = rot_filt.select(channel='*N')[0].data
            rot_e = rot_filt.select(channel='*E')[0].data
            acc_n = acc_filt.select(channel='*N')[0].data
            acc_e = acc_filt.select(channel='*E')[0].data
            
            # Pre-compute sin and cos for rotations
            baz_rad = np.radians(baz_values)
            cos_baz = np.cos(baz_rad)
            sin_baz = np.sin(baz_rad)
            
            times = np.zeros(n_windows)
            baz_optimal = np.zeros(n_windows)
            cc_optimal = np.zeros(n_windows)
            cc_matrix = np.zeros((n_windows, len(baz_values)))
            
            for win_idx in range(n_windows):
                i1 = win_idx * step
                i2 = i1 + win_samples
                times[win_idx] = rot_z[i1:i2].mean()  # Approximate time
                
                # Extract window data
                win_rot_n = rot_n[i1:i2]
                win_rot_e = rot_e[i1:i2]
                win_acc_n = acc_n[i1:i2]
                win_acc_e = acc_e[i1:i2]
                
                # Vectorized rotation for all backazimuths
                rot_r = win_rot_n[:, np.newaxis] * cos_baz + win_rot_e[:, np.newaxis] * sin_baz
                rot_t = -win_rot_n[:, np.newaxis] * sin_baz + win_rot_e[:, np.newaxis] * cos_baz
                acc_r = win_acc_n[:, np.newaxis] * cos_baz + win_acc_e[:, np.newaxis] * sin_baz
                acc_t = -win_acc_n[:, np.newaxis] * sin_baz + win_acc_e[:, np.newaxis] * cos_baz
                
                # Compute correlations for all backazimuths at once
                if wave_type.lower() == 'love':
                    signal1 = rot_z[i1:i2]
                    signal2 = acc_t
                elif wave_type.lower() == 'rayleigh':
                    signal1 = rot_t
                    signal2 = acc_z[i1:i2, np.newaxis]
                else:
                    raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
                
                # Vectorized correlation computation
                cc_values = np.array([xcorr_max(correlate(signal1, sig2, 0))[1] for sig2 in signal2.T])
                cc_matrix[win_idx] = cc_values
                
                max_idx = np.argmax(cc_values)
                baz_optimal[win_idx] = baz_values[max_idx]
                cc_optimal[win_idx] = cc_values[max_idx]
            
            return {
                'frequency': {'min': fl, 'max': fu, 'center': fc},
                'times': times,
                'backazimuth': baz_optimal,
                'cc_matrix': cc_matrix,
                'cc_optimal': cc_optimal,
                'window_samples': win_samples,
                'step': step
            }

        # Default frequency bands if not provided
        if fbands is None:
            fbands = {'fmin': 0.01, 'fmax': 1.0, 'octave_fraction': 3}

        # Generate frequency bands
        flower, fupper, fcenter = self.get_octave_bands(
            fmin=fbands['fmin'],
            fmax=fbands['fmax'],
            faction_of_octave=fbands['octave_fraction']
        )
        
        # Get streams and sampling rate
        rot = self.get_stream("rotation", raw=True)
        acc = self.get_stream("translation", raw=True)
        df = self.sampling_rate
        n_samples = len(rot[0].data)
        
        # Generate backazimuth values
        baz_values = np.arange(0, 360, baz_step)
        
        # Process frequency bands in parallel
        freq_params = list(zip(flower, fupper, fcenter))
        process_func = partial(
            process_frequency_band,
            rot=rot,
            acc=acc,
            df=df,
            n_samples=n_samples,
            baz_values=baz_values,
            wave_type=wave_type,
            overlap=overlap,
            twin_min=twin_min
        )
        
        with ThreadPoolExecutor() as executor:
            results_by_freq = list(filter(None, executor.map(process_func, freq_params)))
        
        if not results_by_freq:
            raise ValueError("No valid windows found for any frequency band")

        # Find optimal frequency band for each time window
        n_total_windows = max(len(r['times']) for r in results_by_freq)
        final_times = np.zeros(n_total_windows)
        final_baz = np.zeros(n_total_windows)
        final_fmin = np.zeros(n_total_windows)
        final_fmax = np.zeros(n_total_windows)
        final_fcenter = np.zeros(n_total_windows)
        final_cc = np.zeros(n_total_windows)
        final_velocities = np.zeros(n_total_windows)
        
        # Vectorized optimal parameter selection
        for win_idx in range(n_total_windows):
            cc_values = np.array([
                r['cc_optimal'][win_idx] if win_idx < len(r['cc_optimal']) else -1
                for r in results_by_freq
            ])
            best_freq_idx = np.argmax(cc_values)
            best_result = results_by_freq[best_freq_idx]
            
            if win_idx < len(best_result['times']):
                final_times[win_idx] = best_result['times'][win_idx]
                final_baz[win_idx] = best_result['backazimuth'][win_idx]
                final_fmin[win_idx] = best_result['frequency']['min']
                final_fmax[win_idx] = best_result['frequency']['max']
                final_fcenter[win_idx] = best_result['frequency']['center']
                final_cc[win_idx] = best_result['cc_optimal'][win_idx]
        
        # Compute velocities using optimal parameters (in parallel)
        def compute_velocity_for_window(win_idx):
            if win_idx >= n_total_windows:
                return np.nan
                
            # Filter with optimal parameters
            rot_opt = filter_stream(rot, final_fmin[win_idx], final_fmax[win_idx])
            acc_opt = filter_stream(acc, final_fmin[win_idx], final_fmax[win_idx])
            
            win_samples = int(max(twin_min, 1/final_fcenter[win_idx]) * df)
            step = int(win_samples * (1 - overlap))
            i1 = win_idx * step
            i2 = i1 + win_samples
            
            if i2 > n_samples:
                return np.nan
            
            rot_r, rot_t = rotate_ne_rt(
                rot_opt.select(channel='*N')[0].data[i1:i2],
                rot_opt.select(channel='*E')[0].data[i1:i2],
                final_baz[win_idx]
            )
            acc_r, acc_t = rotate_ne_rt(
                acc_opt.select(channel='*N')[0].data[i1:i2],
                acc_opt.select(channel='*E')[0].data[i1:i2],
                final_baz[win_idx]
            )
            
            if wave_type.lower() == 'love':
                vel_result = self.compute_odr(
                    x_array=rot_opt.select(channel='*Z')[0].data[i1:i2],
                    y_array=0.5*acc_t,
                    zero_intercept=True
                )
            else:  # rayleigh
                vel_result = self.compute_odr(
                    x_array=rot_t,
                    y_array=acc_opt.select(channel='*Z')[0].data[i1:i2],
                    zero_intercept=True
                )
            
            return vel_result['slope']
        
        with ThreadPoolExecutor() as executor:
            final_velocities = np.array(list(executor.map(compute_velocity_for_window, range(n_total_windows))))
        
        return {
            'times': final_times,
            'backazimuth': {
                'values': baz_values,
                'optimal': final_baz
            },
            'frequency': {
                'min': final_fmin,
                'max': final_fmax,
                'center': final_fcenter,
                'bands': {
                    'lower': flower,
                    'upper': fupper,
                    'center': fcenter
                }
            },
            'cross_correlation': {
                'optimal': final_cc
            },
            'velocity': final_velocities,
            'parameters': {
                'wave_type': wave_type,
                'overlap': overlap,
                'sampling_rate': df
            }
        }

    def regression(self, x_data: np.ndarray, y_data: np.ndarray, method: str = "odr", 
                   zero_intercept: bool = True, verbose: bool = False) -> Dict:
        """
        Perform regression analysis using various methods.
        
        Parameters:
        -----------
        x_data : np.ndarray
            Input data (e.g., rotation data)
        y_data : np.ndarray
            Target data (e.g., translation data)
        method : str, optional
            Regression method ('odr', 'ransac', 'theilsen', 'ols'), by default 'odr'
        zero_intercept : bool, optional
            Force intercept through zero if True, by default True
        verbose : bool, optional
            Print regression results if True, by default False
        
        Returns:
        --------
        Dict
            Dictionary containing:
            - slope: Regression slope
            - intercept: Y-intercept (0 if zero_intercept=True)
            - r_squared: R-squared value
            - method: Method used
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
        from scipy import odr
        from scipy.stats import pearsonr
        
        # Validate inputs
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same length")
        
        if len(x_data) < 2:
            raise ValueError("Need at least 2 data points for regression")
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 2:
            raise ValueError("Not enough valid data points after removing NaN values")
        
        # Reshape for sklearn compatibility
        X = x_clean.reshape(-1, 1)
        y = y_clean
        
        # Initialize results
        result = {
            'slope': np.nan,
            'intercept': 0.0 if zero_intercept else np.nan,
            'r_squared': np.nan,
            'method': method.lower()
        }
        
        try:
            if method.lower() == "odr":
                # Orthogonal Distance Regression
                def linear_func(B, x):
                    if zero_intercept:
                        return B[0] * x
                    else:
                        return B[0] * x + B[1]
                
                # Create ODR model
                model = odr.Model(linear_func)
                
                # Estimate uncertainties
                sx = np.std(x_clean) * np.ones_like(x_clean)
                sy = np.std(y_clean) * np.ones_like(y_clean)
                
                # Create ODR data object
                data = odr.RealData(x_clean, y_clean, sx=sx, sy=sy)
                
                # Set initial parameters
                if zero_intercept:
                    beta0 = [np.mean(y_clean) / np.mean(x_clean)]
                else:
                    beta0 = [np.mean(y_clean) / np.mean(x_clean), 0.0]
                
                # Fit model
                odr_obj = odr.ODR(data, model, beta0=beta0)
                output = odr_obj.run()
                
                result['slope'] = output.beta[0]
                if not zero_intercept:
                    result['intercept'] = output.beta[1]
                
                # Calculate R-squared
                if zero_intercept:
                    y_pred = output.beta[0] * x_clean
                else:
                    y_pred = output.beta[0] * x_clean + output.beta[1]
                
                # Calculate R-squared
                r, _ = pearsonr(x_clean, y_clean)
                result['r_squared'] = r**2

                if verbose:
                    print(f"ODR Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
            
            elif method.lower() == "ransac":
                # RANSAC regression
                try:
                    model = RANSACRegressor(
                        estimator=LinearRegression(fit_intercept=not zero_intercept),
                        random_state=42,
                        max_trials=1000
                    ).fit(X, y)
                except TypeError:
                    # Fallback for older sklearn versions
                    model = RANSACRegressor(
                        base_estimator=LinearRegression(fit_intercept=not zero_intercept),
                        random_state=42,
                        max_trials=1000
                    ).fit(X, y)
                
                result['slope'] = model.estimator_.coef_[0]
                if not zero_intercept:
                    result['intercept'] = model.estimator_.intercept_
                result['r_squared'] = model.score(X, y)
                
                if verbose:
                    print(f"RANSAC Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
            
            elif method.lower() == "theilsen":
                # Theil-Sen regression
                model = TheilSenRegressor(fit_intercept=not zero_intercept, random_state=42).fit(X, y)
                
                result['slope'] = model.coef_[0]
                if not zero_intercept:
                    result['intercept'] = model.intercept_
                result['r_squared'] = model.score(X, y)
                
                if verbose:
                    print(f"Theil-Sen Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
            
            elif method.lower() == "ols":
                # Ordinary Least Squares
                model = LinearRegression(fit_intercept=not zero_intercept).fit(X, y)
                
                result['slope'] = model.coef_[0]
                if not zero_intercept:
                    result['intercept'] = model.intercept_
                result['r_squared'] = model.score(X, y)
                
                if verbose:
                    print(f"OLS Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
            
            else:
                raise ValueError(f"Invalid method: {method}. Use 'odr', 'ransac', 'theilsen', or 'ols'")
        
        except Exception as e:
            if verbose:
                print(f"Regression failed: {str(e)}")
            # Return NaN values if regression fails
            result['slope'] = np.nan
            result['intercept'] = np.nan if not zero_intercept else 0.0
            result['r_squared'] = np.nan
        
        return result

    def rotate_romy_zne(self, st: Stream, inv: Inventory, use_components: List[str] = ["Z", "U", "V"], keep_z: bool = True) -> Stream:
        """
        Rotate ROMY data from specified components to ZNE orientation
        
        Parameters
        ----------
        st : Stream
            Input stream containing ROMY components
        inv : Inventory
            Station inventory with orientation information
        use_components : List[str]
            Components to use for rotation (default: ["Z", "U", "V"])
        keep_z : bool
            Whether to keep original Z component (default: True)
        
        Returns
        -------
        Stream
            Rotated stream with ZNE components
        """
        
        from obspy.signal.rotate import rotate2zne

        if not inv:
            raise ValueError("Inventory required for rotation")

        locs = {"Z":"10", "U":"", "V":"", "W":""}

        # Make dictionary for components with data, azimuth and dip
        components = {}
        for comp in use_components:
            loc = locs[comp]
            try:
                tr = st.select(component=comp)
                if not tr:
                    raise ValueError(f"Component {comp} not found in stream")

                orientation = inv.get_orientation(f"BW.ROMY.{loc}.BJ{comp}")

                if not orientation:
                    raise ValueError(f"Could not get orientation for component {comp}")
                    
                components[comp] = {
                    'data': tr[0].data,
                    'azimuth': orientation['azimuth'],
                    'dip': orientation['dip']
                }
            except Exception as e:
                print(f"Warning: Error processing component {comp}: {e}")
                return st

        # Rotate to ZNE
        try:
            romy_z, romy_n, romy_e = rotate2zne(
                components[use_components[0]]['data'], 
                components[use_components[0]]['azimuth'], 
                components[use_components[0]]['dip'],
                components[use_components[1]]['data'], 
                components[use_components[1]]['azimuth'], 
                components[use_components[1]]['dip'],
                components[use_components[2]]['data'], 
                components[use_components[2]]['azimuth'], 
                components[use_components[2]]['dip'],
                inverse=False
            )
        except Exception as e:
            print(f"Warning: Rotation failed: {e}")
            return st

        # Create new stream with ZNE components
        st_new = st.copy()

        # Update channel codes and data
        try:
            for c, tr in zip(['Z', 'N', 'E'], st_new):
                tr.stats.channel = tr.stats.channel[:2] + c
            
            if keep_z and 'Z' in use_components:
                st_new.select(component='Z')[0].data = components['Z']['data']
            else:
                st_new.select(component='Z')[0].data = romy_z
            
            # set location to empty string
            for tr in st_new:
                tr.stats.location = ""
            
            # set data for channel N and E
            st_new.select(component='N')[0].data = romy_n
            st_new.select(component='E')[0].data = romy_e
        
        except Exception as e:
            print(f"Warning: Error updating rotated data: {e}")
            return st

        return st_new

    def compute_backazimuth_map(self, wave_types=['love', 'rayleigh'], 
                                baz_step=1, baz_win_sec=20.0, 
                                baz_win_overlap=0.5, cc_threshold=0.5, cc_method='mid',
                                tangent_components="rotation"):
        """
        Compute backazimuth estimates for surface waves and tangent methods
        
        Parameters:
        -----------
        wave_types : list
            Types of waves to analyze: 'love', 'rayleigh', 'tangent'
        baz_step : float
            Step size for backazimuth search in degrees
        baz_win_sec : float
            Time window length for correlation analysis in seconds
        baz_win_overlap : float
            Overlap between time windows (0.0 to 1.0)
        cc_threshold : float
            Minimum correlation coefficient threshold
        tangent_components : str
            Components to use for tangent method ('rotation' or 'acceleration')
        cc_method : str
            Value to use for correlation coefficient ('max', 'mid', None)
        Returns:
        --------
        dict : Results dictionary with backazimuth estimates for each wave type
        """
        from numpy import arange, average, isnan, any as npany
        import numpy as np
        import scipy.stats as sts
        
        # Get station coordinates
        if hasattr(self, 'station_latitude') and hasattr(self, 'station_longitude'):
            station_lat = self.station_latitude
            station_lon = self.station_longitude
        else:
            print("Warning: Station coordinates not available")
            station_lat, station_lon = 0.0, 0.0
        
        results = {}
        baz_estimates = {}
        
        # Process each wave type
        for wave_type in wave_types:
            if wave_type.lower() in ['love', 'rayleigh']:

                if wave_type in self.baz_results.keys():
                    print(f"Using precomputed {wave_type} backazimuth results")
                    baz_result = self.baz_results[wave_type]
                else:
                    print(f"Computing {wave_type} wave backazimuth...")
                    baz_result = self.compute_backazimuth(
                        wave_type=wave_type,
                        baz_step=baz_step,
                        baz_win_sec=baz_win_sec,
                        baz_win_overlap=baz_win_overlap,
                        out=True
                    )

                # get times
                time = baz_result['twin_center']

                # select maximal or mid approach results
                if cc_method == 'mid':
                    ccc = baz_result['cc_mid']
                    baz = baz_result['baz_mid']
                elif cc_method == 'max' or cc_method is None:
                    ccc = baz_result['cc_max']
                    baz = baz_result['baz_max']
                
                # apply cc threshold if provided
                if cc_threshold is not None:
                    mask = ccc > cc_threshold
                    time = time[mask]
                    baz = baz[mask]
                    cc = ccc[mask]
                else:
                    baz = baz
                    cc = ccc

                # prepare results
                if np.any(mask):
                    results[wave_type] = {
                        'baz': baz,
                        'cc': cc,
                        'time': time,
                        'n_samples': 0  # Will be updated if KDE is computed
                    }

                # remove NaN values
                valid_mask = ~(isnan(baz) | isnan(cc))
                if npany(valid_mask):
                    baz_valid = baz[valid_mask]
                    cc_valid = cc[valid_mask]

                # compute kde if enough data
                if len(baz_valid) > 5:
                    kde_stats = self.get_kde_stats(baz_valid, cc_valid, _baz_steps=0.5, Ndegree=60, plot=False)
                    baz_estimates[wave_type] = kde_stats['baz_estimate']
                    # Store n_samples in results
                    if wave_type in results:
                        results[wave_type]['n_samples'] = kde_stats.get('n_samples', 0)
                else:
                    baz_estimates[wave_type] = baz[0]
                    # Store n_samples as 0 if not enough data
                    if wave_type in results:
                        results[wave_type]['n_samples'] = 0
 
            elif wave_type.lower() == 'tangent':

                if wave_type in self.baz_results.keys():
                    print(f"Using precomputed {wave_type} backazimuth results")
                    baz_result = self.baz_results[wave_type]
                else:
                    print(f"Computing tangent backazimuth with {tangent_components} components...")
                    baz_result = self.compute_backazimuth(
                        wave_type="tangent",
                        tangent_components=tangent_components,
                        baz_step=baz_step,
                        baz_win_sec=baz_win_sec,
                        baz_win_overlap=baz_win_overlap,
                        out=True
                    )

                # get times
                time = baz_result['twin_center']

                # select results
                ccc = baz_result['cc_max']
                baz = baz_result['baz_max']
            
                # apply cc threshold if provided
                if cc_threshold is not None:
                    mask = ccc > cc_threshold
                    time = time[mask]
                    baz = baz[mask]
                    cc = ccc[mask]
                else:
                    baz = baz
                    cc = ccc

                # prepare results
                if np.any(mask):
                    results[wave_type] = {
                        'baz': baz,
                        'cc': cc,
                        'time': time,
                        'n_samples': 0  # Will be updated if KDE is computed
                    }

                # remove NaN values
                valid_mask = ~(isnan(baz) | isnan(cc))
                if npany(valid_mask):
                    baz_valid = baz[valid_mask]
                    cc_valid = cc[valid_mask]

                    if len(baz_valid) > 5:

                        # Compute KDE and find maximum and index
                        kde_stats = self.get_kde_stats(baz_valid, cc_valid, _baz_steps=0.5, Ndegree=60, plot=False)

                        kde = kde_stats['kde_values']
                        baz_estimates[wave_type] = kde_stats['baz_estimate']
                        
                        # Store n_samples in results
                        results[wave_type]['n_samples'] = kde_stats.get('n_samples', 0)

                        results[wave_type]['baz'] = baz_valid
                        results[wave_type]['cc'] = cc_valid
                    else:
                        print(f"No valid tangent data")
                        # Store n_samples as 0 if not enough data
                        if wave_type in results:
                            results[wave_type]['n_samples'] = 0
                        continue
                else:
                    print(f"No tangent data available")
                    continue

            else:
                print(f"Unknown wave type: {wave_type}")
                continue
        
        if not results:
            print("No valid results for any wave type")
            return {}
        
        # Compile final results
        final_results = {
            'estimates': baz_estimates,
            'detailed_results': results,
            'station_coordinates': {
                'latitude': station_lat,
                'longitude': station_lon
            },
            'parameters': {
                'baz_step': baz_step,
                'baz_win_sec': baz_win_sec,
                'baz_win_overlap': baz_win_overlap,
                'cc_threshold': cc_threshold,
                'tangent_components': tangent_components
            }
        }
        
        return final_results

    def compute_backazimuth(self, wave_type: str="love", baz_step: int=1, baz_win_sec: float=30.0, 
                        rotation_data: Stream=None, translation_data: Stream=None,
                        baz_win_overlap: float=0.5, tangent_components: str="rotation", verbose: bool=False,
                        out: bool=False, cc_threshold: float=0.0, cc_method: str="both") -> Dict:
        """
        Fast backazimuth estimation using two-stage grid search:
        1. Coarse search with 10° steps to find approximate maximum
        2. Fine search with 1° steps around the maximum
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love', 'rayleigh', or 'tangent')
        baz_step : int
            Final step size in degrees for backazimuth search (default: 1)
        baz_win_sec : float
            Length of backazimuth estimation windows in seconds (default: 30.0)
        rotation_data : Stream, optional
            Rotation data stream (if None, uses self.get_stream("rotation"))
        translation_data : Stream, optional
            Translation data stream (if None, uses self.get_stream("translation"))
        baz_win_overlap : float
            Overlap between windows as fraction (0-1) (default: 0.5)
        tangent_components : str
            Components to use for tangent method ('rotation' or 'acceleration')
        verbose : bool
            Print progress information
        out : bool
            Return detailed output dictionary if True
        cc_threshold : float
            Minimum correlation coefficient threshold (default: 0.0)
        cc_method : str
            Method to use for correlation coefficient ('max', 'mid', 'both') (default: 'both')
        Returns:
        --------
        Dict : Backazimuth estimation results
        """
        import scipy.stats as sts
        from obspy.signal.rotate import rotate_ne_rt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import linspace, ones, array, nan, meshgrid, arange, zeros, cov, pi, arctan
        from numpy.linalg import eigh
        from numpy import argsort

        def _padding(_baz, _ccc, _baz_steps, Ndegree=60):
            # get lower and upper array that is padded
            _baz_lower = np.arange(-Ndegree, 0, _baz_steps)
            _baz_upper = np.arange(max(_baz)+_baz_steps, max(_baz)+Ndegree, _baz_steps)

            # pad input baz array
            _baz_pad = np.append(np.append(_baz_lower, _baz), _baz_upper)

            # get sampled size
            Nsample = int(Ndegree/_baz_steps)

            # pad ccc array by  asymetric reflection
            _ccc_pad = np.append(np.append(_ccc[-Nsample-1:-1], _ccc), _ccc[1:Nsample])

            return _baz_pad, _ccc_pad

        def _get_zero_crossings(arr):
            # get nullstellen by sign function and then the difference
            nullstellen = np.diff(np.sign(arr))

            # there should only be one from negative to positive
            # this is a positive value
            nullstelle1 = np.argmax(nullstellen)

            # look for second zero crossing after the first one
            shift = nullstelle1+1
            nullstelle2 = np.argmax(abs(nullstellen[shift:]))+ shift

            return nullstelle1, nullstelle2

        def _get_fine_grid(backazimuth, search_range, baz_step):
            """
            Generate fine grid search around given backazimuths.
            
            Parameters:
            -----------
            backazimuths : array-like
                Backazimuth values to search around
            search_range : float
                Range in degrees around each backazimuth
            baz_step : float
                Step size in degrees for the fine grid
                
            Returns:
            --------
            numpy.ndarray
                Fine grid backazimuths around the input backazimuths
            """            
            # Create range around this backazimuth
            fine_start = backazimuth - search_range
            fine_end = backazimuth + search_range
            
            # Ensure we stay within 0-360 range
            if fine_start < 0:
                fine_start += 360
            if fine_end > 360:
                fine_end -= 360
            
            # Create fine search backazimuths
            if fine_start < fine_end:
                fine_baz = np.arange(fine_start, fine_end + baz_step, baz_step)
            else:
                # Handle wrap-around case
                fine_baz1 = np.arange(fine_start, 360, baz_step)
                fine_baz2 = np.arange(0, fine_end + baz_step, baz_step)
                fine_baz = np.concatenate([fine_baz1, fine_baz2])
            
                # Normalize to 0-360
                fine_baz = fine_baz % 360
            
            # Remove duplicates and sort
            return fine_baz

        def _compute_correlation_for_baz(backazimuth_val, idx1, idx2, wave_type, ACC, ROT):
            """Compute correlation for a specific backazimuth value"""
            try:
                if wave_type.lower() == "love":
                    # rotate NE to RT
                    HR, HT = rotate_ne_rt(
                        ACC.select(channel='*N')[0].data,
                        ACC.select(channel='*E')[0].data,
                        backazimuth_val
                    )

                    JZ = ROT.select(channel="*Z")[0].data

                    # compute correlation for backazimuth
                    ccorr = correlate(
                        JZ[idx1:idx2],
                        HT[idx1:idx2],
                        0, demean=True, normalize='naive', method='auto'
                    )

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)
                    return cc_max

                elif wave_type.lower() == "rayleigh":
                    # rotate NE to RT
                    JR, JT = rotate_ne_rt(
                        ROT.select(channel='*N')[0].data,
                        ROT.select(channel='*E')[0].data,
                        backazimuth_val
                    )

                    HZ = ACC.select(channel="*Z")[0].data

                    # compute correlation for backazimuth
                    ccorr = correlate(
                        HZ[idx1:idx2],
                        JT[idx1:idx2],
                        0, demean=True, normalize='naive', method='auto'
                    )

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)
                    return cc_max

                else:
                    return nan

            except Exception as e:
                if verbose:
                    print(f"Error computing correlation for baz={backazimuth_val}: {e}")
                return nan

        # Check config keywords
        keywords = ['tbeg', 'tend', 'sampling_rate',
                    'station_latitude', 'station_longitude']

        for key in keywords:
            if key not in self.attributes():
                print(f" -> {key} is missing in config!\n")
                return {}  # Return empty dict instead of None

        # Store window parameters and ensure baz_step is integer
        self.baz_step = int(baz_step)
        self.baz_win_sec = baz_win_sec
        self.baz_win_overlap = baz_win_overlap

        # Prepare streams
        if rotation_data is None and translation_data is None:
            ACC = self.get_stream("translation").copy()
            ROT = self.get_stream("rotation").copy()
        elif rotation_data is not None and translation_data is not None:
            ACC = translation_data.copy()
            ROT = rotation_data.copy()
        else:
            raise ValueError("no rotation or translation data provided")

        # if wave_type.lower() == "tangent":
        #     # revert polarity if applied
        #     if hasattr(self, 'pol_applied') and self.pol_applied:
        #         if hasattr(self, 'pol_dict') and self.pol_dict is not None:
        #             for tr in ACC.select(channel="*Z"):
        #                 if tr.stats.channel[1:] in self.pol_dict:
        #                     tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]
        #             for tr in ROT.select(channel="*Z"):
        #                 if tr.stats.channel[1:] in self.pol_dict:
        #                     tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]

        # sampling rate
        df = ROT[0].stats.sampling_rate

        # Get amount of samples for data
        n_data = min([len(tr.data) for tr in ROT])

        # Calculate window parameters
        win_samples = int(baz_win_sec * df)
        overlap_samples = int(win_samples * baz_win_overlap)
        step = win_samples - overlap_samples

        if step == 0:
            print("step is 0, setting to 1")
            step = 1

        # get amount of windows
        n_windows = int((n_data - win_samples) / step) + 1

        # Prepare final backazimuths for fine search
        final_backazimuths = linspace(0, 360 - self.baz_step, int(360 / self.baz_step))
        
        # Prepare data array for final results
        corrbaz = ones([final_backazimuths.size, n_windows])*nan
        
        if verbose:
            print(f"  Final backazimuths: {len(final_backazimuths)} points")
            print(f"  Correlation array shape: {corrbaz.shape}")
            print(f"  Number of windows: {n_windows}")

        degrees = []
        windows = []
        t_center = []
        bazs = ones(n_windows)*nan
        
        # Initialize arrays based on cc_method
        if cc_method in ['max', 'both']:
            maxbaz = np.zeros(n_windows)
            maxcorr = np.zeros(n_windows)
        else:
            maxbaz = np.full(n_windows, np.nan)
            maxcorr = np.full(n_windows, np.nan)
            
        if cc_method in ['mid', 'both']:
            midbaz = np.zeros(n_windows)
            midcorr = np.zeros(n_windows)
        else:
            midbaz = np.full(n_windows, np.nan)
            midcorr = np.full(n_windows, np.nan)

        if verbose:
            print(f"Fast backazimuth estimation: {wave_type} waves")
            print(f"  Coarse search: 10° steps")
            print(f"  Fine search: {baz_step}° steps")
            print(f"  Windows: {n_windows}")

        # _______________________________
        # Two-stage backazimuth estimation
        # _______________________________
        
        if wave_type.lower() == "tangent":
            # For tangent waves, use the original method (no grid search)
            if verbose:
                print(f" > using {wave_type} for backazimuth estimation with {tangent_components} components...")

            # loop over time windows only
            for i_win in range(0, n_windows):
                windows.append(i_win)

                # update indices
                idx1 = i_win * step
                idx2 = idx1 + win_samples

                # get central time of window
                t_center.append((idx1 + (idx2 - idx1)/2) /df)

                try:
                    N = len(ROT[0].data[idx1:idx2])
                except:
                    N = len(ACC[0].data[idx1:idx2])

                # prepare data based on component choice
                dat = zeros((N, 2))

                if tangent_components.lower() == "rotation":
                    # Use rotation components (original method)
                    dat[:, 0] = ROT.select(channel='*E')[0].data[idx1:idx2]
                    dat[:, 1] = ROT.select(channel='*N')[0].data[idx1:idx2]
                elif tangent_components.lower() == "acceleration":
                    # Use acceleration components (new option)
                    dat[:, 0] = ACC.select(channel='*E')[0].data[idx1:idx2]
                    dat[:, 1] = ACC.select(channel='*N')[0].data[idx1:idx2]
                else:
                    raise ValueError(f"Invalid tangent_components: {tangent_components}. Use 'rotation' or 'acceleration'")

                # compute covariance
                covar = cov(dat, rowvar=False)

                # get dominant eigenvector
                Cprime, Q = eigh(covar, UPLO='U')

                # sorting
                loc = argsort(abs(Cprime))[::-1]

                # formatting
                Q = Q[:, loc]

                # get backazimuth using tangent of eigenvectors
                baz0 = -arctan((Q[1, 0]/Q[0, 0]))*180/pi

                # if negative due to tangent, then add 180 degrees
                if baz0 <= 0:
                    baz0 += 180

                # remove 180° ambiguity using appropriate correlation
                if tangent_components.lower() == "rotation":
                    # Original method: rotate rotation components and correlate with acceleration Z
                    JR, JT = rotate_ne_rt(
                        ROT.select(channel='*N')[0].data,
                        ROT.select(channel='*E')[0].data,
                        baz0
                    )
                    
                    HZ = ACC.select(channel="*Z")[0].data

                    # correlate with acceleration
                    ccorr = correlate(
                        HZ[idx1:idx2],
                        JT[idx1:idx2],
                        0, demean=True, normalize='naive', method='auto'
                    )

                # remove 180° ambiguity using closest to theoretical backazimuth for acceleration components
                else:
                    if self.event_info and 'backazimuth' in self.event_info:
                        # compute rmse of baz0 and self.event_info['backazimuth']
                        rmse = np.sqrt(np.mean((baz0 - self.event_info['backazimuth'])**2))
                        
                        # compute rmse of baz0 + 180 and self.event_info['backazimuth']
                        rmse_180 = np.sqrt(np.mean((baz0 + 180 - self.event_info['backazimuth'])**2))

                        # choose the one with the least rmse and set cc_max to 0 (dummy value)
                        if rmse_180 < rmse:
                            ccorr = [1] #[1e-6]
                        else:
                            ccorr = [-1] #[1e-6]

                # get maximum correlation
                xshift, cc_max = xcorr_max(ccorr)

                # if correlation positive add 180 degrees
                if cc_max < 0:  # cc_max > 0: #  changed after removing the polarity reversal above 
                    baz0 += 180

                # take absolute value of correlation for better visualization
                cc_max = abs(cc_max)

                bazs[i_win] = baz0
                corrbaz[0, i_win] = cc_max

        # if wave_type is rayleigh or love
        else:
            # For Love and Rayleigh waves, use two-stage grid search
            if verbose:
                print(f" > using {wave_type} waves for backazimuth estimation ...")

            # Stage 1: Coarse search with 10° steps
            coarse_step = 10
            coarse_backazimuths = linspace(0, 360 - coarse_step, int(360 / coarse_step))
            
            if verbose:
                print(f"  Stage 1: Coarse search with {coarse_step}° steps ({len(coarse_backazimuths)} points)")

            # loop over time windows
            for i_win in range(0, n_windows):
                windows.append(i_win)
            
                # Initialize output
                maxbaz[i_win] = np.nan
                maxcorr[i_win] = np.nan
                midbaz[i_win] = np.nan
                midcorr[i_win] = np.nan
                
                # update indices
                idx1 = i_win * step
                idx2 = idx1 + win_samples

                # get central time of window
                t_center.append((idx1 + (idx2 - idx1)/2) /df)
                
                if verbose and i_win < 3:  # Show first few windows
                    print(f"  Processing window {i_win}: indices {idx1}-{idx2}")

                # Coarse search
                coarse_correlations = []
                for coarse_baz in coarse_backazimuths:
                    cc_max = _compute_correlation_for_baz(coarse_baz, idx1, idx2, wave_type, ACC, ROT)
                    coarse_correlations.append(cc_max)

                # Find best coarse backazimuth
                best_coarse_idx = np.argmax(coarse_correlations)
                best_coarse_baz = coarse_backazimuths[best_coarse_idx]
                best_coarse_cc = coarse_correlations[best_coarse_idx]

                if verbose and i_win < 3:
                    print(f"    Window {i_win}: Best coarse baz = {best_coarse_baz:.1f}° (cc = {best_coarse_cc:.3f})")

                # Stage 2: Fine search around the best coarse result
                if cc_method in ['max', 'both']:
                    try:
                        search_range = 25  # degrees around the coarse maximum
                        fine_start = best_coarse_baz - search_range
                        fine_end = best_coarse_baz + search_range
                        
                        # Ensure we stay within 0-360 range
                        if fine_start < 0:
                            fine_start += 360
                        if fine_end > 360:
                            fine_end -= 360

                        # Create fine search backazimuths using helper function
                        fine_backazimuths = _get_fine_grid(best_coarse_baz, search_range, baz_step)

                        # Fine search
                        fine_correlations = []
                        for fine_baz in fine_backazimuths:
                            cc_max = _compute_correlation_for_baz(fine_baz, idx1, idx2, wave_type, ACC, ROT)
                            fine_correlations.append(cc_max)

                        # Find best fine backazimuth
                        best_fine_idx = np.argmax(fine_correlations)
                        best_fine_baz = fine_backazimuths[best_fine_idx]
                        best_fine_cc = fine_correlations[best_fine_idx]

                        # Store max result
                        final_idx = np.argmin(np.abs(final_backazimuths - best_fine_baz))
                        if final_idx < len(final_backazimuths) and i_win < corrbaz.shape[1]:
                            corrbaz[final_idx, i_win] = best_fine_cc
                        
                        # Store max values
                        maxbaz[i_win] = best_fine_baz
                        maxcorr[i_win] = best_fine_cc
                
                    except Exception as e:
                        print(f"Error in max calculation: {e}")

                # Stage 3: Calculate mid backazimuth if mid method is requested
                if cc_method in ['mid', 'both']:
                    try:
                        # Use zero crossing approach
                        baz_pad, cc_pad = _padding(coarse_backazimuths, np.array(coarse_correlations), coarse_step, Ndegree=60)
                        null1, null2 = _get_zero_crossings(cc_pad)
                        
                        # Calculate mid backazimuth as center between zero crossings
                        baz_lower = baz_pad[null1]
                        baz_upper = baz_pad[null2]
                        # baz_mid = (baz_upper - baz_lower) / 2 + baz_lower
                        
                        # Create fine search backazimuths around zero crossings using helper function
                        search_range = 25  # degrees around the zero crossings
                        fine_baz_lower = _get_fine_grid(baz_lower, search_range, baz_step)
                        fine_baz_upper = _get_fine_grid(baz_upper, search_range, baz_step)

                        # Calculate correlations for fine grid
                        fine_cc_lower = []
                        for fine_baz in fine_baz_lower:
                            cc_max = _compute_correlation_for_baz(fine_baz, idx1, idx2, wave_type, ACC, ROT)
                            fine_cc_lower.append(cc_max)
                        fine_cc_upper = []
                        for fine_baz in fine_baz_upper:
                            cc_max = _compute_correlation_for_baz(fine_baz, idx1, idx2, wave_type, ACC, ROT)
                            fine_cc_upper.append(cc_max)

                        # Find zero crossings in fine grid
                        fine_null_lower_idx = np.argmin(np.abs(fine_cc_lower))
                        fine_null_upper_idx = np.argmin(np.abs(fine_cc_upper))

                        # Calculate final mid backazimuth as center between fine zero crossings
                        fine_baz_lower = fine_baz_lower[fine_null_lower_idx]
                        fine_baz_upper = fine_baz_upper[fine_null_upper_idx]

                        # handle wrap-around
                        if fine_baz_lower > fine_baz_upper:
                            fine_baz_upper += 360

                        # Get mid backazimuth
                        baz_mid = (fine_baz_upper - fine_baz_lower) / 2 + fine_baz_lower

                        # handle wrap-around
                        baz_mid = baz_mid % 360

                        # Get correlation at mid point
                        cc_mid = _compute_correlation_for_baz(baz_mid, idx1, idx2, wave_type, ACC, ROT)

                        # Normalize to 0-360
                        if baz_mid < 0:
                            baz_mid += 360
                        if baz_mid > 360:
                            baz_mid -= 360
                                                
                        # Store mid result
                        midbaz[i_win] = baz_mid
                        midcorr[i_win] = cc_mid
                        
                    except Exception as e:
                        print(f"Error in mid calculation: {e}")

        # Handle tangent waves
        if wave_type.lower() == "tangent":
            if cc_method in ['max', 'both']:
                maxbaz = bazs
                maxcorr = corrbaz[0, :]
            if cc_method in ['mid', 'both']:
                midbaz = bazs.copy()
                midcorr = corrbaz[0, :].copy()

        # For Love/Rayleigh waves, max and mid values are already calculated in the main loop
        # No additional extraction needed since we store them directly during computation

        # create mesh grid
        t_win = arange(0, baz_win_sec*n_windows+baz_win_sec, baz_win_sec)
        t_win = t_win[:-1]+baz_win_sec/2
        grid = meshgrid(t_win, final_backazimuths)

        # add one element for axes
        if len(windows) > 0:
            windows.append(windows[-1]+1)
        if len(degrees) > 0:
            degrees.append(degrees[-1]+self.baz_step)
        else:
            degrees.append(self.baz_step)

        # prepare results
        results = {
            'baz_mesh': grid,
            'baz_corr': corrbaz,
            'baz_time': t_win,
            'acc': ACC,
            'rot': ROT,
            'twin_center': np.array(t_center),
            'cc_max_y': maxbaz,
            'baz_max': maxbaz,
            'cc_max': maxcorr,
            'baz_mid': midbaz,
            'cc_mid': midcorr,
            'component_type': tangent_components if wave_type.lower() == "tangent" else None,
            'parameters': {
                'baz_win_sec': baz_win_sec,
                'baz_win_overlap': baz_win_overlap,
                'baz_step': baz_step,
                'wave_type': wave_type,
                'cc_method': cc_method,
            }
        }

        # add results to object
        self.baz_results[wave_type] = results

        # return output if out required
        if out:
            return results

    def compute_velocities(self, wave_type: str="love", win_time_s: float=None, overlap: float=0.5, 
                          cc_threshold: float=0.2, baz: float=None, method: str='odr') -> Dict:
        """
        Compute phase velocities in time intervals for Love or Rayleigh waves
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love' or 'rayleigh')
        win_time_s : float or None
            Window length in seconds. If None, uses 1/fmin
        overlap : float
            Window overlap in percent (0-1)
        cc_threshold : float
            Minimum cross-correlation coefficient threshold
        baz : float or None
            Backazimuth in degrees. If None, uses theoretical or estimated BAZ
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - time: array of time points
            - velocity: array of phase velocities
            - ccoef: array of cross-correlation coefficients
            - terr: array of time window lengths
        """
        import numpy as np
        from obspy.signal.rotate import rotate_ne_rt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        
        # Get and process streams
        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()
        
        # Get sampling rate
        df = self.sampling_rate
        
        # Set window length if not provided
        if win_time_s is None:
            win_time_s = 1/self.fmin
            
        # Get backazimuth if not provided
        if baz is None:
            if hasattr(self, 'theoretical_baz'):
                print(f"Using theoretical BAZ {self.theoretical_baz}")
                baz = self.theoretical_baz
            elif hasattr(self, 'event_info') and 'backazimuth' in self.event_info:
                print(f"Using event BAZ {self.event_info['backazimuth']}")
                baz = self.event_info['backazimuth']
            else:
                raise ValueError("No backazimuth provided or available")
        
        # Get Z component and rotate components to radial-transverse
        if wave_type == 'rayleigh':
            acc_z = acc.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(rot.select(channel='*N')[0].data,
                                    rot.select(channel='*E')[0].data,
                                    baz)
            n_samples = len(acc_z)

        elif wave_type == 'love':
            rot_z = rot.select(channel="*Z")[0].data
            acc_r, acc_t = rotate_ne_rt(acc.select(channel='*N')[0].data,
                                    acc.select(channel='*E')[0].data,
                                    baz)
            n_samples = len(rot_z)

        # Calculate window parameters
        win_samples = int(win_time_s * df)
        overlap_samples = int(win_samples * overlap)
        step = win_samples - overlap_samples

        n_windows = int((n_samples - win_samples) / step) + 1
        
        # Initialize arrays
        times = np.zeros(n_windows)
        velocities = np.zeros(n_windows)
        cc_coeffs = np.zeros(n_windows)
        
        # Loop through windows
        for i in range(n_windows):
            i1 = i * step
            i2 = i1 + win_samples
            
            # compute cross-correlation coefficient
            if wave_type.lower() == 'love':
                # For Love waves: use transverse acceleration and vertical rotation
                cc = xcorr_max(correlate(rot_z[i1:i2], acc_t[i1:i2], 0))[1]

            elif wave_type.lower() == 'rayleigh':
                # For Rayleigh waves: use vertical acceleration and transverse rotation
                cc = xcorr_max(correlate(rot_t[i1:i2], acc_z[i1:i2], 0))[1]
            else:
                raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")

            # Compute velocity using amplitude ratio
            if abs(cc) > cc_threshold:
                if wave_type.lower() == 'love':
                    # get velocity from amplitude ratio via regression
                    # if method.lower() == 'odr':
                    #     velocities[i] = self.compute_odr(rot_z[i1:i2], 0.5*acc_t[i1:i2])['slope']
                    # elif method.lower() == 'ransac':
                    #     velocities[i] = self.compute_regression(rot_z[i1:i2], 0.5*acc_t[i1:i2], method='ransac', zero_intercept=True)['slope']
                    reg_result = self.regression(
                        rot_z[i1:i2],
                        0.5*acc_t[i1:i2],
                        method=method.lower(),
                        zero_intercept=True,
                        verbose=False
                    )
                    velocities[i] = reg_result['slope']
                elif wave_type.lower() == 'rayleigh':
                    # get velocity from amplitude ratio via regression
                    # if method.lower() == 'odr':
                    #     velocities[i] = self.compute_odr(rot_t[i1:i2], acc_z[i1:i2])['slope']
                    # elif method.lower() == 'ransac':
                    #     velocities[i] = self.compute_regression(rot_t[i1:i2], acc_z[i1:i2], method='ransac', zero_intercept=True)['slope']
                    reg_result = self.regression(
                        rot_t[i1:i2],
                        acc_z[i1:i2],
                        method=method.lower(),
                        zero_intercept=True,
                        verbose=False
                    )
                    velocities[i] = reg_result['slope']
                else:
                    raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
            
                # add central time of window
                times[i] = (i1 + win_samples/2) / df
                
                # add cross-correlation coefficient 
                cc_coeffs[i] = abs(cc)
            else:
                times[i] = (i1 + win_samples/2) / df
                velocities[i] = np.nan
                cc_coeffs[i] = abs(cc)
        
        # Create output dictionary
        results = {
            'time': times,
            'velocity': velocities,
            'ccoef': cc_coeffs,
            'terr': np.ones_like(times) * win_time_s/2,
            'parameters': {
                'wave_type': wave_type,
                'win_time_s': win_time_s,
                'overlap': overlap,
                'cc_threshold': cc_threshold,
                'baz': baz,
                'fmin': self.fmin,
                'fmax': self.fmax
            }
        }
        
        return results

    def compute_velocities_envelope(self, wave_type: str="love", win_time_s: float=None, overlap: float=0.5, 
                                   cc_threshold: float=0.2, baz: float=None, method: str='odr') -> Dict:
        """
        Compute phase velocities in time intervals using waveform envelopes instead of raw waveforms.
        This is useful for analyzing amplitude-modulated signals or when phase information is less reliable.
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love' or 'rayleigh')
        win_time_s : float or None
            Window length in seconds. If None, uses 1/fmin
        overlap : float
            Window overlap in percent (0-1)
        cc_threshold : float
            Minimum cross-correlation coefficient threshold
        baz : float or None
            Backazimuth in degrees. If None, uses theoretical or estimated BAZ
        method : str
            Regression method ('odr' or 'ransac')
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - time: array of time points
            - velocity: array of phase velocities
            - ccoef: array of cross-correlation coefficients
            - terr: array of time window lengths
            - parameters: dictionary of parameters including envelope flag
        """
        import numpy as np
        from obspy.signal.rotate import rotate_ne_rt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from scipy.signal import hilbert
        
        # Get and process streams
        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()
        
        # Get sampling rate
        df = self.sampling_rate
        
        # Set window length if not provided
        if win_time_s is None:
            win_time_s = 1/self.fmin
            
        # Get backazimuth if not provided
        if baz is None:
            if hasattr(self, 'theoretical_baz'):
                if self.verbose:
                    print(f"Using theoretical BAZ {self.theoretical_baz}")
                baz = self.theoretical_baz
            elif hasattr(self, 'baz_estimated'):
                if self.verbose:
                    print(f"Using estimated BAZ {self.baz_estimated[wave_type.lower()]}")
                baz = self.baz_estimated[wave_type.lower()]
            else:
                raise ValueError("No backazimuth provided or available")
        
        # Get Z component and rotate components to radial-transverse
        if wave_type == 'rayleigh':
            acc_z = acc.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(rot.select(channel='*N')[0].data,
                                    rot.select(channel='*E')[0].data,
                                    baz)
            n_samples = len(acc_z)
            
            # Compute envelopes using Hilbert transform
            acc_z_env = np.abs(hilbert(acc_z))
            rot_t_env = np.abs(hilbert(rot_t))

        elif wave_type == 'love':
            rot_z = rot.select(channel="*Z")[0].data
            acc_r, acc_t = rotate_ne_rt(acc.select(channel='*N')[0].data,
                                    acc.select(channel='*E')[0].data,
                                    baz)
            n_samples = len(rot_z)
            
            # Compute envelopes using Hilbert transform
            rot_z_env = np.abs(hilbert(rot_z))
            acc_t_env = np.abs(hilbert(acc_t))

        # Calculate window parameters
        win_samples = int(win_time_s * df)
        overlap_samples = int(win_samples * overlap)
        step = win_samples - overlap_samples

        n_windows = int((n_samples - win_samples) / step) + 1
        
        # Initialize arrays
        times = np.zeros(n_windows)
        velocities = np.zeros(n_windows)
        cc_coeffs = np.zeros(n_windows)
        
        # Loop through windows
        for i in range(n_windows):
            i1 = i * step
            i2 = i1 + win_samples
            
            # compute cross-correlation coefficient using envelopes
            if wave_type.lower() == 'love':
                # For Love waves: use transverse acceleration envelope and vertical rotation envelope
                cc = xcorr_max(correlate(rot_z_env[i1:i2], acc_t_env[i1:i2], 0))[1]

            elif wave_type.lower() == 'rayleigh':
                # For Rayleigh waves: use vertical acceleration envelope and transverse rotation envelope
                cc = xcorr_max(correlate(rot_t_env[i1:i2], acc_z_env[i1:i2], 0))[1]
            else:
                raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")

            # Compute velocity using amplitude ratio of envelopes
            if abs(cc) > cc_threshold:
                if wave_type.lower() == 'love':
                    # get velocity from envelope amplitude ratio via regression
                    reg_result = self.regression(
                        rot_z_env[i1:i2],
                        0.5*acc_t_env[i1:i2],
                        method=method.lower(),
                        zero_intercept=True,
                        verbose=False
                    )
                    velocities[i] = reg_result['slope']
                elif wave_type.lower() == 'rayleigh':
                    # get velocity from envelope amplitude ratio via regression
                    reg_result = self.regression(
                        rot_t_env[i1:i2],
                        acc_z_env[i1:i2],
                        method=method.lower(),
                        zero_intercept=True,
                        verbose=False
                    )
                    velocities[i] = reg_result['slope']
                else:
                    raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
            
                # add central time of window
                times[i] = (i1 + win_samples/2) / df
                
                # add cross-correlation coefficient 
                cc_coeffs[i] = abs(cc)
            else:
                times[i] = (i1 + win_samples/2) / df
                velocities[i] = np.nan
                cc_coeffs[i] = abs(cc)
        
        # Create output dictionary
        results = {
            'time': times,
            'velocity': velocities,
            'ccoef': cc_coeffs,
            'terr': np.ones_like(times) * win_time_s/2,
            'parameters': {
                'wave_type': wave_type,
                'win_time_s': win_time_s,
                'overlap': overlap,
                'cc_threshold': cc_threshold,
                'baz': baz,
                'fmin': self.fmin,
                'fmax': self.fmax,
                'use_envelope': True,
                'method': method
            }
        }
        
        return results

    def compare_backazimuth_methods(self, Twin: float, Toverlap: float, baz_theo: float=None, 
                                  baz_theo_margin: float=10, baz_step: int=1, minors: bool=True,
                                  cc_threshold: float=0, cc_method: str='max', plot: bool=False, output: bool=False,
                                  precomputed: bool=True, wave_types: list=None, colorcode_tangent: bool=True) -> Tuple[plt.Figure, Dict]:
        """
        Compare different backazimuth estimation methods
        
        Parameters:
        -----------
        Twin : float
            Window length in seconds for backazimuth estimation
        Toverlap : float
            Window overlap in percent (0-100)
        baz_theo : float, optional
            Theoretical backazimuth in degrees
        baz_theo_margin : float, optional
            Margin around theoretical backazimuth in degrees
        baz_step : int, optional
            Step size for backazimuth search in degrees
        minors : bool, optional
            Add minor ticks to axes if True
        cc_threshold : float, optional
            Minimum cross-correlation coefficient threshold (default: 0)
        plot : bool, optional
            Whether to create and return plot
        output : bool, optional
            Whether to return results dictionary
        invert_rot_z : bool, optional
            Invert vertical rotation component if True
        invert_acc_z : bool, optional
            Invert vertical acceleration component if True
        cc_method : str, optional
            Method to use for cross-correlation coefficient thresholding ('max' or 'mid')
        wave_types : list, optional
            List of wave types to show. Options: 'love', 'rayleigh', 'tangent'. 
            Default is None which shows all wave types.
        colorcode_tangent : bool, optional
            Colorcode tangent method if True, otherwise use blue color
        Returns:
        --------
        Tuple[Figure, Dict] or Dict
            Figure and results dictionary if plot=True, else just results dictionary
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import scipy.stats as sts
        from numpy import ones, linspace, histogram, concatenate, average
        from numpy import argmax, sqrt, cov, array, arange, nan
        
        # Set default wave types if not provided
        if wave_types is None:
            wave_types = ['love', 'rayleigh', 'tangent']
        else:
            # Validate wave_types
            valid_wave_types = ['love', 'rayleigh', 'tangent']
            wave_types = [wt.lower() for wt in wave_types]
            invalid_types = [wt for wt in wave_types if wt not in valid_wave_types]
            if invalid_types:
                raise ValueError(f"Invalid wave_types: {invalid_types}. Valid options are: {valid_wave_types}")
        
        # Get and process streams
        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()
        
        # Initialize results dictionary
        results_dict = {}
        baz_estimated = {}
        
        # Create figure if plotting
        if plot:
            n_plots = len(wave_types)
            fig = plt.figure(figsize=(15, 3.5 * n_plots))
            gs = GridSpec(n_plots, 8, figure=fig)
            
            # Create subplots dynamically based on wave_types
            axes_dict = {}
            for idx, wave_type in enumerate(['love', 'rayleigh', 'tangent']):
                if wave_type in wave_types:
                    plot_idx = wave_types.index(wave_type)
                    axes_dict[wave_type] = {
                        'main': fig.add_subplot(gs[plot_idx, :8]),
                        'hist': fig.add_subplot(gs[plot_idx, 7:])
                    }
                    axes_dict[wave_type]['hist'].set_axis_off()
            
            # Create color map
            cmap = plt.get_cmap("viridis", 10)
        
        # Plot settings
        font = 12
        deltaa = 10
        angles1 = arange(0, 365, deltaa)
        angles2 = arange(0, 365, 1)
        t1, t2 = 0, rot[0].stats.endtime - rot[0].stats.starttime
        
        # Initialize scatter for colorbar (only needed for love/rayleigh)
        scatter = None
        
        # Process each wave type
        for wave_type, label in [('love', 'Love'), ('rayleigh', 'Rayleigh'), ('tangent', 'Tangent')]:
            # Skip if wave_type not in requested list
            if wave_type not in wave_types:
                continue
            
            # Compute backazimuth for each wave type

            if precomputed and wave_type in self.baz_results.keys():
                print(f"Using precomputed {wave_type} backazimuth results")
                wave_results = self.baz_results[wave_type]
            else:
                print(f"Computing {wave_type} wave backazimuth...")
                wave_results = self.compute_backazimuth(
                    wave_type=wave_type,
                    baz_step=baz_step,
                    baz_win_sec=Twin,
                    baz_win_overlap=Toverlap,
                    out=True
                )
            
            # Filter out low correlation coefficients
            if cc_method.lower() == 'max':
                mask = wave_results['cc_max'] > cc_threshold
                times_filtered = wave_results['twin_center'][mask]
                baz_filtered = wave_results['baz_max'][mask]
                cc_filtered = wave_results['cc_max'][mask]
            elif cc_method.lower() == 'mid':
                mask = wave_results['cc_mid'] > cc_threshold
                times_filtered = wave_results['twin_center'][mask]
                baz_filtered = wave_results['baz_mid'][mask]
                cc_filtered = wave_results['cc_mid'][mask]
            else:
                raise ValueError(f"Invalid cc_method: {cc_method}. Use 'max' or 'mid'")
            
            # Tangent method is special as it has both positive and negative cc_max at -1 and 1
            if wave_type == 'tangent':
                mask = abs(wave_results['cc_max']) > cc_threshold
                times_filtered = wave_results['twin_center'][mask]
                baz_filtered = wave_results['baz_max'][mask]
                cc_filtered = wave_results['cc_max'][mask]

            # Store filtered results
            results_dict[wave_type] = {
                'time': times_filtered,
                'backazimuth': baz_filtered,
                'correlation': cc_filtered,
                'twin_center': wave_results['twin_center'],
                'cc_max': wave_results['cc_max'],
                'cc_mid': wave_results['cc_mid'],
                'baz_max': wave_results['baz_max'],
                'baz_mid': wave_results['baz_mid'],
                'parameters': {
                    'wave_type': wave_type,
                    'baz_step': baz_step,
                    'baz_win_sec': Twin,
                    'baz_win_overlap': Toverlap,
                    'cc_threshold': cc_threshold,
                }
            }
            
            if plot:
                # Get the appropriate axes for this wave type
                ax = axes_dict[wave_type]['main']
                
                # Plot results for each wave type
                if wave_type == 'love':
                    scatter = ax.scatter(
                        times_filtered,
                        baz_filtered,
                        c=cc_filtered,
                        cmap=cmap,
                        s=70,
                        alpha=0.7,
                        vmin=0,
                        vmax=1,
                        edgecolors="k",
                        lw=1,
                        zorder=3
                    )
                elif wave_type == 'rayleigh':
                    scatter = ax.scatter(
                        times_filtered,
                        baz_filtered,
                        c=cc_filtered,
                        cmap=cmap,
                        s=70,
                        alpha=0.7,
                        vmin=0,
                        vmax=1,
                        edgecolors="k",
                        lw=1,
                        zorder=3
                    )
                else:  # tangent
                    if colorcode_tangent:
                        ax.scatter(
                            wave_results['twin_center'][wave_results['cc_max'] > cc_threshold], 
                            wave_results['baz_max'][wave_results['cc_max'] > cc_threshold],
                            c=cc_filtered,
                            cmap=cmap,
                            s=70,
                            alpha=0.7,
                            vmin=0,
                            vmax=1,
                            edgecolors="k",
                            lw=1,
                            zorder=3
                         )
                    else:
                        ax.scatter(
                            wave_results['twin_center'][wave_results['cc_max'] > cc_threshold], 
                            wave_results['baz_max'][wave_results['cc_max'] > cc_threshold],
                            c='tab:blue',
                            s=70,
                            alpha=0.7,
                            edgecolors="k",
                            lw=1,
                            zorder=3
                        )

                try:
                    # Compute and plot histogram
                    hist = histogram(baz_filtered,
                                bins=len(angles1)-1,
                                range=[min(angles1), max(angles1)],
                                weights=cc_filtered,
                                density=True)
                except:
                    pass
                
            # Compute KDE
            if len(baz_filtered) > 5:  # Need at least 5 points for KDE
                # get kde stats
                kde_stats = self.get_kde_stats(baz_filtered, cc_filtered, _baz_steps=0.5, Ndegree=60, plot=False)

                baz_estimated[wave_type] = kde_stats['baz_estimate']
                results_dict[wave_type]['kde'] = kde_stats['kde_values']
                results_dict[wave_type]['kde_angles'] = kde_stats['kde_angles']
                results_dict[wave_type]['baz_std'] = kde_stats['kde_dev']
                results_dict[wave_type]['baz_estimate'] = kde_stats['baz_estimate']
                results_dict[wave_type]['n_samples'] = kde_stats.get('n_samples', 0)
            else:
                baz_estimated[wave_type] = nan
                results_dict[wave_type]['n_samples'] = 0
            
            print(f"\nEstimated BAz {label} = {baz_estimated[wave_type]}° (CC ≥ {cc_threshold})")
    
        if plot:

            # add histograms and KDEs to subplots
            for wave_type in wave_types:
                label = wave_type
                ax = axes_dict[wave_type]['main']
                ax_hist = axes_dict[wave_type]['hist']
                
                # Get filtered data for this wave type
                baz_filtered_current = results_dict[label]['backazimuth']
                
                if len(baz_filtered_current) > 0:
                    try:
                        ax_hist.hist(results_dict[label]['backazimuth'], 
                                    bins=len(angles1)-1,
                                    range=[min(angles1), max(angles1)],
                                    weights=results_dict[label]['correlation'],
                                    orientation="horizontal", density=True, color="grey")
                    except Exception:
                        # Skip histogram if there's an issue (e.g., all weights are zero)
                        pass
                    
                    # Check if KDE exists before plotting
                    if (len(baz_filtered_current) > 5 and 
                        'kde' in results_dict[label] and 
                        'kde_angles' in results_dict[label] and
                        results_dict[label].get('kde') is not None and
                        results_dict[label].get('kde_angles') is not None):
                        try:
                            ax_hist.plot(results_dict[label]['kde'],
                                        results_dict[label]['kde_angles'],
                                        color='k', lw=3)
                        except (KeyError, TypeError, ValueError):
                            # Skip KDE plot if data is missing or invalid
                            pass
                    
                    ax_hist.yaxis.tick_right()
                    ax_hist.invert_xaxis()
                    ax_hist.set_ylim(-5, 365)

            # Add theoretical BAZ if provided
            if baz_theo is not None:
                for wave_type in wave_types:
                    ax = axes_dict[wave_type]['main']
                    ax.plot([t1, t2], [baz_theo, baz_theo], color='k', ls='--', label='Theoretical BAZ')
                    ax.fill_between([t1, t2], 
                                  baz_theo-baz_theo_margin,
                                  baz_theo+baz_theo_margin,
                                  color='grey', alpha=0.3, zorder=1)
            
            # Configure axes
            for wave_type in wave_types:
                ax = axes_dict[wave_type]['main']
                ax.set_ylim(-5, 365)
                ax.set_yticks(range(0, 360+60, 60))
                ax.grid(True, alpha=0.3)
                ax.set_xlim(t1, t2*1.15)
                if minors:
                    ax.minorticks_on()
            
            # Add labels
            title_map = {'love': 'Love', 'rayleigh': 'Rayleigh', 'tangent': 'Tangent'}
            for wave_type in wave_types:
                ax = axes_dict[wave_type]['main']
                baz_val = baz_estimated.get(wave_type, nan)
                title = f"{title_map[wave_type]} Method BAz (estimated = {baz_val}°)"
                ax.set_title(title, fontsize=font)
                ax.set_ylabel("BAz (°)", fontsize=font)
            
            # Set xlabel on last axis
            if wave_types:
                axes_dict[wave_types[-1]]['main'].set_xlabel("Time (s)", fontsize=font)
            
            # Add colorbar (only if there are scatter plots with colorbar)
            if scatter is not None:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                plt.colorbar(scatter, cax=cbar_ax, label='CC coefficient')
            
            # Add title
            title = f"{rot[0].stats.starttime.date} {str(rot[0].stats.starttime.time).split('.')[0]} UTC"
            title += f" | {self.fmin}-{self.fmax} Hz | T = {Twin} s | {Toverlap*100:.0f}% overlap"
            if baz_theo is not None:
                title += f" | expected BAz = {baz_theo:.1f}°"
            if cc_threshold > 0:
                title += f" | CC ≥ {cc_threshold}"
            fig.suptitle(title, fontsize=font+2, y=0.99)
            
            # plt.tight_layout()
            # plt.show()

        # Prepare output
        if output:
            if plot:
                return fig, results_dict
            else:
                return results_dict

    def compute_envelope_statistics(self, rotation_data: Stream=None, translation_data: Stream=None,
                                 love_baz_results: Dict=None, rayleigh_baz_results: Dict=None, 
                                 baz_mode: str='mid', cc_threshold: float=0.0) -> Dict:
        """
        Compute mean and standard deviation of signal envelopes in time intervals for Love and Rayleigh waves
        
        Parameters:
        -----------
        rotation_data : Stream
            Rotational data stream
        translation_data : Stream
            Translation data stream
        love_baz_results : Dict
            Dictionary containing backazimuth results for Love waves
        rayleigh_baz_results : Dict
            Dictionary containing backazimuth results for Rayleigh waves
        baz_mode : str
            Mode to use for backazimuth selection ('max' or 'mid')
        cc_threshold : float, optional
            Minimum cross-correlation coefficient to consider, by default 0.0
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - times : array of time points
            - envelope_means: dict with component means
            - envelope_stds: dict with component standard deviations
            - cc_value: dict with cc values for love and rayleigh
            - backazimuth: dict with baz values for love and rayleigh
        """
        import numpy as np
        from obspy.signal.rotate import rotate_ne_rt
        from scipy.signal import hilbert

        # Validate inputs
        if rotation_data is None or translation_data is None:
            raise ValueError("Both rotation and translation data must be provided")
        if love_baz_results is None or rayleigh_baz_results is None:
            raise ValueError("Both Love and Rayleigh backazimuth results must be provided")
        if baz_mode.lower() not in ['max', 'mid']:
            raise ValueError(f"Invalid baz mode: {baz_mode}. Use 'max' or 'mid'")

        # Make copies to avoid modifying original data
        rot = rotation_data.copy()
        tra = translation_data.copy()

        # Get sampling rate and validate
        df = rot[0].stats.sampling_rate
        if df <= 0:
            raise ValueError(f"Invalid sampling rate: {df}")

        # Extract parameters from baz_results for both wave types
        try:
            # Love wave parameters
            love_win_time_s = love_baz_results['parameters']['baz_win_sec']
            love_overlap = love_baz_results['parameters']['baz_win_overlap']
            love_ttt = love_baz_results['twin_center']
            if baz_mode.lower() == 'max':
                love_baz = love_baz_results['baz_max']
                love_ccc = love_baz_results['cc_max']
            else:  # 'mid'
                love_baz = love_baz_results['baz_mid']
                love_ccc = love_baz_results['cc_mid']

            # Rayleigh wave parameters
            rayleigh_win_time_s = rayleigh_baz_results['parameters']['baz_win_sec']
            rayleigh_overlap = rayleigh_baz_results['parameters']['baz_win_overlap']
            rayleigh_ttt = rayleigh_baz_results['twin_center']
            if baz_mode.lower() == 'max':
                rayleigh_baz = rayleigh_baz_results['baz_max']
                rayleigh_ccc = rayleigh_baz_results['cc_max']
            else:  # 'mid'
                rayleigh_baz = rayleigh_baz_results['baz_mid']
                rayleigh_ccc = rayleigh_baz_results['cc_mid']

            # Verify window parameters match
            if (love_win_time_s != rayleigh_win_time_s or 
                love_overlap != rayleigh_overlap or 
                not np.array_equal(love_ttt, rayleigh_ttt)):
                raise ValueError("Window parameters must match between Love and Rayleigh results")

        except KeyError as e:
            raise ValueError(f"Missing required key in backazimuth results: {e}")

        # Use Love wave parameters for window calculations since they should match
        win_time_s = love_win_time_s
        overlap = love_overlap
        ttt = love_ttt

        # Calculate window parameters
        win_samples = int(win_time_s * df)
        if win_samples <= 0:
            raise ValueError(f"Invalid window size: {win_samples} samples")

        overlap_samples = int(win_samples * overlap)
        step = win_samples - overlap_samples

        # number of windows
        n_windows = len(ttt)

        # Initialize output arrays
        envelope_means = {
            'rot_z': np.zeros(n_windows),  # Love
            'rot_r': np.zeros(n_windows),  # Rayleigh
            'rot_t': np.zeros(n_windows),  # Rayleigh
            'acc_z': np.zeros(n_windows),  # Rayleigh
            'acc_r': np.zeros(n_windows),  # Love
            'acc_t': np.zeros(n_windows)   # Love
        }
        
        envelope_stds = {
            'rot_z': np.zeros(n_windows),  # Love
            'rot_r': np.zeros(n_windows),  # Rayleigh
            'rot_t': np.zeros(n_windows),  # Rayleigh
            'acc_z': np.zeros(n_windows),  # Rayleigh
            'acc_r': np.zeros(n_windows),  # Love
            'acc_t': np.zeros(n_windows)   # Love
        }

        def compute_envelope(data):
            """Helper function to compute signal envelope using Hilbert transform"""
            return np.abs(hilbert(data))

        # Loop through windows
        for i in range(n_windows):
            i1 = i * step
            i2 = i1 + win_samples

            # Get vertical components (used for both Love and Rayleigh)
            rot_z = rot.select(channel="*Z")[0].data[i1:i2]
            acc_z = tra.select(channel="*Z")[0].data[i1:i2]

            # Process Love wave components if above threshold
            if love_ccc[i] > cc_threshold:
                # Rotate horizontals using Love backazimuth
                acc_r_love, acc_t_love = rotate_ne_rt(
                    tra.select(channel='*N')[0].data[i1:i2],
                    tra.select(channel='*E')[0].data[i1:i2],
                    love_baz[i]
                )
                
                # Compute envelopes for Love components
                envelope_means['rot_z'][i] = np.nanmean(compute_envelope(rot_z))
                envelope_stds['rot_z'][i] = np.nanstd(compute_envelope(rot_z))
                envelope_means['acc_r'][i] = np.nanmean(compute_envelope(acc_r_love))
                envelope_stds['acc_r'][i] = np.nanstd(compute_envelope(acc_r_love))
                envelope_means['acc_t'][i] = np.nanmean(compute_envelope(acc_t_love))
                envelope_stds['acc_t'][i] = np.nanstd(compute_envelope(acc_t_love))
            else:
                envelope_means['rot_z'][i] = np.nan
                envelope_stds['rot_z'][i] = np.nan
                envelope_means['acc_r'][i] = np.nan
                envelope_stds['acc_r'][i] = np.nan
                envelope_means['acc_t'][i] = np.nan
                envelope_stds['acc_t'][i] = np.nan

            # Process Rayleigh wave components if above threshold
            if rayleigh_ccc[i] > cc_threshold:
                # Rotate horizontals using Rayleigh backazimuth
                rot_r_rayleigh, rot_t_rayleigh = rotate_ne_rt(
                    rot.select(channel='*N')[0].data[i1:i2],
                    rot.select(channel='*E')[0].data[i1:i2],
                    rayleigh_baz[i]
                )
                
                # Compute envelopes for Rayleigh components
                envelope_means['rot_r'][i] = np.nanmean(compute_envelope(rot_r_rayleigh))
                envelope_stds['rot_r'][i] = np.nanstd(compute_envelope(rot_r_rayleigh))
                envelope_means['rot_t'][i] = np.nanmean(compute_envelope(rot_t_rayleigh))
                envelope_stds['rot_t'][i] = np.nanstd(compute_envelope(rot_t_rayleigh))
                envelope_means['acc_z'][i] = np.nanmean(compute_envelope(acc_z))
                envelope_stds['acc_z'][i] = np.nanstd(compute_envelope(acc_z))
            else:
                envelope_means['rot_r'][i] = np.nan
                envelope_stds['rot_r'][i] = np.nan
                envelope_means['rot_t'][i] = np.nan
                envelope_stds['rot_t'][i] = np.nan
                envelope_means['acc_z'][i] = np.nan
                envelope_stds['acc_z'][i] = np.nan


        # get oveerall mean and std
        envelope_stats = {}

        for comp in ['rot_z', 'rot_r', 'rot_t', 'acc_z', 'acc_r', 'acc_t']:
            envelope_stats[comp] = {}
            envelope_stats[comp]['mean'] = np.nanmean(envelope_means[comp])
            envelope_stats[comp]['std'] = np.nanstd(envelope_means[comp])
            envelope_stats[comp]['median'] = np.nanmedian(envelope_means[comp])
            envelope_stats[comp]['mad'] = np.nanmedian(np.abs(envelope_means[comp] - np.nanmedian(envelope_means[comp])))

        return {
            'times': ttt,
            'envelope_means': envelope_means,
            'envelope_stds': envelope_stds,
            'envelope_stats': envelope_stats,
            'cc_value': {
                'love': love_ccc,
                'rayleigh': rayleigh_ccc
            },
            'backazimuth': {
                'love': love_baz,
                'rayleigh': rayleigh_baz
            },
            'parameters': {
                'win_time_s': win_time_s,
                'overlap': overlap,
                'baz_mode': baz_mode
            }
        }

    def compute_spectra(self, method='welch', nperseg=None, noverlap=None, nfft=None, 
                        window='hann', detrend='constant', scaling='density', raw=False,
                        time_bandwidth=4.0, nw=None, kspec=8, store=True, output=False):
        """
        Compute spectra for each channel using specified method.
        
        Parameters
        ----------
        method : str, optional
            Spectral estimation method: 'welch', 'multitaper', or 'fft'. Default is 'welch'.
        nperseg : int, optional
            Length of each segment for Welch's method. If None, defaults to 256.
        noverlap : int, optional
            Number of points to overlap between segments for Welch's method.
            If None, defaults to nperseg//2.
        nfft : int, optional
            Length of FFT. If None, defaults to nperseg.
        raw : bool, optional
            Whether to return the raw spectra. Default is False.
        window : str or tuple, optional
            Window function for Welch's method. Default is 'hann'.
        detrend : str or function, optional
            Detrending function. Default is 'constant'.
        scaling : str, optional
            Scaling mode for PSD computation: 'density' or 'spectrum'. Default is 'density'.
        time_bandwidth : float, optional
            Time-bandwidth product for multitaper method. Default is 4.0.
        nw : int, optional
            Number of tapers for multitaper method. If None, calculated from time_bandwidth.
        kspec : int, optional
            Number of tapers to use in multitaper method. Default is 8.
        store : bool, optional
            Whether to store the computed spectra. Default is True.
        output : bool, optional
            Whether to return the computed spectra. Default is False.
            
        Returns
        -------
        dict or None
            If output=True, returns dictionary containing frequencies and spectra for each channel.
            Keys are channel names, values are tuples of (frequencies, spectra).
        """
        import numpy as np
        from scipy import signal
        from scipy.fft import fft, fftfreq, fftshift

        if raw:
            stream = self.st0.copy()
        else:
            stream = self.st.copy()
        
        # Input validation
        if method not in ['welch', 'multitaper', 'fft']:
            raise ValueError("Method must be one of: 'welch', 'multitaper', 'fft'")
        
        # Initialize results dictionary
        spectra_dict = {}
        
        # Process all traces
        for tr in stream:
            # Get sampling parameters
            dt = tr.stats.delta
            fs = 1.0 / dt
            
            if nperseg is None:
                nperseg = min(256, len(tr.data))
            if noverlap is None:
                noverlap = nperseg // 2
            if nfft is None:
                nfft = nperseg
            
            # Compute spectra based on method
            if method == 'welch':
                freqs, psd = signal.welch(
                    tr.data, 
                    fs=fs, 
                    window=window,
                    nperseg=nperseg, 
                    noverlap=noverlap,
                    nfft=nfft, 
                    detrend=detrend,
                    scaling=scaling
                )
                spectra = psd
                
            elif method == 'multitaper':
                try:
                    import multitaper as mt
                except ImportError:
                    raise ImportError("multitaper package required for multitaper analysis")
                    
                if nw is None:
                    nw = time_bandwidth
                    
                out_psd = mt.MTSpec(tr.data, nw=time_bandwidth, kspec=kspec, dt=dt, iadapt=2).rspec()
                freqs, spectra = out_psd[0][1:], out_psd[1][1:]
 
            elif method == 'fft':

                # determine length of the input time series
                n = int(len(tr.data))

                # calculate spectrum (with or without window function applied to time series)
                if window is not None:
                    win = signal.get_window(window, n);
                    spectrum = fftshift(fft(tr.data * win))
                else:
                    spectrum = fftshift(fft(tr.data))

                # calculate frequency array
                freqs = fftshift(fftfreq(n, d=dt))

                # calculate amplitude spectrum
                spectra = abs(spectrum) * 2.0 / n

            # Store results
            spectra_dict[tr.stats.channel] = (freqs, spectra)
    
        # Store if requested
        if store:
            self.spectra = {
                'method': method,
                'spectra': spectra_dict,
                'params': {
                    'nperseg': nperseg,
                    'noverlap': noverlap,
                    'nfft': nfft,
                    'window': window,
                    'detrend': detrend,
                    'scaling': scaling
                }
            }
            if method == 'multitaper':
                self.spectra['params'].update({
                    'time_bandwidth': time_bandwidth,
                    'nw': nw,
                    'kspec': kspec
                })
        
        # Return if requested
        if output:
            return spectra_dict
    
    @staticmethod
    def get_time_windows(tbeg: Union[None, str, UTCDateTime]=None, tend: Union[None, str, UTCDateTime]=None, interval_seconds: int=3600, fractional_overlap: float=0) -> List[Tuple[UTCDateTime, UTCDateTime]]:
        '''
        Obtain time intervals
        '''

        from obspy import UTCDateTime

        tbeg = UTCDateTime(tbeg)
        tend = UTCDateTime(tend)

        times = []
        t1, t2 = tbeg, tbeg + interval_seconds

        while t2 <= tend:
            times.append((t1, t2))
            t1 = t1 + interval_seconds - interval_seconds * fractional_overlap
            t2 = t2 + interval_seconds - interval_seconds * fractional_overlap

        return times

    @staticmethod
    def sync_twin_axes(ax1, ax2):
        """Synchronize twin axes with same grid"""
        
        # Get limits
        ymin1, ymax1 = ax1.get_ylim()
        ymin2, ymax2 = ax2.get_ylim()
        
        # Calculate normalized limits
        ymax = max(abs(ymax1), abs(ymax2))
        ymin = -ymax
        
        # Set same number of ticks
        ax1.yaxis.set_major_locator(plt.MaxNLocator(8))
        ax2.yaxis.set_major_locator(plt.MaxNLocator(8))
        
        # Set symmetric limits
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)
        
        # Add grid from primary axis only
        ax1.grid(True, which='both', ls=':', alpha=0.7)

    @staticmethod
    def read_from_sds(path_to_archive: str, seed: str, tbeg: Union[str, UTCDateTime], tend: Union[str, UTCDateTime], data_format: str="MSEED") -> Stream:
        """
        VARIABLES:
         - path_to_archive
         - seed
         - tbeg, tend
         - data_format

        DEPENDENCIES:
         - from obspy.core import UTCDateTime
         - from obspy.clients.filesystem.sds import Client

        OUTPUT:
         - stream

        EXAMPLE:
        >>> st = read_sds(path_to_archive, seed, tbeg, tend, data_format="MSEED")
        """

        import os
        from obspy.core import UTCDateTime, Stream
        from obspy.clients.filesystem.sds import Client

        tbeg, tend = UTCDateTime(tbeg), UTCDateTime(tend)

        if not os.path.exists(path_to_archive):
            print(f" -> {path_to_archive} does not exist!")
            return

        ## separate seed id
        net, sta, loc, cha = seed.split('.')

        ## define SDS client
        client = Client(path_to_archive, sds_type='D', format=data_format)

        ## read waveforms
        try:
            st = client.get_waveforms(net, sta, loc, cha, tbeg, tend, merge=-1)
        except:
            print(f" -> failed to obtain waveforms!")
            st = Stream()

        return st

    @staticmethod
    def interpolate_nan(array_like: NDArray[np.float64]) -> NDArray[np.float64]:
        '''
        interpolate NaN values in array linearly
        '''

        from numpy import isnan, interp

        array = array_like.copy()

        nans = isnan(array)

        def get_x(a):
            return a.nonzero()[0]

        array[nans] = interp(get_x(nans), get_x(~nans), array[~nans])

        return array

    @staticmethod
    def get_fft(arr: array, dt: float, window: Union[None, str]=None) -> Tuple[array, array, array]:

        '''
        Calculating a simple 1D FastFourierSpectrum of a time series.

        >>> frequencies, spectrum, phase = get_fft(signal_in, dt ,window=None,normalize=None)
        '''

        from scipy.fft import fft, fftfreq, fftshift
        from scipy import signal
        from numpy import angle, imag

        # determine length of the input time series
        n = int(len(arr))

        # calculate spectrum (with or without window function applied to time series)
        if window is not None:
            win = signal.get_window(window, n);
            spectrum = fft(arr * win)
        else:
            spectrum = fft(arr)

        # calculate frequency array
        frequencies = fftfreq(n, d=dt)

        # calculate amplitude spectrum
        magnitude = abs(spectrum) * 2.0 / n

        # calculate phase spectrum
        phase = angle(spectrum, deg=False)

        return frequencies[0:n//2], magnitude[0:n//2], phase[0:n//2]

    @staticmethod
    def get_kde_stats(_baz, _ccc, _baz_steps=5, Ndegree=180, plot=False):
        """
        Get the statistics of the kde of the backazimuth and the cc values
        
        Parameters:
        -----------
        _baz : array-like
            Backazimuth values
        _ccc : array-like
            Cross-correlation coefficient values (weights)
        _baz_steps : int, optional
            Step size for backazimuth (default: 5)
        Ndegree : int, optional
            Number of degrees (default: 180)
        plot : bool, optional
            Whether to plot the KDE (default: False)
        
        Returns:
        --------
        dict
            Dictionary containing KDE statistics with keys:
            - baz_estimate: Estimated backazimuth (NaN if insufficient samples)
            - kde_max: Maximum KDE value position (NaN if insufficient samples)
            - shift: Shift applied (NaN if insufficient samples)
            - kde_values: KDE values array (empty if insufficient samples)
            - kde_angles: KDE angles array (empty if insufficient samples)
            - kde_dev: Standard deviation (NaN if insufficient samples)
            - kde_mad: Median absolute deviation (NaN if insufficient samples)
            - n_samples: Number of samples used for KDE estimation (0 if insufficient)
        """
        import numpy as np
        import scipy.stats as sts
        import matplotlib.pyplot as plt

        # Convert to numpy arrays and filter out NaN/inf values
        _baz = np.asarray(_baz)
        _ccc = np.asarray(_ccc)
        
        # Filter out invalid values
        valid_mask = np.isfinite(_baz) & np.isfinite(_ccc)
        _baz_valid = _baz[valid_mask]
        _ccc_valid = _ccc[valid_mask]
        
        # Count valid samples
        n_samples = len(_baz_valid)
        
        # Check if we have enough samples for KDE (need at least 2)
        if n_samples < 5:
            # Return NaN values and zero sample count
            kde_angles = np.arange(0, 361, 1)
            return {
                'baz_estimate': np.nan,
                'kde_max': np.nan,
                'shift': np.nan,
                'kde_values': np.array([]),
                'kde_angles': kde_angles,
                'kde_dev': np.nan,
                'kde_mad': np.nan,
                'n_samples': 0,
            }

        # define angles for kde and histogram
        kde_angles = np.arange(0, 361, 1)
        hist_angles = np.arange(0, 365, 5)

        try:
            # get first kde estimate to determine the shift
            kde = sts.gaussian_kde(_baz_valid, weights=_ccc_valid, bw_method='scott')
            kde_max = np.argmax(kde.pdf(kde_angles))

            # determine the shift with respect to 180°
            shift = 180 - kde_max

            # shift the backazimuth array to the center of the x-axis
            _baz_shifted = (_baz_valid + shift) % 360

            # get second kde estimate
            kde_shifted = sts.gaussian_kde(_baz_shifted, weights=_ccc_valid, bw_method='scott')
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

            # get standard deviation (weighted)
            try:
                # Compute weighted mean
                weighted_mean = np.average(_baz_shifted, weights=_ccc_valid)
                # Compute weighted variance
                weighted_variance = np.average((_baz_shifted - weighted_mean)**2, weights=_ccc_valid)
                dev = int(np.round(np.sqrt(weighted_variance), 0))
            except Exception:
                # Fallback to unweighted standard deviation
                dev = int(np.round(np.std(_baz_shifted), 0))

            # get median absolute deviation
            try:
                mad = int(np.round(np.median(np.abs(_baz_shifted - baz_estimate)), 0))
            except Exception:
                mad = np.nan

            if plot:
                plt.figure(figsize=(10, 5))

                plt.hist(_baz_valid, bins=hist_angles, weights=_ccc_valid, density=True, alpha=0.5)
                plt.hist(_baz_shifted, bins=hist_angles, weights=_ccc_valid, density=True, alpha=0.5)

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
                'kde_mad': mad,
                'n_samples': n_samples,
            }

            return out
            
        except (ValueError, np.linalg.LinAlgError) as e:
            # KDE failed (e.g., not enough samples, singular matrix)
            # Return NaN values and actual sample count
            return {
                'baz_estimate': np.nan,
                'kde_max': np.nan,
                'shift': np.nan,
                'kde_values': np.array([]),
                'kde_angles': kde_angles,
                'kde_dev': np.nan,
                'kde_mad': np.nan,
                'n_samples': n_samples,
            }

    @staticmethod
    def compute_cwt(times: array, data: array, dt: float, datalabel: str="data", log: bool=False, 
                    period: bool=False, tscale: str='sec', scale_value: float=2, 
                    ymax: Union[float, None]=None, normalize: bool=True, plot: bool=False,
                    dj: float=1/48, J: Union[int, None]=None, fmin: Union[float, None]=None,
                    fmax: Union[float, None]=None) -> Dict:
        """
        Compute continuous wavelet transform for time series data
        
        Parameters:
        -----------
        times : array
            Time array
        data : array
            Data array
        dt : float
            Time step
        datalabel : str
            Label for the data in plots
        log : bool
            Use logarithmic scale for power if True
        period : bool
            Plot period instead of frequency if True
        tscale : str
            Time scale ('sec', 'min', 'hour')
        scale_value : float
            Starting scale for wavelet transform
        ymax : float or None
            Maximum y-axis limit (deprecated, use fmax instead)
        normalize : bool
            Normalize wavelet power if True
        plot : bool
            Generate diagnostic plot if True
        dj : float
            Scale resolution (fractional octave step). Default: 1/48 (finer resolution).
            Smaller values = more scales = wider frequency range but slower computation.
        J : int or None
            Number of scales. If None, calculated automatically based on fmin/fmax or default J=168.
            Larger J = lower frequencies covered.
        fmin : float or None
            Desired minimum frequency (Hz). If provided, J will be calculated to cover this frequency.
        fmax : float or None
            Desired maximum frequency (Hz). If provided, scale_value may be adjusted.
            
        Returns:
        --------
        Dict : CWT analysis results
        """
        from pycwt import cwt, Morlet
        from numpy import std, nanmean, nan, nansum, nanmax, polyfit, polyval, array, reshape, nanpercentile, ones
        import numpy as np
        
        def _mask_cone(arr2d: array, ff: array, thresholds: array, fill: float=nan) -> array:
            """Create cone of influence mask"""
            mask = ones(arr2d.shape)
            for k in range(arr2d.shape[0]):
                for l in range(arr2d.shape[1]):
                    if ff[k] < thresholds[l]:
                        mask[k,l] = fill
            return mask

        # Convert inputs to float64
        times = array(times, dtype='float64')
        data = array(data, dtype='float64')
        
        # Detrend and normalize data
        p = polyfit(times - times[0], data, 1)
        data_detrend = data - polyval(p, times - times[0])
        data_norm = data_detrend / data_detrend.std()
        
        # Set up wavelet transform parameters
        mother = Morlet(6)
        s0 = scale_value * dt
        # dj = 1/12 #OLD
        # J = int(7/dj) #OLD

        # Calculate J if not provided or if fmin is specified
        if J is None:
            if fmin is not None:
                # Calculate J to cover down to fmin
                # For Morlet(6), frequency ≈ 1 / (scale * dt)
                # Maximum scale needed: scale_max = 1 / (fmin * dt)
                # J = log2(scale_max / s0) / dj
                scale_max = 1.0 / (fmin * dt)
                J = int(np.ceil(np.log2(scale_max / s0) / dj))
            else:
                # Default: use original calculation
                J = int(7/dj)
        
        # Adjust scale_value if fmax is specified to ensure we cover up to fmax
        if fmax is not None:
            # For Morlet(6), frequency ≈ 1 / (scale * dt)
            # Minimum scale needed: scale_min = 1 / (fmax * dt)
            # Adjust s0 if needed
            scale_min_needed = 1.0 / (fmax * dt)
            if s0 > scale_min_needed:
                # Need smaller starting scale to reach higher frequencies
                s0 = scale_min_needed * 0.9  # Slightly smaller to ensure coverage
        
        # Compute wavelet transform
        wave, scales, freqs, coi, fft, fftfreqs = cwt(
            data_norm, dt=dt, dj=dj, s0=s0, J=J, wavelet=mother
        )

        # Convert time scales if needed
        scale_factors = {'sec': 1, 'min': 60, 'hour': 3600}
        sf = scale_factors.get(tscale, 1)
        times = times / sf
        coi = coi / sf
        periods = 1/freqs / sf
        
        # Calculate power and apply cone of influence
        power = abs(wave)
        if normalize:
            power /= nanmax(power)
        
        cone_mask = _mask_cone(power, freqs, 1/coi)
        power_masked = power * cone_mask
        
        # Calculate global statistics
        global_mean = nanmean(power_masked, axis=1)
        global_sum = nansum(power_masked, axis=1)
        
        # Generate diagnostic plot if requested
        if plot:
            pass
        
        # Prepare output dictionary
        return {
            'times': times,
            'frequencies': freqs,
            'periods': periods,
            'cwt_power': power,
            'cone_mask': cone_mask,
            'cone': 1/coi,
            'global_mean_cwt': global_mean,
            'global_sum_cwt': global_sum
        }


    # # OLD
    # # def compute_backazimuth_old(self, wave_type: str="", baz_step: int=1, baz_win_sec: float=30.0, 
    #                     rotation_data: Stream=None, translation_data: Stream=None,
    #                     baz_win_overlap: float=0.5, tangent_components: str="rotation", verbose: bool=False,
    #                     out: bool=False, cc_threshold: float=0.0) -> Dict:
    #     """
    #     Estimate backazimuth for Love, Rayleigh, or tangent waves
        
    #     Parameters:
    #     -----------
    #     wave_type : str
    #         Type of wave to analyze ('love', 'rayleigh', or 'tangent')
    #     baz_step : int
    #         Step size in degrees for backazimuth search (default: 1)
    #     baz_win_sec : float
    #         Length of backazimuth estimation windows in seconds (default: 30.0)
    #     baz_win_overlap : float
    #         Overlap between windows as fraction (0-1) (default: 0.5)
    #     tangent_components : str
    #         Components to use for tangent method ('rotation' or 'acceleration')
    #     cc_threshold : float
    #         Minimum correlation coefficient threshold (default: 0.0)
    #     out : bool
    #         Return detailed output dictionary if True
            
    #     Returns:
    #     --------
    #     Dict : Backazimuth estimation results
    #     """
    #     import scipy.stats as sts
    #     from obspy.signal.rotate import rotate_ne_rt
    #     from obspy.signal.cross_correlation import correlate, xcorr_max
    #     from numpy import linspace, ones, array, nan, meshgrid, arange, zeros, cov, pi, arctan
    #     from numpy.linalg import eigh
    #     from numpy import argsort

    #     def _padding(_baz, _ccc, _baz_steps, Ndegree=60):

    #         # get lower and upper array that is padded
    #         _baz_lower = np.arange(-Ndegree, 0, _baz_steps)
    #         _baz_upper = np.arange(max(_baz)+_baz_steps, max(_baz)+Ndegree, _baz_steps)

    #         # pad input baz array
    #         _baz_pad = np.append(np.append(_baz_lower, _baz), _baz_upper)

    #         # get sampled size
    #         Nsample = int(Ndegree/_baz_steps)

    #         # pad ccc array by  asymetric reflection
    #         _ccc_pad = np.append(np.append(_ccc[-Nsample-1:-1], _ccc), _ccc[1:Nsample])

    #         return _baz_pad, _ccc_pad

    #     def _get_zero_crossings(arr):

    #         # get nullstellen by sign function and then the difference
    #         nullstellen = np.diff(np.sign(arr))

    #         # there should only be one from negative to positive
    #         # this is a positive value
    #         nullstelle1 = np.argmax(nullstellen)

    #         # look for second zero crossing after the first one
    #         shift = nullstelle1+1
    #         nullstelle2 = np.argmax(abs(nullstellen[shift:]))+ shift

    #         return nullstelle1, nullstelle2

    #     # Check config keywords
    #     keywords = ['tbeg', 'tend', 'sampling_rate',
    #                 'station_latitude', 'station_longitude']

    #     for key in keywords:
    #         if key not in self.attributes():
    #             print(f" -> {key} is missing in config!\n")
    #             return {}  # Return empty dict instead of None

    #     # Store window parameters and ensure baz_step is integer
    #     self.baz_step = int(baz_step)
    #     self.baz_win_sec = baz_win_sec
    #     self.baz_win_overlap = baz_win_overlap

    #     # Prepare streams
    #     if rotation_data is None and translation_data is None:
    #         ACC = self.get_stream("translation").copy()
    #         ROT = self.get_stream("rotation").copy()
    #     elif rotation_data is not None and translation_data is not None:
    #         ACC = translation_data.copy()
    #         ROT = rotation_data.copy()
    #     else:
    #         raise ValueError("no rotation or translation data provided")

    #     if wave_type.lower() == "tangent":
    #         # revert polarity if applied
    #         if hasattr(self, 'pol_applied') and self.pol_applied:
    #             if hasattr(self, 'pol_dict') and self.pol_dict is not None:
    #                 for tr in ACC.select(channel="*Z"):
    #                     if tr.stats.channel[1:] in self.pol_dict:
    #                         tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]
    #                 for tr in ROT.select(channel="*Z"):
    #                     if tr.stats.channel[1:] in self.pol_dict:
    #                         tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]

    #     # sampling rate
    #     df = ROT[0].stats.sampling_rate

    #     # # Get amount of samples for data
    #     n_data = min([len(tr.data) for tr in ROT])

    #     # Prepare backazimuths for loop using integer step size
    #     backazimuths = linspace(0, 360 - self.baz_step, int(360 / self.baz_step))

    #     # Calculate window parameters
    #     win_samples = int(baz_win_sec * df)
    #     overlap_samples = int(win_samples * baz_win_overlap)
    #     step = win_samples - overlap_samples

    #     if step == 0:
    #         print("step is 0, setting to 1")
    #         step = 1

    #     # get amount of windows
    #     n_windows = int((n_data - win_samples) / step) + 1
        
    #     # Prepare data array
    #     corrbaz = ones([backazimuths.size, n_windows])*nan

    #     degrees = []
    #     windows = []
    #     t_center = []
    #     bazs = ones(n_windows)*nan

    #     # _______________________________
    #     # backazimuth estimation with Love, Rayleigh, or tangent waves
    #     # loop over backazimuth degrees
    #     for i_deg in range(0, len(backazimuths)):

    #         degrees.append(i_deg)

    #         # loop over time windows
    #         for i_win in range(0, n_windows):

    #             if i_deg == 0:
    #                 windows.append(i_win)

    #             # update indices
    #             idx1 = i_win * step
    #             idx2 = idx1 + win_samples

    #             # get central time of window
    #             if i_deg == 0:
    #                 t_center.append((idx1 + (idx2 - idx1)/2) /df)

    #             # prepare traces according to selected wave type
    #             if wave_type.lower() == "love":

    #                 if verbose and i_deg == 0 and i_win == 0:
    #                     print(f"> using {wave_type} waves for backazimuth estimation ...")

    #                 # rotate NE to RT
    #                 HR, HT = rotate_ne_rt(
    #                     ACC.select(channel='*N')[0].data,
    #                     ACC.select(channel='*E')[0].data,
    #                     backazimuths[i_deg]
    #                 )

    #                 JZ = ROT.select(channel="*Z")[0].data

    #                 # compute correlation for backazimuth
    #                 ccorr = correlate(
    #                     JZ[idx1:idx2],
    #                     HT[idx1:idx2],
    #                     0, demean=True, normalize='naive', method='auto'
    #                 )

    #                 # get maximum correlation
    #                 xshift, cc_max = xcorr_max(ccorr)

    #                 if xshift != 0 and verbose:
    #                     print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

    #             elif wave_type.lower() == "rayleigh":

    #                 if verbose and i_deg == 0 and i_win == 0:
    #                     print(f"> using {wave_type} waves for backazimuth estimation ...")

    #                 # rotate NE to RT
    #                 JR, JT = rotate_ne_rt(
    #                     ROT.select(channel='*N')[0].data,
    #                     ROT.select(channel='*E')[0].data,
    #                     backazimuths[i_deg]
    #                 )

    #                 HZ = ACC.select(channel="*Z")[0].data

    #                 # compute correlation for backazimuth
    #                 ccorr = correlate(
    #                     HZ[idx1:idx2],
    #                     JT[idx1:idx2],
    #                     0, demean=True, normalize='naive', method='auto'
    #                 )

    #                 # get maximum correlation
    #                 xshift, cc_max = xcorr_max(ccorr)

    #                 if xshift != 0 and self.verbose:
    #                     print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

    #             elif wave_type.lower() == "tangent":

    #                 if verbose and i_deg == 0 and i_win == 0:
    #                     print(f" > using {wave_type} for backazimuth estimation with {tangent_components} components...")

    #                 # no grid search, no degrees loop required
    #                 if i_deg > 0:
    #                     continue

    #                 try:
    #                     N = len(ROT[0].data[idx1:idx2])
    #                 except:
    #                     N = len(ACC[0].data[idx1:idx2])

    #                 # prepare data based on component choice
    #                 dat = zeros((N, 2))

    #                 if tangent_components.lower() == "rotation":
    #                     # Use rotation components (original method)
    #                     dat[:, 0] = ROT.select(channel='*E')[0].data[idx1:idx2]
    #                     dat[:, 1] = ROT.select(channel='*N')[0].data[idx1:idx2]
    #                 elif tangent_components.lower() == "acceleration":
    #                     # Use acceleration components (new option)
    #                     dat[:, 0] = ACC.select(channel='*E')[0].data[idx1:idx2]
    #                     dat[:, 1] = ACC.select(channel='*N')[0].data[idx1:idx2]
    #                 else:
    #                     raise ValueError(f"Invalid tangent_components: {tangent_components}. Use 'rotation' or 'acceleration'")

    #                 # compute covariance
    #                 covar = cov(dat, rowvar=False)

    #                 # get dominant eigenvector
    #                 Cprime, Q = eigh(covar, UPLO='U')

    #                 # sorting
    #                 loc = argsort(abs(Cprime))[::-1]

    #                 # formatting
    #                 Q = Q[:, loc]

    #                 # get backazimuth using tangent of eigenvectors
    #                 baz0 = -arctan((Q[1, 0]/Q[0, 0]))*180/pi

    #                 # if negative due to tangent, then add 180 degrees
    #                 if baz0 <= 0:
    #                     baz0 += 180

    #                 # remove 180° ambiguity using appropriate correlation
    #                 if tangent_components.lower() == "rotation":
    #                     # Original method: rotate rotation components and correlate with acceleration Z
    #                     JR, JT = rotate_ne_rt(
    #                         ROT.select(channel='*N')[0].data,
    #                         ROT.select(channel='*E')[0].data,
    #                         baz0
    #                     )
                        
    #                     HZ = ACC.select(channel="*Z")[0].data

    #                     # correlate with acceleration
    #                     ccorr = correlate(
    #                         HZ[idx1:idx2],
    #                         JT[idx1:idx2],
    #                         0, demean=True, normalize='naive', method='auto'
    #                     )

    #                 # remove 180° ambiguity using closest to theoretical backazimuth
    #                 else:
    #                     if self.event_info and 'backazimuth' in self.event_info:
    #                         # compute rmse of baz0 and self.event_info['backazimuth']
    #                         rmse = np.sqrt(np.mean((baz0 - self.event_info['backazimuth'])**2))
                            
    #                         # compute rmse of baz0 + 180 and self.event_info['backazimuth']
    #                         rmse_180 = np.sqrt(np.mean((baz0 + 180 - self.event_info['backazimuth'])**2))

    #                         # choose the one with the least rmse and set cc_max to 0 (dummy value)
    #                         if rmse_180 < rmse:
    #                             ccorr = [1] #[1e-6]
    #                         else:
    #                             ccorr = [-1] #[1e-6]

    #                 # get maximum correlation
    #                 xshift, cc_max = xcorr_max(ccorr)

    #                 # if correlation positive add 180 degrees
    #                 if cc_max > 0:
    #                     baz0 += 180

    #                 # take absolute value of correlation for better visualization
    #                 cc_max = abs(cc_max)

    #             else:
    #                 print(f" -> unknown wave type: {wave_type}!")
    #                 continue

    #             corrbaz[i_deg, i_win] = cc_max

    #             if wave_type.lower() == "tangent":
    #                 bazs[i_win] = baz0

    #     # extract maxima
    #     if wave_type.lower() == "tangent":
    #         maxbaz = bazs
    #         maxcorr = corrbaz[0, :]
    #     else:
    #         maxbaz = array([backazimuths[corrbaz[:, w1].argmax()] for w1 in range(0, n_windows)])
    #         maxcorr = array([max(corrbaz[:, w1]) for w1 in range(0, n_windows)])

    #     # extract mid baz based on zero crossing
    #     midbaz = np.zeros(n_windows)
    #     midcorr = np.zeros(n_windows)

    #     # loop over all windows
    #     if wave_type.lower() in ["love", "rayleigh"]:

    #         for _k, _win in enumerate(range(0, n_windows)):

    #             # pad baz array and cc array
    #             baz_pad, cc_pad = _padding(backazimuths, corrbaz[:, _win], baz_step, Ndegree=180)

    #             # get zero crossings of cc function
    #             null1, null2 = _get_zero_crossings(cc_pad)

    #             # get middle baz
    #             baz_lower = baz_pad[null1]
    #             baz_upper = baz_pad[null2]
    #             baz_mid = (baz_upper - baz_lower)/2 + baz_lower

    #             # avoid maximum outside range 0 - 360 due to padding
    #             if baz_mid < 0:
    #                 baz_mid += 360

    #             if baz_mid > 360:
    #                 baz_mid -= 360

    #             # write to array
    #             midbaz[_k] = baz_mid

    #             # get cc value at baz_mid and write to array
    #             idx_mid = int((null2 - null1)/2 + null1)
    #             midcorr[_k] = cc_pad[idx_mid]

    #     # create mesh grid
    #     t_win = arange(0, baz_win_sec*n_windows+baz_win_sec, baz_win_sec)
    #     t_win = t_win[:-1]+baz_win_sec/2
    #     grid = meshgrid(t_win, backazimuths)

    #     # add one element for axes
    #     windows.append(windows[-1]+1)
    #     degrees.append(degrees[-1]+self.baz_step)

    #     # prepare results
    #     results = {
    #         'baz_mesh': grid,
    #         'baz_corr': corrbaz,
    #         'baz_time': t_win,
    #         'acc': ACC,
    #         'rot': ROT,
    #         'twin_center': np.array(t_center),
    #         'cc_max_y': maxbaz,
    #         'baz_max': maxbaz,
    #         'cc_max': maxcorr,
    #         'baz_mid': midbaz,
    #         'cc_mid': midcorr,
    #         'component_type': tangent_components if wave_type.lower() == "tangent" else None,
    #         'parameters': {
    #             'baz_win_sec': baz_win_sec,
    #             'baz_win_overlap': baz_win_overlap,
    #             'baz_step': baz_step,
    #             'wave_type': wave_type,
    #         }
    #     }

    #     # add results to object
    #     self.baz_results[wave_type] = results

    #     # return output if out required
    #     if out:
    #         return results

    # # OLD
    # # def compute_velocities_optimized(self, rotation_data: Stream=None, translation_data: Stream=None,
    #                                  wave_type: str='love', baz_results: Dict=None, baz_mode: str='mid',
    #                                  method: str='odr', cc_threshold: float=0.0, r_squared_threshold: float=0.0, 
    #                                  zero_intercept: bool=True, verbose: bool=False, plot: bool=False) -> Dict:
    #     """
    #     Compute phase velocities in time intervals for Love or Rayleigh waves
        
    #     Parameters:
    #     -----------
    #     rotation_data : Stream, optional
    #         Rotational data stream (if None, uses self.get_stream("rotation", raw=True))
    #     translation_data : Stream, optional
    #         Translation data stream (if None, uses self.get_stream("translation", raw=True))
    #     wave_type : str
    #         Type of wave to analyze ('love' or 'rayleigh')
    #     baz_results : Dict
    #         Dictionary containing backazimuth results
    #     baz_mode : str
    #         Mode to use for backazimuth selection ('max' or 'mid')
    #     method : str
    #         Method to use for velocity computation ('odr', 'ransac', or 'theilsen')
    #     cc_threshold : float, optional
    #         Minimum cross-correlation coefficient to consider, by default 0.0
    #     r_squared_threshold : float, optional
    #         Minimum R-squared value for regression quality, by default 0.0
    #     zero_intercept : bool
    #         Force intercept to be zero if True
    #     verbose : bool
    #         Print regression results if True
    #     plot : bool
    #         Plot regression results if True
    #     Returns:
    #     --------
    #     Dict
    #         Dictionary containing:
    #         - time : array of time points
    #         - velocity: array of phase velocities
    #         - ccoef: array of cross-correlation coefficients
    #         - backazimuth: array of backazimuths
    #         - r_squared: array of R-squared values
    #         - parameters: dictionary of parameters
    #     """
    #     import numpy as np
    #     from tqdm import tqdm
    #     from obspy.signal.rotate import rotate_ne_rt

    #     # Get streams - use object data if not provided
    #     if rotation_data is None:
    #         rot = self.get_stream("rotation", raw=False).copy()
    #     else:
    #         rot = rotation_data.copy()
        
    #     if translation_data is None:
    #         tra = self.get_stream("translation", raw=True).copy()
    #     else:
    #         tra = translation_data.copy()

    #     # Validate inputs
    #     if baz_results is None:
    #         raise ValueError("Backazimuth results must be provided")
    #     if wave_type.lower() not in ['love', 'rayleigh']:
    #         raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
    #     if baz_mode.lower() not in ['max', 'mid']:
    #         raise ValueError(f"Invalid baz mode: {baz_mode}. Use 'max' or 'mid'")
    #     if method.lower() not in ['odr', 'ransac', 'theilsen']:
    #         raise ValueError(f"Invalid method: {method}. Use 'odr' or 'ransac' or 'theilsen'")

    #     # Get sampling rate and validate
    #     df = rot[0].stats.sampling_rate
    #     if df <= 0:
    #         raise ValueError(f"Invalid sampling rate: {df}")

    #     # Extract parameters from baz_results
    #     try:
    #         win_time_s = baz_results['parameters']['baz_win_sec']
    #         overlap = baz_results['parameters']['baz_win_overlap']
    #         ttt = baz_results['twin_center']

    #         if baz_mode.lower() == 'max':
    #             baz = baz_results['baz_max']
    #             ccc = baz_results['cc_max']
    #         else:  # 'mid'
    #             baz = baz_results['baz_mid']
    #             ccc = baz_results['cc_mid']
    #     except KeyError as e:
    #         raise ValueError(f"Missing required key in baz_results: {e}")

    #     # Validate array lengths
    #     if not (len(baz) == len(ttt) == len(ccc)):
    #         raise ValueError("Inconsistent lengths in backazimuth results")

    #     # Get components
    #     try:
    #         rot_z = rot.select(channel="*Z")[0].data
    #         acc_z = tra.select(channel="*Z")[0].data
    #     except Exception as e:
    #         raise RuntimeError(f"Error accessing vertical components: {str(e)}")

    #     # Calculate window parameters
    #     win_samples = int(win_time_s * df)
    #     if win_samples <= 0:
    #         raise ValueError(f"Invalid window size: {win_samples} samples")

    #     overlap_samples = int(win_samples * overlap)
    #     step = win_samples - overlap_samples

    #     # number of windows
    #     n_windows = len(ttt)

    #     # Initialize arrays
    #     times = np.zeros(n_windows)
    #     velocities = np.zeros(n_windows)
    #     cc_coeffs = np.zeros(n_windows)
    #     r_squared = np.zeros(n_windows)

    #     # Loop through windows
    #     velocities = np.ones_like(baz) * np.nan

    #     for i, (_baz, _ttt, _ccc) in enumerate(zip(baz, ttt, ccc)):
    #         i1 = i * step
    #         i2 = i1 + win_samples

    #         # apply cc threshold
    #         if _ccc <= cc_threshold:
    #             velocities[i] = np.nan
    #             r_squared[i] = np.nan
    #             continue

    #         # Rotate components to radial-transverse
    #         if wave_type.lower() == 'rayleigh':
    #             rot_r, rot_t = rotate_ne_rt(
    #                 rot.select(channel='*N')[0].data,
    #                 rot.select(channel='*E')[0].data,
    #                 _baz
    #             )
    #         elif wave_type.lower() == 'love':
    #             acc_r, acc_t = rotate_ne_rt(
    #                 tra.select(channel='*N')[0].data,
    #                 tra.select(channel='*E')[0].data,
    #                 _baz
    #             )

    #         # Compute velocity using amplitude ratio
    #         if wave_type.lower() == 'love':
    #             # get velocity from amplitude ratio via regression
    #             reg_results = self.regression(
    #                 rot_z[i1:i2],
    #                 0.5*acc_t[i1:i2],
    #                 method=method.lower(),
    #                 zero_intercept=zero_intercept,
    #                 verbose=verbose
    #             )
    #             if plot:
    #                 plt.figure()
    #                 plt.scatter(rot_z[i1:i2], 0.5*acc_t[i1:i2], color='black')
    #                 plt.plot(rot_z[i1:i2], reg_results['slope']*rot_z[i1:i2] + reg_results['intercept'], color='red')
    #                 plt.title(f"Love Wave Regression: baz={_baz:.2f}°, slope={reg_results['slope']:.2f}, R²={reg_results['r_squared']:.4f}")
    #                 plt.xlabel("Vertical Rotation")
    #                 plt.ylabel("Transverse Acceleration")
    #                 plt.show()
  
            
    #         elif wave_type.lower() == 'rayleigh':
    #             # get velocity from amplitude ratio via regression
    #             reg_results = self.regression(
    #                 rot_t[i1:i2],
    #                 acc_z[i1:i2],
    #                 method=method.lower(),
    #                 zero_intercept=zero_intercept,
    #                 verbose=verbose
    #             )
    #             if plot:
    #                 plt.figure()
    #                 plt.scatter(rot_t[i1:i2], acc_z[i1:i2], color='black')
    #                 plt.plot(rot_t[i1:i2], reg_results['slope']*rot_t[i1:i2] + reg_results['intercept'], color='red')
    #                 plt.title(f"Rayleigh Wave Regression: slope={reg_results['slope']:.6f}, R²={reg_results['r_squared']:.4f}")
    #                 plt.xlabel("Transverse Rotation")
    #                 plt.ylabel("Vertical Acceleration")
    #                 plt.show()
            
    #         # add r_squared
    #         r_squared[i] = reg_results['r_squared']

    #         # Apply R² threshold filter
    #         if reg_results['r_squared'] < r_squared_threshold:
    #             velocities[i] = np.nan
    #         else:
    #             velocities[i] = reg_results['slope']

    #     return {
    #         'time': ttt,
    #         'velocity': velocities,
    #         'ccoef': ccc,
    #         'terr': np.full(len(ttt), win_time_s/2),
    #         'backazimuth': baz,
    #         'r_squared': r_squared,
    #         'parameters': {
    #             'wave_type': wave_type,
    #             'win_time_s': win_time_s,
    #             'overlap': overlap,
    #             'method': method,
    #             'baz_mode': baz_mode,
    #             'r_squared_threshold': r_squared_threshold
    #         }
    #     }

    # # OLD
    # # def _process_frequency_band(self, freq_params, wave_type, t_win_factor, overlap, baz_mode, method, cc_threshold, rot_data, tra_data):
    #     """
    #     Process a single frequency band for parallel computation.
        
    #     Parameters
    #     ----------
    #     freq_params : tuple
    #         (lower_freq, upper_freq, center_freq)
    #     wave_type : str
    #         Type of wave to analyze
    #     t_win_factor : float
    #         Window length factor
    #     overlap : float
    #         Window overlap fraction
    #     baz_mode : str
    #         Backazimuth computation mode
    #     method : str
    #         Velocity computation method
    #     cc_threshold : float
    #         Cross-correlation threshold
    #     rot_data : Stream
    #         Rotation data stream
    #     tra_data : Stream
    #         Translation data stream
            
    #     Returns
    #     -------
    #     dict or None
    #         Results for this frequency band
    #     """

    #     def filter_data(st, fmin: Optional[float]=0.1, fmax: Optional[float]=0.5):
    #         """Filter stream data with error handling"""
    #         try:
    #             st = st.copy()  # Don't modify original
    #             st = st.detrend("linear")
    #             st = st.detrend("demean")
    #             st = st.taper(0.05, type='cosine')

    #             if fmin is not None and fmax is not None:
    #                 st = st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True) 
    #             elif fmin is not None:
    #                 st = st.filter("lowpass", freq=fmin, corners=4, zerophase=True)
    #             elif fmax is not None:
    #                 st = st.filter("highpass", freq=fmax, corners=4, zerophase=True)

    #             return st
    #         except Exception as e:
    #             raise RuntimeError(f"Error filtering data: {str(e)}")

    #     # get frequency band parameters
    #     fl, fu, fc = freq_params
        
    #     # Calculate window length based on center frequency
    #     win_time_s = int(t_win_factor / fc)
        
    #     # define output
    #     results = {
    #         'frequency': fc,
    #         'backazimuth': np.nan,
    #         'cc_values': np.nan,
    #         'velocity': np.nan,
    #         'baz_mad': np.nan,
    #         'ccc_mad': np.nan,
    #         'vel_mad': np.nan,
    #         'win_time_s': win_time_s
    #     }

    #     # get raw data
    #     rot = self.get_stream("rotation", raw=True)
    #     tra = self.get_stream("translation", raw=True)

    #     # Filter data
    #     rot = filter_data(rot, fmin=fl, fmax=fu)
    #     tra = filter_data(tra, fmin=fl, fmax=fu)

    #     # Compute backazimuths
    #     baz_results = self.compute_backazimuth(
    #         rotation_data=rot,
    #         translation_data=tra,
    #         wave_type=wave_type,
    #         baz_step=1,
    #         baz_win_sec=win_time_s,
    #         baz_win_overlap=overlap,
    #         out=True
    #     )

    #     if baz_results is None:
    #         print("No baz results")
    #         return results

    #     # Compute velocities
    #     vel_results = self.compute_velocities_optimized(
    #         rotation_data=rot,
    #         translation_data=tra,
    #         wave_type=wave_type,
    #         baz_results=baz_results,
    #         baz_mode=baz_mode,
    #         method=method,
    #         cc_threshold=cc_threshold
    #     )
        
    #     if vel_results is None:
    #         print("No vel results")
    #         return results
        
    #     # Apply threshold and clean data
    #     mask = vel_results['cc_value'] > cc_threshold
    #     baz_cc_filtered = vel_results['backazimuth'][mask]
    #     ccc_cc_filtered = vel_results['cc_value'][mask]
    #     vel_cc_filtered = vel_results['velocity'][mask]
        
    #     # Remove NaN values
    #     vel_valid = vel_cc_filtered[~np.isnan(vel_cc_filtered)]
    #     baz_valid = baz_cc_filtered[~np.isnan(baz_cc_filtered)]
    #     ccc_valid = ccc_cc_filtered[~np.isnan(ccc_cc_filtered)]
        
    #     # compute median
    #     baz_median = np.nanmedian(baz_valid)
    #     ccc_median = np.nanmedian(ccc_valid)
    #     vel_median = np.nanmedian(vel_valid)

    #     # compute mad
    #     baz_mad = np.nanmedian(np.abs(baz_valid - baz_median))
    #     ccc_mad = np.nanmedian(np.abs(ccc_valid - ccc_median))
    #     vel_mad = np.nanmedian(np.abs(vel_valid - vel_median))

    #     # update results
    #     results = {
    #         'frequency': fc,
    #         'backazimuth': baz_median,
    #         'cc_values': ccc_median,
    #         'velocity': vel_median,
    #         'baz_mad': baz_mad,
    #         'ccc_mad': ccc_mad,
    #         'vel_mad': vel_mad,
    #         'win_time_s': win_time_s
    #     }

    #     return results

    # # OLD
    # # def compute_frequency_dependent_parameters_parallel(self, wave_type: str='love', fbands: Dict=None, 
    #                                             t_win_factor: float=2, overlap: float=0.5, baz_mode: str='mid', 
    #                                             method: str='odr', cc_threshold: float=0.0, n_jobs: int=-1) -> Dict:
    #     """
    #     Parallel version of compute_frequency_dependent_parameters.
    #     Uses multiprocessing to speed up computation across frequency bands.
        
    #     Parameters
    #     ----------
    #     wave_type : str, optional
    #         Type of wave to analyze ('love' or 'rayleigh'), by default 'love'
    #     fbands : Dict, optional
    #         Dictionary containing frequency band parameters:
    #         - 'fmin': minimum frequency
    #         - 'fmax': maximum frequency
    #         - 'octave_fraction': fraction of octave for band division
    #         If None, default values will be used
    #     t_win_factor : float, optional
    #         Window length factor for backazimuth computation, by default 2.0
    #         The actual window length will be t_win_factor / center_frequency
    #     overlap : float, optional
    #         Overlap fraction between windows for backazimuth computation, by default 0.5
    #     cc_threshold : float, optional
    #         Minimum cross-correlation coefficient to consider, by default 0.0
    #     baz_mode : str, optional
    #         Mode of backazimuth computation ('max' or 'mid'), by default 'mid'
    #     method : str, optional
    #         Method to use for velocity computation ('odr' or 'ransac'), by default 'odr'
    #     n_jobs : int, optional
    #         Number of parallel jobs. -1 means using all processors, by default -1
    #     """
    #     import numpy as np
    #     from multiprocessing import Pool, cpu_count
    #     from functools import partial
    #     from tqdm import tqdm

    #     # Validate input parameters
    #     if wave_type.lower() not in ['love', 'rayleigh']:
    #         raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
    #     if baz_mode.lower() not in ['max', 'mid']:
    #         raise ValueError(f"Invalid baz mode: {baz_mode}. Use 'max' or 'mid'")
    #     if method.lower() not in ['odr', 'ransac']:
    #         raise ValueError(f"Invalid method: {method}. Use 'odr' or 'ransac'")
    #     if not 0 <= overlap < 1:
    #         raise ValueError(f"Overlap must be between 0 and 1, got {overlap}")
    #     if t_win_factor <= 0:
    #         raise ValueError(f"Window length factor must be positive, got {t_win_factor}")

    #     # Default frequency bands if not provided
    #     if fbands is None:
    #         fbands = {
    #             'fmin': 0.01,
    #             'fmax': 1.0,
    #             'octave_fraction': 3
    #         }

    #     # Get frequency bands
    #     flower, fupper, fcenter = self.get_octave_bands(
    #         fmin=fbands['fmin'],
    #         fmax=fbands['fmax'],
    #         faction_of_octave=fbands['octave_fraction']
    #     )

    #     # Get raw data once
    #     rot_data = self.get_stream("rotation", raw=True)
    #     tra_data = self.get_stream("translation", raw=True)

    #     # Prepare frequency band parameters
    #     freq_params = list(zip(flower, fupper, fcenter))
        
    #     # Set up parallel processing
    #     if n_jobs == -1:
    #         n_jobs = cpu_count()
    #     n_jobs = min(n_jobs, len(freq_params))  # Don't use more processes than frequency bands
        
    #     print(f"Processing {len(freq_params)} frequency bands using {n_jobs} processes...")
        
    #     # Create partial function with fixed parameters
    #     process_func = partial(
    #         self._process_frequency_band,
    #         wave_type=wave_type,
    #         t_win_factor=t_win_factor,
    #         overlap=overlap,
    #         baz_mode=baz_mode,
    #         method=method,
    #         cc_threshold=cc_threshold,
    #         rot_data=rot_data,
    #         tra_data=tra_data
    #     )

    #     # Initialize results
    #     results = {
    #         'frequency': [],
    #         'backazimuth': [],
    #         'cc_values': [],
    #         'velocity': [],
    #         'baz_mad': [],
    #         'ccc_mad': [],
    #         'vel_mad': [],
    #         'parameters': {
    #             'wave_type': wave_type,
    #             't_win_factor': t_win_factor,
    #             'overlap': overlap,
    #             'baz_mode': baz_mode,
    #             'method': method,
    #             'cc_threshold': cc_threshold,
    #             'frequency_bands': {
    #                 'lower': flower.tolist(),
    #                 'upper': fupper.tolist(),
    #                 'center': fcenter.tolist()
    #             }
    #         }
    #     }

    #     # Process frequency bands in parallel with progress bar
    #     with Pool(n_jobs) as pool:
    #         # Use imap_unordered for better progress tracking
    #         for freq_result in tqdm(pool.imap_unordered(process_func, freq_params), 
    #                               total=len(freq_params),
    #                               desc="Processing frequency bands",
    #                               unit="band"):
    #             if freq_result is not None:
    #                 results['frequency'].append(freq_result['frequency'])
    #                 results['backazimuth'].append(freq_result['backazimuth'])
    #                 results['cc_values'].append(freq_result['cc_values'])
    #                 results['velocity'].append(freq_result['velocity'])
    #                 results['baz_mad'].append(freq_result['baz_mad'])
    #                 results['ccc_mad'].append(freq_result['ccc_mad'])
    #                 results['vel_mad'].append(freq_result['vel_mad'])

    #                 # Print detailed progress for each completed band
    #                 print(f"✓ Processed {freq_result['frequency']:.3f} Hz (T={freq_result['win_time_s']:.1f}s)")

    #     if not results['frequency']:
    #         raise RuntimeError("No valid results for any frequency band")

    #     print(f"Completed processing {len(results['frequency'])} frequency bands successfully.")
    #     return results

    # # OLD
    # # def compute_odr(self, x_array: ndarray, y_array: ndarray, xerr: Union[float, ndarray]=None, 
    #                yerr: Union[float, ndarray]=None, zero_intercept: bool=False) -> Dict:
    #     """
    #     Compute Orthogonal Distance Regression between two arrays
        
    #     Parameters:
    #     -----------
    #     x_array : numpy.ndarray
    #         Independent variable data
    #     y_array : numpy.ndarray
    #         Dependent variable data
    #     xerr : float or numpy.ndarray, optional
    #         Error in x. If None, uses standard deviation of x
    #     yerr : float or numpy.ndarray, optional
    #         Error in y. If None, uses standard deviation of y
    #     zero_intercept : bool, optional
    #         Force intercept to be zero if True
            
    #     Returns:
    #     --------
    #     Dict
    #         Dictionary containing regression results
    #     """
    #     from scipy import odr
    #     import numpy as np
    #     from scipy.stats import pearsonr

    #     # Set default errors if not provided
    #     if xerr is None:
    #         xerr = np.std(x_array)
    #     if yerr is None:
    #         yerr = np.std(y_array)
            
    #     # Convert to arrays if scalar errors provided
    #     if np.isscalar(xerr):
    #         xerr = np.ones_like(x_array) * xerr
    #     if np.isscalar(yerr):
    #         yerr = np.ones_like(y_array) * yerr
            
    #     # Define model function
    #     def linear_func(params, x):
    #         if zero_intercept:
    #             return params[0] * x
    #         else:
    #             return params[0] * x + params[1]
            
    #     # Create model
    #     linear = odr.Model(linear_func)
        
    #     # Create data object
    #     data = odr.RealData(x_array, y_array, sx=xerr, sy=yerr)
        
    #     # Set initial parameters
    #     if zero_intercept:
    #         beta0 = [np.mean(y_array)/np.mean(x_array)]
    #     else:
    #         beta0 = [np.mean(y_array)/np.mean(x_array), 0]
            
    #     # Create ODR object and run regression
    #     odr_obj = odr.ODR(data, linear, beta0=beta0)
    #     output = odr_obj.run()
        
    #     # Calculate R-squared
    #     if zero_intercept:
    #         y_fit = output.beta[0] * x_array
    #         intercept = 0
    #         intercept_err = 0
    #     else:
    #         y_fit = output.beta[0] * x_array + output.beta[1]
    #         intercept = output.beta[1]
    #         intercept_err = output.sd_beta[1]
        
    #     # Calculate R-squared
    #     r, _ = pearsonr(x_array, y_array)
    #     r_squared = r**2

    #     # Prepare results
    #     results = {
    #         'slope': output.beta[0],
    #         'intercept': intercept,
    #         'slope_err': output.sd_beta[0],
    #         'intercept_err': intercept_err,
    #         'r_squared': r_squared,
    #         'fit_params': {
    #             'x': x_array,
    #             'y': y_array,
    #             'y_fit': y_fit,
    #             'xerr': xerr,
    #             'yerr': yerr
    #         }
    #     }
        
    #     return results

    # # OLD
    # # def compute_regression(self, x_array: ndarray, y_array: ndarray, 
    #                       method: str='ransac', zero_intercept: bool=False,
    #                       trials: int=1000, min_samples: int=2,
    #                       residual_threshold: float=None) -> Dict:
    #     """
    #     Compute linear regression using RANSAC or Theil-Sen methods
        
    #     Parameters:
    #     -----------
    #     x_array : numpy.ndarray
    #         Independent variable data
    #     y_array : numpy.ndarray
    #         Dependent variable data
    #     method : str
    #         Regression method ('ransac' or 'theilsen')
    #     zero_intercept : bool
    #         Force intercept to be zero if True
    #     trials : int
    #         Number of trials for RANSAC (only used if method='ransac')
    #     min_samples : int
    #         Minimum samples for RANSAC (only used if method='ransac')
    #     residual_threshold : float or None
    #         Maximum residual for RANSAC inliers. If None, uses median absolute deviation
            
    #     Returns:
    #     --------
    #     Dict
    #         Dictionary containing:
    #         - slope: regression slope
    #         - intercept: regression intercept (0 if zero_intercept=True)
    #         - r_squared: R-squared value of fit
    #         - inlier_mask: boolean mask of inliers (only for RANSAC)
    #         - fit_params: dictionary with fit parameters
    #     """
    #     from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
    #     import numpy as np
        
    #     # Reshape arrays for sklearn
    #     X = x_array.reshape(-1, 1)
    #     y = y_array.reshape(-1, 1)
        
    #     if zero_intercept:
    #         # For zero intercept, modify data to force through origin
    #         X = np.hstack([X, np.zeros_like(X)])
        
    #     # Select regression method
    #     if method.lower() == 'ransac':
    #         regressor = RANSACRegressor(
    #             random_state=42,
    #             max_trials=trials,
    #             min_samples=min_samples,
    #             residual_threshold=residual_threshold,
    #             loss='absolute_error'
    #         )
    #     elif method.lower() == 'theilsen':
    #         regressor = TheilSenRegressor(
    #             random_state=42,
    #             max_iter=trials,
    #             n_subsamples=min_samples
    #         )
    #     else:
    #         raise ValueError(f"Invalid method: {method}. Use 'ransac' or 'theilsen'")
        
    #     # Fit regression
    #     if zero_intercept:
    #         regressor.fit(X[:, 0].reshape(-1, 1), y)
    #         slope = regressor.estimator_.coef_[0]
    #         intercept = regressor.estimator_.intercept_
    #     else:
    #         regressor.fit(X, y)
    #         slope = regressor.estimator_.coef_[0]
    #         intercept = regressor.estimator_.intercept_
        
    #     # Calculate fitted values
    #     y_fit = slope * x_array + intercept
        
    #     # Calculate R-squared
    #     ss_res = np.sum((y_array - y_fit) ** 2)
    #     ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
    #     r_squared = 1 - (ss_res / ss_tot)
        
    #     # Prepare results dictionary
    #     results = {
    #         'slope': slope,
    #         'intercept': intercept,
    #         'r_squared': r_squared,
    #         'fit_params': {
    #             'x': x_array,
    #             'y': y_array,
    #             'y_fit': y_fit,
    #             'method': method
    #         }
    #     }
        
    #     # Add RANSAC-specific results
    #     if method.lower() == 'ransac':
    #         results['inlier_mask'] = regressor.inlier_mask_
    #         results['fit_params']['n_inliers'] = np.sum(regressor.inlier_mask_)
    #         results['fit_params']['n_trials'] = regressor.n_trials_
        
    #     return results

