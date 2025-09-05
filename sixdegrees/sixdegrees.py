'''
@package sixdegrees
@copyright:
    Andreas Brotzer (rotzleffel@tutanota.com)
@license:
    GNU Lesser General Public License, Version 3
    (http://www.gnu.org/copyleft/lesser.html)
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

        # add dummy trace
        self.dummy_trace = conf.get('dummy_trace', False)

        # rotate_zne    
        self.rotate_zne = conf.get('rotate_zne', False)

        # remove_response_tra
        self.tra_remove_response = conf.get('tra_remove_response', False)

        # remove_response_rot
        self.rot_remove_response = conf.get('rot_remove_response', False)

        # output type for remove response
        self.tra_output = conf.get('tra_output', "ACC")

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
                'distance_km': float(distance) / 1000,
                'distance_deg': kilometers2degrees(float(distance) / 1000),
                'azimuth': float(az),
                'backazimuth': float(baz),
                'catalog': str(self.base_catalog),
                'event_id': str(event.resource_id.id)
            }
            
            if self.verbose:
                print(f"Found event:")
                print(f"Origin time: {self.event_info['origin_time']}")
                print(f"Magnitude: {self.event_info['magnitude']} {self.event_info['magnitude_type']}")
                print(f"Location: {self.event_info['latitude']:.3f}°N, {self.event_info['longitude']:.3f}°E")
                print(f"Depth: {self.event_info['depth_km']:.1f} km")
                print(f"Epicentral Distance: {self.event_info['distance_km']:.1f} km")
                print(f"Epicentral Distance: {self.event_info['distance_deg']:.1f}°")
                print(f"Backazimuth: {self.event_info['backazimuth']:.1f}°")
            
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

    def get_component_lag(self, normalize: bool=True, baz: float=None, correct: bool=True) -> Dict:
        """
        Get lag between rotation and translation components
        
        Parameters:
        -----------
        component : str
            Component to analyze ('Z', 'N', 'E', or 'T')
        normalize : bool
            Normalize cross-correlation
            
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

        # Get components
        rot = self.get_stream("rotation").copy()
        tra = self.get_stream("translation").copy()
        
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
        cc = correlate(tra_z, rot_t, len(rot_t), normalize=normalize)
        lag_samples_h, cc_max_h = xcorr_max(cc)
        
        # Convert to time
        lag_time_h = lag_samples_h / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-T & ACC-Z:  lag_time: {lag_time_h} s, lag_samples: {lag_samples_h}, cc_max: {cc_max_h:.2f}")

        # Compute cross-correlation
        cc = correlate(tra_t, rot_z, len(rot_z), normalize=normalize)
        lag_samples_z, cc_max_z = xcorr_max(cc)
        
        # Convert to time
        lag_time_z = lag_samples_z / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-Z & ACC-T:  lag_time: {lag_time_z} s, lag_samples: {lag_samples_z}, cc_max: {cc_max_z:.2f}")
       
        # shift rotataion waveforms
        if correct:
            for tr in rot:
                if tr.stats.channel.endswith("Z"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_z
            
                if tr.stats.channel.endswith("N") or tr.stats.channel.endswith("E"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_h

        # shift rotataion waveforms for raw stream
        rot_raw = self.get_stream("rotation", raw=True)
        if correct:
            for tr in rot_raw:
                if tr.stats.channel.endswith("Z"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_z
            
                if tr.stats.channel.endswith("N") or tr.stats.channel.endswith("E"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_h
              
        # update and trim raw stream
        if correct:
            # reassign raw stream
            self.st0 = rot_raw + self.get_stream("translation", raw=True)
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

        # resample stream
        if resample_rate is not None:
            if self.verbose:
                print(f"-> resampling stream to {resample_rate} Hz")
            for tr in st0:
                tr = tr.detrend("demean")
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

        # add dummy trace
        if self.dummy_trace:
            for dummy in self.dummy_trace:
                if dummy[-2] == "H":
                    if self.verbose:
                        print(f"-> adding dummy trace: {dummy}")
                    tra = self.add_dummy_trace(tra, dummy)

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

        # add dummy trace
        if self.dummy_trace:
            for dummy in self.dummy_trace:
                if dummy[-2] == "J":
                    if self.verbose:
                        print(f"-> adding dummy trace: {dummy}")
                    rot = self.add_dummy_trace(rot, dummy)

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

    def filter_data(self, fmin: Optional[float]=0.1, fmax: Optional[float]=0.5, output: bool=False):

        # reset stream to raw stream
        self.st = self.st0.copy()

        # set fmin and fmax
        self.fmin = fmin
        self.fmax = fmax

        # detrend and filter
        self.st = self.st.detrend("linear")
        self.st = self.st.detrend("demean")
        self.st = self.st.taper(0.05)

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
            print(f" -> stream size inconsistent: {n_samples}")

            # if difference not larger than one -> adjust
            if any([abs(x-n_samples[0]) > 1 for x in n_samples]):

                # set to common minimum interval
                if set_common:
                    _tbeg = max([tr.stats.starttime for tr in self.st])
                    _tend = min([tr.stats.endtime for tr in self.st])
                    self.st = self.st.trim(_tbeg, _tend, nearest_sample=True)
                    print(f"  -> adjusted: {_get_size(self.st)}")

                    if set_interpolate:
                        _times = arange(0, min(_get_size(self.st)), self.st[0].stats.delta)
                        for tr in self.st:
                            tr.data = interp(_times, tr.times(reftime=_tbeg), tr.data)
            else:
                # adjust for difference of one sample
                for tr in self.st:
                    tr.data = tr.data[:min(n_samples)]
                print(f"  -> adjusted: {_get_size(self.st)}")

    def correct_tilt(self, g: float=9.81, raw: bool=False):
        '''
        Correct tilt of data
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

        def filter_stream(stream: Stream, fmin: float, fmax: float) -> Stream:
            stream_copy = stream.copy()
            stream_copy.detrend('linear')
            stream_copy.taper(max_percentage=0.01)
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
                else:  # rayleigh
                    signal1 = rot_t
                    signal2 = acc_z[i1:i2, np.newaxis]
                
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

    def regression(self, features: List[str], target: str, reg: str = "theilsen", 
              zero_intercept: bool = False, verbose: bool = True) -> Dict:
        """
        Perform regression analysis using various methods.
        
        Args:
            features: List of feature columns
            target: Target variable column
            reg: Regression method ('ols', 'ransac', 'theilsen', 'odr')
            zero_intercept: Force intercept through zero if True
            verbose: Print regression results if True
        
        Returns:
            Dictionary containing regression results including:
            - model: Fitted regression model
            - r2: R-squared score
            - tp: Time points
            - dp: Model predictions
            - slope: Regression slope(s)
            - inter: Y-intercept (0 if zero_intercept=True)
        """
        from sklearn import linear_model
        from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor
        from scipy import odr
        from numpy import array, std, ones, mean, ones_like
        import pandas as pd

        # Validate regression method
        valid_methods = ['ols', 'ransac', 'theilsen', 'odr']
        if reg.lower() not in valid_methods:
            raise ValueError(f"Invalid regression method. Must be one of {valid_methods}")

        # Create DataFrame if needed
        if isinstance(features, pd.DataFrame):
            _df = features.copy()
        else:
            # Create DataFrame from stream data
            data = {}
            for tr in self.st:
                data[tr.stats.channel] = tr.data
            _df = pd.DataFrame(data)
            _df['time'] = self.st[0].times()

        # Remove time and target from features if present
        if target in features:
            features.remove(target)
        if "time" in features:
            features.remove("time")

        # Define x and y data
        X = _df[features].values.reshape(-1, len(features))
        y = _df[target].values

        # Initialize predictions list and model
        model_predict = []
        model = None

        # Rest of regression code remains the same as in baroseis...
        if reg.lower() == "ols":
            model = linear_model.LinearRegression(fit_intercept=not zero_intercept)
            model.fit(X, y)
            if verbose:
                print("R2:", model.score(X, y))
                if not zero_intercept:
                    print("X0:", model.intercept_)
                print("Coef: ", model.coef_)
                for _f, _c in zip(features, model.coef_):
                    print(f"{_f} : {_c}")
            
            # Make predictions
            for o, row in _df[features].iterrows():
                x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
                model_predict.append(model.predict(x_pred)[0])

        elif reg.lower() == "ransac":
            try:
                model = RANSACRegressor(
                    estimator=LinearRegression(fit_intercept=not zero_intercept),
                    random_state=1
                ).fit(X, y)
            except TypeError:
                model = RANSACRegressor(
                    base_estimator=LinearRegression(fit_intercept=not zero_intercept),
                    random_state=1
                ).fit(X, y)
                
            if verbose:
                print("R2:", model.score(X, y))
                if not zero_intercept:
                    print("IC: ", model.estimator_.intercept_)
                print("Coef: ", model.estimator_.coef_)
                for _f, _c in zip(features, model.estimator_.coef_):
                    print(f"{_f} : {_c}")
            
            # Make predictions
            for o, row in _df[features].iterrows():
                x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
                model_predict.append(model.predict(x_pred)[0])

        elif reg.lower() == "theilsen":
            model = TheilSenRegressor(fit_intercept=not zero_intercept).fit(X, y)
            if verbose:
                print("R2:", model.score(X, y))
                if not zero_intercept:
                    print("X0:", model.intercept_)
                print("Coef: ", model.coef_)
                for _f, _c in zip(features, model.coef_):
                    print(f"{_f} : {_c}")
            
            # Make predictions
            for o, row in _df[features].iterrows():
                x_pred = array([row[feat] for feat in features]).reshape(-1, len(features))
                model_predict.append(model.predict(x_pred)[0])

        elif reg.lower() == "odr":
            # Define ODR model function for single feature
            def f(B, x):
                if zero_intercept:
                    return B[0] * x
                else:
                    return B[0] * x + B[1]
            
            # Create ODR model
            linear = odr.Model(f)
            
            # Prepare data for ODR (ensure correct shapes)
            X_odr = X.reshape(-1)  # Flatten for single feature
            
            # Estimate data uncertainties if not provided
            sx = std(X_odr) * ones_like(X_odr)
            sy = std(y) * ones_like(y)
            
            # Create ODR data object
            data = odr.RealData(X_odr, y, sx=sx, sy=sy)
            
            # Set initial parameter guess
            if zero_intercept:
                beta0 = [1.0]
            else:
                beta0 = [1.0, 0.0]
            
            # Create ODR object and fit
            odr_obj = odr.ODR(data, linear, beta0=beta0)
            model = odr_obj.run()
            
            if verbose:
                print("R2:", 1 - (model.sum_square / sum((y - mean(y))**2)))
                print("Parameters:", model.beta)
                print("Parameter errors:", model.sd_beta)
                if not zero_intercept:
                    print("Intercept:", model.beta[1])
                print(f"Slope: {model.beta[0]}")

            # Make predictions for ODR
            for o, row in _df[features].iterrows():
                x_pred = array([row[feat] for feat in features]).reshape(-1)[0]  # Get single value
                if zero_intercept:
                    pred = model.beta[0] * x_pred
                else:
                    pred = model.beta[0] * x_pred + model.beta[1]
                model_predict.append(pred)

        # Verify model was created
        if model is None:
            raise RuntimeError("Failed to create regression model")

        # Prepare output dictionary
        out = {
            'model': model,
            'r2': (1 - (model.sum_square / sum((y - mean(y))**2))) if reg.lower() == "odr" 
                  else model.score(X, y),
            'tp': _df.time,
            'dp': model_predict
        }

        # Add slope and intercept based on regression method
        if reg.lower() == "ransac":
            out['slope'] = model.estimator_.coef_[0]
            out['inter'] = 0.0 if zero_intercept else model.estimator_.intercept_
        elif reg.lower() == "theilsen":
            out['slope'] = model.coef_[0]
            out['inter'] = 0.0 if zero_intercept else model.intercept_
        elif reg.lower() == "ols":
            out['slope'] = model.coef_[0]
            out['inter'] = 0.0 if zero_intercept else model.intercept_
        elif reg.lower() == "odr":
            out['slope'] = model.beta[0]
            out['inter'] = 0.0 if zero_intercept else model.beta[1]

        return out

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
                        'time': time
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

                else:
                    baz_estimates[wave_type] = baz[0]
 
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
                        'time': time
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

                        results[wave_type]['baz'] = baz_valid
                        results[wave_type]['cc'] = cc_valid
                    else:
                        print(f"No valid tangent data")
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

    def compute_backazimuth(self, wave_type: str="", baz_step: int=1, baz_win_sec: float=30.0, 
                        rotation_data: Stream=None, translation_data: Stream=None,
                        baz_win_overlap: float=0.5, tangent_components: str="rotation", verbose: bool=False,
                        out: bool=False, cc_threshold: float=0.0) -> Dict:
        """
        Estimate backazimuth for Love, Rayleigh, or tangent waves
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love', 'rayleigh', or 'tangent')
        baz_step : int
            Step size in degrees for backazimuth search (default: 1)
        baz_win_sec : float
            Length of backazimuth estimation windows in seconds (default: 30.0)
        baz_win_overlap : float
            Overlap between windows as fraction (0-1) (default: 0.5)
        tangent_components : str
            Components to use for tangent method ('rotation' or 'acceleration')
        out : bool
            Return detailed output dictionary if True
            
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

        def padding(_baz, _ccc, _baz_steps, Ndegree=60):

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

        def get_zero_crossings(arr):

            # get nullstellen by sign function and then the difference
            nullstellen = np.diff(np.sign(arr))

            # there should only be one from negative to positive
            # this is a positive value
            nullstelle1 = np.argmax(nullstellen)

            # look for second zero crossing after the first one
            shift = nullstelle1+1
            nullstelle2 = np.argmax(abs(nullstellen[shift:]))+ shift

            return nullstelle1, nullstelle2

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

        if wave_type.lower() == "tangent":
            # revert polarity if applied
            if hasattr(self, 'pol_applied') and self.pol_applied:
                if hasattr(self, 'pol_dict') and self.pol_dict is not None:
                    for tr in ACC.select(channel="*Z"):
                        if tr.stats.channel[1:] in self.pol_dict:
                            tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]
                    for tr in ROT.select(channel="*Z"):
                        if tr.stats.channel[1:] in self.pol_dict:
                            tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]

        # sampling rate
        df = ROT[0].stats.sampling_rate

        # # Get amount of samples for data
        n_data = min([len(tr.data) for tr in ROT])

        # Prepare backazimuths for loop using integer step size
        backazimuths = linspace(0, 360 - self.baz_step, int(360 / self.baz_step))

        # Calculate window parameters
        win_samples = int(baz_win_sec * df)
        overlap_samples = int(win_samples * baz_win_overlap)
        step = win_samples - overlap_samples

        if step == 0:
            print("step is 0, setting to 1")
            step = 1

        # get amount of windows
        n_windows = int((n_data - win_samples) / step) + 1
        
        # Prepare data array
        corrbaz = ones([backazimuths.size, n_windows])*nan

        degrees = []
        windows = []
        t_center = []
        bazs = ones(n_windows)*nan

        # _______________________________
        # backazimuth estimation with Love, Rayleigh, or tangent waves
        # loop over backazimuth degrees
        for i_deg in range(0, len(backazimuths)):

            degrees.append(i_deg)

            # loop over time windows
            for i_win in range(0, n_windows):

                if i_deg == 0:
                    windows.append(i_win)

                # update indices
                idx1 = i_win * step
                idx2 = idx1 + win_samples

                # get central time of window
                if i_deg == 0:
                    t_center.append((idx1 + (idx2 - idx1)/2) /df)

                # prepare traces according to selected wave type
                if wave_type.lower() == "love":

                    if verbose and i_deg == 0 and i_win == 0:
                        print(f"> using {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    HR, HT = rotate_ne_rt(
                        ACC.select(channel='*N')[0].data,
                        ACC.select(channel='*E')[0].data,
                        backazimuths[i_deg]
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

                    if xshift != 0 and verbose:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "rayleigh":

                    if verbose and i_deg == 0 and i_win == 0:
                        print(f"> using {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    JR, JT = rotate_ne_rt(
                        ROT.select(channel='*N')[0].data,
                        ROT.select(channel='*E')[0].data,
                        backazimuths[i_deg]
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

                    if xshift != 0 and self.verbose:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "tangent":

                    if verbose and i_deg == 0 and i_win == 0:
                        print(f" > using {wave_type} for backazimuth estimation with {tangent_components} components...")

                    # no grid search, no degrees loop required
                    if i_deg > 0:
                        continue

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

                    # remove 180° ambiguity using closest to theoretical backazimuth
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
                    if cc_max > 0:
                        baz0 += 180

                    # take absolute value of correlation for better visualization
                    cc_max = abs(cc_max)

                else:
                    print(f" -> unknown wave type: {wave_type}!")
                    continue

                corrbaz[i_deg, i_win] = cc_max

                if wave_type.lower() == "tangent":
                    bazs[i_win] = baz0

        # extract maxima
        if wave_type.lower() == "tangent":
            maxbaz = bazs
            maxcorr = corrbaz[0, :]
        else:
            maxbaz = array([backazimuths[corrbaz[:, w1].argmax()] for w1 in range(0, n_windows)])
            maxcorr = array([max(corrbaz[:, w1]) for w1 in range(0, n_windows)])

        # extract mid baz based on zero crossing
        midbaz = np.zeros(n_windows)
        midcorr = np.zeros(n_windows)

        # loop over all windows
        if wave_type.lower() in ["love", "rayleigh"]:

            for _k, _win in enumerate(range(0, n_windows)):

                # pad baz array and cc array
                baz_pad, cc_pad = padding(backazimuths, corrbaz[:, _win], baz_step, Ndegree=180)

                # get zero crossings of cc function
                null1, null2 = get_zero_crossings(cc_pad)

                # get middle baz
                baz_lower = baz_pad[null1]
                baz_upper = baz_pad[null2]
                baz_mid = (baz_upper - baz_lower)/2 + baz_lower

                # avoid maximum outside range 0 - 360 due to padding
                if baz_mid < 0:
                    baz_mid += 360

                if baz_mid > 360:
                    baz_mid -= 360

                # write to array
                midbaz[_k] = baz_mid

                # get cc value at baz_mid and write to array
                idx_mid = int((null2 - null1)/2 + null1)
                midcorr[_k] = cc_pad[idx_mid]

        # create mesh grid
        t_win = arange(0, baz_win_sec*n_windows+baz_win_sec, baz_win_sec)
        t_win = t_win[:-1]+baz_win_sec/2
        grid = meshgrid(t_win, backazimuths)

        # add one element for axes
        windows.append(windows[-1]+1)
        degrees.append(degrees[-1]+self.baz_step)

        # prepare results
        output = {}
        output['baz_mesh'] = grid
        output['baz_corr'] = corrbaz
        output['baz_time'] = t_win
        output['acc'] = ACC
        output['rot'] = ROT
        output['twin_center'] = np.array(t_center)
        output['cc_max_y'] = maxbaz
        output['baz_max'] = maxbaz
        output['cc_max'] = maxcorr
        output['baz_mid'] = midbaz
        output['cc_mid'] = midcorr
        output['component_type'] = tangent_components if wave_type.lower() == "tangent" else None
        output['parameters'] = {
            'baz_win_sec': baz_win_sec,
            'baz_win_overlap': baz_win_overlap,
            'baz_step': baz_step,
            'wave_type': wave_type,
        }

        # add results to object
        self.baz_results[wave_type] = output

        # return output if out required
        if out:
            return output

    # OLD
    def compute_backazimuth_fast(self, wave_type: str="love", baz_step: int=1, baz_win_sec: float=30.0, 
                        baz_win_overlap: float=0.5, tangent_components: str="rotation", verbose: bool=True,
                        out: bool=False, n_jobs: int=-1) -> Dict:
        """
        Estimate backazimuth for Love, Rayleigh, or tangent waves (optimized with parallelization)
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love', 'rayleigh', or 'tangent')
        baz_step : int
            Step size in degrees for backazimuth search (default: 1)
        baz_win_sec : float
            Length of backazimuth estimation windows in seconds (default: 30.0)
        baz_win_overlap : float
            Overlap between windows as fraction (0-1) (default: 0.5)
        tangent_components : str
            Components to use for tangent method ('rotation' or 'acceleration')
        verbose : bool
            Print progress information
        out : bool
            Return detailed output dictionary if True
        n_jobs : int
            Number of parallel jobs (-1 for all cores, 1 for sequential)
        
        Returns:
        --------
        Dict : Backazimuth estimation results
        """
        from obspy.signal.rotate import rotate_ne_rt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import linspace, ones, array, nan, meshgrid, arange, zeros, cov, pi, arctan
        from numpy.linalg import eigh
        from numpy import argsort
        from joblib import Parallel, delayed
        import numpy as np

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
        ACC = self.get_stream("translation").copy()
        ROT = self.get_stream("rotation").copy()

        if wave_type.lower() == "tangent":
            # revert polarity if applied
            if hasattr(self, 'pol_applied') and self.pol_applied:
                if hasattr(self, 'pol_dict') and self.pol_dict is not None:
                    for tr in ACC.select(channel="*Z"):
                        if tr.stats.channel[1:] in self.pol_dict:
                            tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]
                    for tr in ROT.select(channel="*Z"):
                        if tr.stats.channel[1:] in self.pol_dict:
                            tr.data = tr.data * self.pol_dict[tr.stats.channel[1:]]

        # Get amount of samples for data
        n_samples = len(ROT.select(channel="*Z")[0])

        # Calculate overlap in samples
        overlap = int(baz_win_overlap * baz_win_sec * self.sampling_rate)

        # Prepare time windows for loop
        n_windows = n_samples // (int(self.sampling_rate * baz_win_sec))

        # Prepare backazimuths for loop using integer step size
        backazimuths = linspace(0, 360 - self.baz_step, int(360 / self.baz_step))

        # Prepare data array
        corrbaz = ones([backazimuths.size, n_windows])*nan

        degrees = []
        windows = []

        bazs = ones(n_windows)*nan

        # Extract the core computation logic for parallelization
        def _compute_single_baz_window(i_deg, i_win, backazimuth_val):
            """Core computation for a single (degree, window) combination"""
            try:
                # infer indices
                idx1 = int(self.sampling_rate * baz_win_sec * i_win)
                idx2 = int(self.sampling_rate * baz_win_sec * (i_win + 1))

                # add overlap
                if i_win > 0 and i_win < n_windows:
                    idx1 = int(idx1 - overlap * baz_win_sec * self.sampling_rate)
                    idx2 = int(idx2 + overlap * baz_win_sec * self.sampling_rate)

                # prepare traces according to selected wave type
                if wave_type.lower() == "love":
                    if verbose and i_deg == 0 and i_win == 0:
                        print(f"> using {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    R, T = rotate_ne_rt(ACC.select(channel='*N')[0].data,
                                        ACC.select(channel='*E')[0].data,
                                        backazimuth_val)

                    # compute correlation for backazimuth
                    ccorr = correlate(ROT.select(channel="*Z")[0][idx1:idx2],
                                    T[idx1:idx2],
                                    0, demean=True, normalize='naive', method='fft')

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)

                    if xshift != 0 and verbose:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "rayleigh":
                    if verbose and i_deg == 0 and i_win == 0:
                        print(f"> using {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    R, T = rotate_ne_rt(ROT.select(channel='*N')[0].data,
                                        ROT.select(channel='*E')[0].data,
                                        backazimuth_val)

                    # compute correlation for backazimuth
                    ccorr = correlate(ACC.select(channel="*Z")[0][idx1:idx2],
                                    T[idx1:idx2],
                                    0, demean=True, normalize='naive', method='fft')

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)

                    if xshift != 0 and verbose:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "tangent":
                    if verbose and i_deg == 0 and i_win == 0:
                        print(f" > using {wave_type} for backazimuth estimation with {tangent_components} components...")

                    # no grid search, no degrees loop required
                    if i_deg > 0:
                        return i_deg, i_win, nan, nan

                    try:
                        N = len(ROT[0].data[idx1:idx2])
                    except:
                        N = len(ACC[0].data[idx1:idx2])

                    # prepare data based on component choice
                    dat = zeros((N, 2))

                    if tangent_components.lower() == "rotation":
                        dat[:, 0] = ROT.select(channel='*E')[0].data[idx1:idx2]
                        dat[:, 1] = ROT.select(channel='*N')[0].data[idx1:idx2]
                    elif tangent_components.lower() == "acceleration":
                        dat[:, 0] = ACC.select(channel='*E')[0].data[idx1:idx2]
                        dat[:, 1] = ACC.select(channel='*N')[0].data[idx1:idx2]
                    else:
                        raise ValueError(f"Invalid tangent_components: {tangent_components}")

                    # compute covariance
                    covar = cov(dat, rowvar=False)

                    # get dominant eigenvector
                    Cprime, Q = eigh(covar, UPLO='U')

                    # sorting and formatting
                    loc = argsort(abs(Cprime))[::-1]
                    Q = Q[:, loc]

                    # get backazimuth using tangent of eigenvectors
                    baz0 = -arctan((Q[1, 0]/Q[0, 0]))*180/pi

                    # if negative due to tangent, then add 180 degrees
                    if baz0 <= 0:
                        baz0 += 180

                    # remove 180° ambiguity
                    if tangent_components.lower() == "rotation":
                        R, T = rotate_ne_rt(ROT.select(channel='*N')[0].data,
                                            ROT.select(channel='*E')[0].data,
                                            baz0)
                        
                        ccorr = correlate(ACC.select(channel="*Z")[0][idx1:idx2],
                                        T[idx1:idx2],
                                        0, demean=True, normalize='naive', method='fft')
                    else:
                        if hasattr(self, 'event_info') and self.event_info and 'backazimuth' in self.event_info:
                            rmse = np.sqrt(np.mean((baz0 - self.event_info['backazimuth'])**2))
                            rmse_180 = np.sqrt(np.mean((baz0 + 180 - self.event_info['backazimuth'])**2))
                            ccorr = [1e-6] if rmse_180 < rmse else [-1e-6]
                        else:
                            ccorr = [-1e-6]  # Default if no event info

                    xshift, cc_max = xcorr_max(ccorr)
                    if cc_max > 0:
                        baz0 += 180
                    cc_max = abs(cc_max)

                    return i_deg, i_win, cc_max, baz0

                else:
                    print(f" -> unknown wave type: {wave_type}!")
                    return i_deg, i_win, nan, nan

                return i_deg, i_win, cc_max, nan

            except Exception as e:
                if verbose:
                    print(f"Error in degree {i_deg}, window {i_win}: {e}")
                return i_deg, i_win, nan, nan

        # Create parameter combinations for parallelization
        param_combinations = []
        for i_deg in range(len(backazimuths)):
            for i_win in range(n_windows):
                if wave_type.lower() == "tangent" and i_deg > 0:
                    continue
                param_combinations.append((i_deg, i_win, backazimuths[i_deg]))

        # Parallel computation
        if n_jobs == 1 or len(param_combinations) < 10:
            computation_results = []
            for i_deg, i_win, baz_val in param_combinations:
                computation_results.append(_compute_single_baz_window(i_deg, i_win, baz_val))
        else:
            computation_results = Parallel(n_jobs=n_jobs, backend='threading', verbose=0)(
                delayed(_compute_single_baz_window)(i_deg, i_win, baz_val) 
                for i_deg, i_win, baz_val in param_combinations
            )

        # Process results
        for i_deg in range(len(backazimuths)):
            degrees.append(i_deg)
        for i_win in range(n_windows):
            windows.append(i_win)

        for i_deg, i_win, cc_max, baz0 in computation_results:
            if not np.isnan(cc_max):
                corrbaz[i_deg, i_win] = cc_max
            if wave_type.lower() == "tangent" and not np.isnan(baz0):
                bazs[i_win] = baz0

        # Extract maxima and create output
        if wave_type.lower() == "tangent":
            maxbaz = bazs
            maxcorr = corrbaz[0, :]
        else:
            maxbaz = array([backazimuths[corrbaz[:, l1].argmax()] for l1 in range(n_windows)])
            maxcorr = array([max(corrbaz[:, l1]) for l1 in range(n_windows)])

        # Create mesh grid and time windows
        t_win = arange(0, baz_win_sec*n_windows+baz_win_sec, baz_win_sec)
        t_win_center = t_win[:-1]+baz_win_sec/2
        grid = meshgrid(t_win, backazimuths)

        # Add one element for axes
        windows.append(windows[-1]+1)
        degrees.append(degrees[-1]+self.baz_step)

        # Store results based on wave type
        result_type = wave_type.lower()
        if result_type in ["love", "rayleigh"]:
            setattr(self, f'baz_grid_{result_type}', corrbaz)
            setattr(self, f'baz_degrees_{result_type}', degrees)
            setattr(self, f'baz_windows_{result_type}', windows)
            setattr(self, f'baz_corr_{result_type}', maxcorr)
            setattr(self, f'baz_max_{result_type}', maxbaz)
            setattr(self, f'baz_times_{result_type}', t_win_center)
        elif result_type == "tangent":
            comp_suffix = f"_{tangent_components.lower()}"
            for attr in ['grid', 'degrees', 'windows', 'corr', 'max', 'times']:
                setattr(self, f'baz_{attr}_tangent{comp_suffix}', locals()[f'{"corrbaz" if attr=="grid" else "max"+attr if attr in ["corr","baz"] else attr}'])
                setattr(self, f'baz_{attr}_tangent', locals()[f'{"corrbaz" if attr=="grid" else "max"+attr if attr in ["corr","baz"] else attr}'])

        if out:
            return {
                'baz_mesh': grid,
                'baz_corr': corrbaz,
                'acc': ACC,
                'rot': ROT,
                'twin_center': t_win_center,
                'cc_max_y': maxbaz,
                'cc_max': maxcorr,
                'component_type': tangent_components if wave_type.lower() == "tangent" else None,
                'parameters': {
                    'baz_win_sec': baz_win_sec,
                    'baz_win_soverlap': baz_win_overlap,
                    'baz_step': baz_step,
                    'wave_type': wave_type,
                }
            }

    def compute_odr(self, x_array: ndarray, y_array: ndarray, xerr: Union[float, ndarray]=None, 
                   yerr: Union[float, ndarray]=None, zero_intercept: bool=False) -> Dict:
        """
        Compute Orthogonal Distance Regression between two arrays
        
        Parameters:
        -----------
        x_array : numpy.ndarray
            Independent variable data
        y_array : numpy.ndarray
            Dependent variable data
        xerr : float or numpy.ndarray, optional
            Error in x. If None, uses standard deviation of x
        yerr : float or numpy.ndarray, optional
            Error in y. If None, uses standard deviation of y
        zero_intercept : bool, optional
            Force intercept to be zero if True
            
        Returns:
        --------
        Dict
            Dictionary containing regression results
        """
        from scipy import odr
        import numpy as np
        
        # Set default errors if not provided
        if xerr is None:
            xerr = np.std(x_array)
        if yerr is None:
            yerr = np.std(y_array)
            
        # Convert to arrays if scalar errors provided
        if np.isscalar(xerr):
            xerr = np.ones_like(x_array) * xerr
        if np.isscalar(yerr):
            yerr = np.ones_like(y_array) * yerr
            
        # Define model function
        def linear_func(params, x):
            if zero_intercept:
                return params[0] * x
            else:
                return params[0] * x + params[1]
            
        # Create model
        linear = odr.Model(linear_func)
        
        # Create data object
        data = odr.RealData(x_array, y_array, sx=xerr, sy=yerr)
        
        # Set initial parameters
        if zero_intercept:
            beta0 = [np.mean(y_array)/np.mean(x_array)]
        else:
            beta0 = [np.mean(y_array)/np.mean(x_array), 0]
            
        # Create ODR object and run regression
        odr_obj = odr.ODR(data, linear, beta0=beta0)
        output = odr_obj.run()
        
        # Calculate R-squared
        if zero_intercept:
            y_fit = output.beta[0] * x_array
            intercept = 0
            intercept_err = 0
        else:
            y_fit = output.beta[0] * x_array + output.beta[1]
            intercept = output.beta[1]
            intercept_err = output.sd_beta[1]
            
        ss_res = np.sum((y_array - y_fit) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Prepare results
        results = {
            'slope': output.beta[0],
            'intercept': intercept,
            'slope_err': output.sd_beta[0],
            'intercept_err': intercept_err,
            'r_squared': r_squared,
            'fit_params': {
                'x': x_array,
                'y': y_array,
                'y_fit': y_fit,
                'xerr': xerr,
                'yerr': yerr
            }
        }
        
        return results

    def compute_regression(self, x_array: ndarray, y_array: ndarray, 
                          method: str='ransac', zero_intercept: bool=False,
                          trials: int=1000, min_samples: int=2,
                          residual_threshold: float=None) -> Dict:
        """
        Compute linear regression using RANSAC or Theil-Sen methods
        
        Parameters:
        -----------
        x_array : numpy.ndarray
            Independent variable data
        y_array : numpy.ndarray
            Dependent variable data
        method : str
            Regression method ('ransac' or 'theilsen')
        zero_intercept : bool
            Force intercept to be zero if True
        trials : int
            Number of trials for RANSAC (only used if method='ransac')
        min_samples : int
            Minimum samples for RANSAC (only used if method='ransac')
        residual_threshold : float or None
            Maximum residual for RANSAC inliers. If None, uses median absolute deviation
            
        Returns:
        --------
        Dict
            Dictionary containing:
            - slope: regression slope
            - intercept: regression intercept (0 if zero_intercept=True)
            - r_squared: R-squared value of fit
            - inlier_mask: boolean mask of inliers (only for RANSAC)
            - fit_params: dictionary with fit parameters
        """
        from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
        import numpy as np
        
        # Reshape arrays for sklearn
        X = x_array.reshape(-1, 1)
        y = y_array.reshape(-1, 1)
        
        if zero_intercept:
            # For zero intercept, modify data to force through origin
            X = np.hstack([X, np.zeros_like(X)])
        
        # Select regression method
        if method.lower() == 'ransac':
            regressor = RANSACRegressor(
                random_state=42,
                max_trials=trials,
                min_samples=min_samples,
                residual_threshold=residual_threshold,
                loss='absolute_error'
            )
        elif method.lower() == 'theilsen':
            regressor = TheilSenRegressor(
                random_state=42,
                max_iter=trials,
                n_subsamples=min_samples
            )
        else:
            raise ValueError(f"Invalid method: {method}. Use 'ransac' or 'theilsen'")
        
        # Fit regression
        if zero_intercept:
            regressor.fit(X[:, 0].reshape(-1, 1), y)
            slope = regressor.estimator_.coef_[0]
            intercept = regressor.estimator_.intercept_
        else:
            regressor.fit(X, y)
            slope = regressor.estimator_.coef_[0]
            intercept = regressor.estimator_.intercept_
        
        # Calculate fitted values
        y_fit = slope * x_array + intercept
        
        # Calculate R-squared
        ss_res = np.sum((y_array - y_fit) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Prepare results dictionary
        results = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'fit_params': {
                'x': x_array,
                'y': y_array,
                'y_fit': y_fit,
                'method': method
            }
        }
        
        # Add RANSAC-specific results
        if method.lower() == 'ransac':
            results['inlier_mask'] = regressor.inlier_mask_
            results['fit_params']['n_inliers'] = np.sum(regressor.inlier_mask_)
            results['fit_params']['n_trials'] = regressor.n_trials_
        
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
            elif hasattr(self, 'baz_estimated'):
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
                    if method.lower() == 'odr':
                        velocities[i] = self.compute_odr(rot_z[i1:i2], 0.5*acc_t[i1:i2])['slope']
                    elif method.lower() == 'ransac':
                        velocities[i] = self.compute_regression(rot_z[i1:i2], 0.5*acc_t[i1:i2], method='ransac', zero_intercept=True)['slope']

                elif wave_type.lower() == 'rayleigh':
                    # get velocity from amplitude ratio via regression
                    if method.lower() == 'odr':
                        velocities[i] = self.compute_odr(rot_t[i1:i2], acc_z[i1:i2])['slope']
                    elif method.lower() == 'ransac':
                        velocities[i] = self.compute_regression(rot_t[i1:i2], acc_z[i1:i2], method='ransac', zero_intercept=True)['slope']
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

    # OLD
    def compute_velocities_in_windows(self, wave_type: str="love", overlap: float=0.5, win_time_s: float=None,
                                      cc_threshold: float=0.2, adjusted_baz: bool=False, baz: float=None,
                                      method: str='odr', fmin: float=None, fmax: float=None) -> Dict:
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
        if fmin is not None and fmax is not None:
            rot = self.get_stream("rotation", raw=True).copy()
            acc = self.get_stream("translation", raw=True).copy()
            rot = rot.detrend('linear').filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
            acc = acc.detrend('linear').filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
        else:
            rot = self.get_stream("rotation").copy()
            acc = self.get_stream("translation").copy()
            
        if not adjusted_baz and baz is None:
            raise ValueError("Backazimuth must be provided if adjusted_baz is False")
        
        # Get sampling rate
        df = self.sampling_rate
        
        # Set window length if not provided
        if win_time_s is None:
            win_time_s = 1/self.fmin

        # select mode of operation
        if not adjusted_baz:
            # Rotate components to radial-transverse
            rot_r, rot_t = rotate_ne_rt(rot.select(channel='*N')[0].data,
                                    rot.select(channel='*E')[0].data,
                                    baz)
            acc_r, acc_t = rotate_ne_rt(acc.select(channel='*N')[0].data,
                                    acc.select(channel='*E')[0].data,
                                    baz)
        else:
            # compute backazimuth for windows
            baz_results = self.compute_backazimuth(
                wave_type=wave_type,
                baz_step=1,
                baz_win_sec=win_time_s,
                baz_win_overlap=overlap,
                out=True
            )

        # Get vertical components
        rot_z = rot.select(channel="*Z")[0].data
        acc_z = acc.select(channel="*Z")[0].data
        
        # Calculate window parameters
        win_samples = int(win_time_s * df)
        overlap_samples = int(win_samples * overlap)
        step = win_samples - overlap_samples

        # number of windows
        n_windows = len(rot_z) // win_samples

        # Initialize arrays
        times = np.zeros(n_windows)
        velocities = np.zeros(n_windows)
        cc_coeffs = np.zeros(n_windows)
        
        # Loop through windows
        for i in range(n_windows):
            i1 = i * step
            i2 = i1 + win_samples
            
            if adjusted_baz:
                # Rotate components to radial-transverse
                rot_r, rot_t = rotate_ne_rt(rot.select(channel='*N')[0].data,
                                        rot.select(channel='*E')[0].data,
                                        baz_results['cc_max_y'][i])
                acc_r, acc_t = rotate_ne_rt(acc.select(channel='*N')[0].data,
                                        acc.select(channel='*E')[0].data,
                                        baz_results['cc_max_y'][i])
            
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
                    if method.lower() == 'odr':
                        velocities[i] = self.compute_odr(rot_z[i1:i2], 0.5*acc_t[i1:i2])['slope']
                    elif method.lower() == 'ransac':
                        velocities[i] = self.compute_regression(rot_z[i1:i2], 0.5*acc_t[i1:i2], method='ransac', zero_intercept=True)['slope']

                elif wave_type.lower() == 'rayleigh':
                    # get velocity from amplitude ratio via regression
                    if method.lower() == 'odr':
                        velocities[i] = self.compute_odr(rot_t[i1:i2], acc_z[i1:i2])['slope']
                    elif method.lower() == 'ransac':
                        velocities[i] = self.compute_regression(rot_t[i1:i2], acc_z[i1:i2], method='ransac', zero_intercept=True)['slope']
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
        
        # center time of window
        times_center = np.ones_like(times) * win_time_s/2

        # Create output dictionary
        results = {
            'time': times,
            'velocity': velocities,
            'ccoef': cc_coeffs,
            'terr': times_center,
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
        
        # add backazimuth if adjusted mode
        if adjusted_baz:
            results['backazimuth'] = baz_results['cc_max_y']
        else:
            results['backazimuth'] = baz*np.ones_like(times)

        return results

    def compute_velocities_optimized(self, rotation_data: Stream=None, translation_data: Stream=None,
                                     wave_type: str='love', baz_results: Dict=None, baz_mode: str='mid',
                                     method: str='odr', cc_threshold: float=0.0) -> Dict:
        """
        Compute phase velocities in time intervals for Love or Rayleigh waves
        
        Parameters:
        -----------
        rotation_data : Stream
            Rotational data stream
        translation_data : Stream
            Translation data stream
        wave_type : str
            Type of wave to analyze ('love' or 'rayleigh')
        baz_results : Dict
            Dictionary containing backazimuth results
        baz_mode : str
            Mode to use for backazimuth selection ('max' or 'mid')
        method : str
            Method to use for velocity computation ('odr' or 'ransac')
        cc_threshold : float, optional
            Minimum cross-correlation coefficient to consider, by default 0.0
        Returns:
        --------
        Dict
            Dictionary containing:
            - times : array of time points
            - velocity: array of phase velocities
            - cc_method: array of cross-correlation coefficients
            - backazimuth: array of backazimuths
        """
        import numpy as np
        from tqdm import tqdm
        from obspy.signal.rotate import rotate_ne_rt

        # Validate inputs
        if rotation_data is None or translation_data is None:
            raise ValueError("Both rotation and translation data must be provided")
        if baz_results is None:
            raise ValueError("Backazimuth results must be provided")
        if wave_type.lower() not in ['love', 'rayleigh']:
            raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
        if baz_mode.lower() not in ['max', 'mid']:
            raise ValueError(f"Invalid baz mode: {baz_mode}. Use 'max' or 'mid'")
        if method.lower() not in ['odr', 'ransac']:
            raise ValueError(f"Invalid method: {method}. Use 'odr' or 'ransac'")

        # Make copies to avoid modifying original data
        rot = rotation_data.copy()
        tra = translation_data.copy()

        # Get sampling rate and validate
        df = rot[0].stats.sampling_rate
        if df <= 0:
            raise ValueError(f"Invalid sampling rate: {df}")

        # Extract parameters from baz_results
        try:
            win_time_s = baz_results['parameters']['baz_win_sec']
            overlap = baz_results['parameters']['baz_win_overlap']
            ttt = baz_results['twin_center']

            if baz_mode.lower() == 'max':
                baz = baz_results['baz_max']
                ccc = baz_results['cc_max']
            else:  # 'mid'
                baz = baz_results['baz_mid']
                ccc = baz_results['cc_mid']
        except KeyError as e:
            raise ValueError(f"Missing required key in baz_results: {e}")

        # Validate array lengths
        if not (len(baz) == len(ttt) == len(ccc)):
            raise ValueError("Inconsistent lengths in backazimuth results")

        # Get components
        try:
            rot_z = rot.select(channel="*Z")[0].data
            acc_z = tra.select(channel="*Z")[0].data
        except Exception as e:
            raise RuntimeError(f"Error accessing vertical components: {str(e)}")

        # Calculate window parameters
        win_samples = int(win_time_s * df)
        if win_samples <= 0:
            raise ValueError(f"Invalid window size: {win_samples} samples")

        overlap_samples = int(win_samples * overlap)
        step = win_samples - overlap_samples

        # number of windows
        n_windows = len(ttt)

        # Initialize arrays
        times = np.zeros(n_windows)
        velocities = np.zeros(n_windows)
        cc_coeffs = np.zeros(n_windows)
        
        # Loop through windows
        velocities = np.ones_like(baz) * np.nan

        for i, (_baz, _ttt, _ccc) in tqdm(enumerate(zip(baz, ttt, ccc))):
            i1 = i * step
            i2 = i1 + win_samples
            
            # apply cc threshold
            if _ccc <= cc_threshold:
                velocities[i] = np.nan
                continue

            # Rotate components to radial-transverse
            if wave_type.lower() == 'rayleigh':
                rot_r, rot_t = rotate_ne_rt(
                    rot.select(channel='*N')[0].data,
                    rot.select(channel='*E')[0].data,
                    _baz
                )
            elif wave_type.lower() == 'love':
                acc_r, acc_t = rotate_ne_rt(
                    tra.select(channel='*N')[0].data,
                    tra.select(channel='*E')[0].data,
                    _baz
                )

            # Compute velocity using amplitude ratio
            if wave_type.lower() == 'love':
                # get velocity from amplitude ratio via regression
                if method.lower() == 'odr':
                    velocities[i] = self.compute_odr(rot_z[i1:i2], 0.5*acc_t[i1:i2])['slope']
                elif method.lower() == 'ransac':
                    velocities[i] = self.compute_regression(rot_z[i1:i2], 0.5*acc_t[i1:i2], 
                                                            method='ransac', zero_intercept=True)['slope']

            elif wave_type.lower() == 'rayleigh':
                # get velocity from amplitude ratio via regression
                if method.lower() == 'odr':
                    velocities[i] = self.compute_odr(rot_t[i1:i2], acc_z[i1:i2])['slope']
                elif method.lower() == 'ransac':
                    velocities[i] = self.compute_regression(rot_t[i1:i2], acc_z[i1:i2], 
                                                            method='ransac', zero_intercept=True)['slope']
            
            else:
                raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")

        return {
            'times': ttt,
            'velocity': velocities,
            'cc_value': ccc,
            'backazimuth': baz,
            'parameters': {
                'wave_type': wave_type,
                'win_time_s': win_time_s,
                'overlap': overlap,
                'method': method,
                'baz_mode': baz_mode
            }
        }

    def _process_frequency_band(self, freq_params, wave_type, t_win_factor, overlap, baz_mode, method, cc_threshold, rot_data, tra_data):
        """
        Process a single frequency band for parallel computation.
        
        Parameters
        ----------
        freq_params : tuple
            (lower_freq, upper_freq, center_freq)
        wave_type : str
            Type of wave to analyze
        t_win_factor : float
            Window length factor
        overlap : float
            Window overlap fraction
        baz_mode : str
            Backazimuth computation mode
        method : str
            Velocity computation method
        cc_threshold : float
            Cross-correlation threshold
        rot_data : Stream
            Rotation data stream
        tra_data : Stream
            Translation data stream
            
        Returns
        -------
        dict or None
            Results for this frequency band
        """

        def filter_data(st, fmin: Optional[float]=0.1, fmax: Optional[float]=0.5):
            """Filter stream data with error handling"""
            try:
                st = st.copy()  # Don't modify original
                st = st.detrend("linear")
                st = st.detrend("demean")
                st = st.taper(0.05)

                if fmin is not None and fmax is not None:
                    st = st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True) 
                elif fmin is not None:
                    st = st.filter("lowpass", freq=fmin, corners=4, zerophase=True)
                elif fmax is not None:
                    st = st.filter("highpass", freq=fmax, corners=4, zerophase=True)

                return st
            except Exception as e:
                raise RuntimeError(f"Error filtering data: {str(e)}")

        # get frequency band parameters
        fl, fu, fc = freq_params
        
        # Calculate window length based on center frequency
        win_time_s = int(t_win_factor / fc)
        
        # define output
        results = {
            'frequency': fc,
            'backazimuth': np.nan,
            'cc_values': np.nan,
            'velocity': np.nan,
            'baz_mad': np.nan,
            'ccc_mad': np.nan,
            'vel_mad': np.nan,
            'win_time_s': win_time_s
        }

        # get raw data
        rot = self.get_stream("rotation", raw=True)
        tra = self.get_stream("translation", raw=True)

        # Filter data
        rot = filter_data(rot, fmin=fl, fmax=fu)
        tra = filter_data(tra, fmin=fl, fmax=fu)

        # Compute backazimuths
        baz_results = self.compute_backazimuth(
            rotation_data=rot,
            translation_data=tra,
            wave_type=wave_type,
            baz_step=1,
            baz_win_sec=win_time_s,
            baz_win_overlap=overlap,
            out=True
        )

        if baz_results is None:
            print("No baz results")
            return results

        # Compute velocities
        vel_results = self.compute_velocities_optimized(
            rotation_data=rot,
            translation_data=tra,
            wave_type=wave_type,
            baz_results=baz_results,
            baz_mode=baz_mode,
            method=method,
            cc_threshold=cc_threshold
        )
        
        if vel_results is None:
            print("No vel results")
            return results
        
        # Apply threshold and clean data
        mask = vel_results['cc_value'] > cc_threshold
        baz_cc_filtered = vel_results['backazimuth'][mask]
        ccc_cc_filtered = vel_results['cc_value'][mask]
        vel_cc_filtered = vel_results['velocity'][mask]
        
        # Remove NaN values
        vel_valid = vel_cc_filtered[~np.isnan(vel_cc_filtered)]
        baz_valid = baz_cc_filtered[~np.isnan(baz_cc_filtered)]
        ccc_valid = ccc_cc_filtered[~np.isnan(ccc_cc_filtered)]
        
        # compute median
        baz_median = np.nanmedian(baz_valid)
        ccc_median = np.nanmedian(ccc_valid)
        vel_median = np.nanmedian(vel_valid)

        # compute mad
        baz_mad = np.nanmedian(np.abs(baz_valid - baz_median))
        ccc_mad = np.nanmedian(np.abs(ccc_valid - ccc_median))
        vel_mad = np.nanmedian(np.abs(vel_valid - vel_median))

        # update results
        results = {
            'frequency': fc,
            'backazimuth': baz_median,
            'cc_values': ccc_median,
            'velocity': vel_median,
            'baz_mad': baz_mad,
            'ccc_mad': ccc_mad,
            'vel_mad': vel_mad,
            'win_time_s': win_time_s
        }

        return results

    def compute_frequency_dependent_parameters_parallel(self, wave_type: str='love', fbands: Dict=None, 
                                                t_win_factor: float=2, overlap: float=0.5, baz_mode: str='mid', 
                                                method: str='odr', cc_threshold: float=0.0, n_jobs: int=-1) -> Dict:
        """
        Parallel version of compute_frequency_dependent_parameters.
        Uses multiprocessing to speed up computation across frequency bands.
        
        Parameters
        ----------
        wave_type : str, optional
            Type of wave to analyze ('love' or 'rayleigh'), by default 'love'
        fbands : Dict, optional
            Dictionary containing frequency band parameters:
            - 'fmin': minimum frequency
            - 'fmax': maximum frequency
            - 'octave_fraction': fraction of octave for band division
            If None, default values will be used
        t_win_factor : float, optional
            Window length factor for backazimuth computation, by default 2.0
            The actual window length will be t_win_factor / center_frequency
        overlap : float, optional
            Overlap fraction between windows for backazimuth computation, by default 0.5
        cc_threshold : float, optional
            Minimum cross-correlation coefficient to consider, by default 0.0
        baz_mode : str, optional
            Mode of backazimuth computation ('max' or 'mid'), by default 'mid'
        method : str, optional
            Method to use for velocity computation ('odr' or 'ransac'), by default 'odr'
        n_jobs : int, optional
            Number of parallel jobs. -1 means using all processors, by default -1
        """
        import numpy as np
        from multiprocessing import Pool, cpu_count
        from functools import partial
        from tqdm import tqdm

        # Validate input parameters
        if wave_type.lower() not in ['love', 'rayleigh']:
            raise ValueError(f"Invalid wave type: {wave_type}. Use 'love' or 'rayleigh'")
        if baz_mode.lower() not in ['max', 'mid']:
            raise ValueError(f"Invalid baz mode: {baz_mode}. Use 'max' or 'mid'")
        if method.lower() not in ['odr', 'ransac']:
            raise ValueError(f"Invalid method: {method}. Use 'odr' or 'ransac'")
        if not 0 <= overlap < 1:
            raise ValueError(f"Overlap must be between 0 and 1, got {overlap}")
        if t_win_factor <= 0:
            raise ValueError(f"Window length factor must be positive, got {t_win_factor}")

        # Default frequency bands if not provided
        if fbands is None:
            fbands = {
                'fmin': 0.01,
                'fmax': 1.0,
                'octave_fraction': 3
            }

        # Get frequency bands
        flower, fupper, fcenter = self.get_octave_bands(
            fmin=fbands['fmin'],
            fmax=fbands['fmax'],
            faction_of_octave=fbands['octave_fraction']
        )

        # Get raw data once
        rot_data = self.get_stream("rotation", raw=True)
        tra_data = self.get_stream("translation", raw=True)

        # Prepare frequency band parameters
        freq_params = list(zip(flower, fupper, fcenter))
        
        # Set up parallel processing
        if n_jobs == -1:
            n_jobs = cpu_count()
        n_jobs = min(n_jobs, len(freq_params))  # Don't use more processes than frequency bands
        
        print(f"Processing {len(freq_params)} frequency bands using {n_jobs} processes...")
        
        # Create partial function with fixed parameters
        process_func = partial(
            self._process_frequency_band,
            wave_type=wave_type,
            t_win_factor=t_win_factor,
            overlap=overlap,
            baz_mode=baz_mode,
            method=method,
            cc_threshold=cc_threshold,
            rot_data=rot_data,
            tra_data=tra_data
        )

        # Initialize results
        results = {
            'frequency': [],
            'backazimuth': [],
            'cc_values': [],
            'velocity': [],
            'baz_mad': [],
            'ccc_mad': [],
            'vel_mad': [],
            'parameters': {
                'wave_type': wave_type,
                't_win_factor': t_win_factor,
                'overlap': overlap,
                'baz_mode': baz_mode,
                'method': method,
                'cc_threshold': cc_threshold,
                'frequency_bands': {
                    'lower': flower.tolist(),
                    'upper': fupper.tolist(),
                    'center': fcenter.tolist()
                }
            }
        }

        # Process frequency bands in parallel with progress bar
        with Pool(n_jobs) as pool:
            # Use imap_unordered for better progress tracking
            for freq_result in tqdm(pool.imap_unordered(process_func, freq_params), 
                                  total=len(freq_params),
                                  desc="Processing frequency bands",
                                  unit="band"):
                if freq_result is not None:
                    results['frequency'].append(freq_result['frequency'])
                    results['backazimuth'].append(freq_result['backazimuth'])
                    results['cc_values'].append(freq_result['cc_values'])
                    results['velocity'].append(freq_result['velocity'])
                    results['baz_mad'].append(freq_result['baz_mad'])
                    results['ccc_mad'].append(freq_result['ccc_mad'])
                    results['vel_mad'].append(freq_result['vel_mad'])

                    # Print detailed progress for each completed band
                    print(f"✓ Processed {freq_result['frequency']:.3f} Hz (T={freq_result['win_time_s']:.1f}s)")

        if not results['frequency']:
            raise RuntimeError("No valid results for any frequency band")

        print(f"Completed processing {len(results['frequency'])} frequency bands successfully.")
        return results

    def compare_backazimuth_methods(self, Twin: float, Toverlap: float, baz_theo: float=None, 
                                  baz_theo_margin: float=10, baz_step: int=1, minors: bool=True,
                                  cc_threshold: float=0, cc_method: str='max', plot: bool=False, output: bool=False,
                                  precomputed: bool=True) -> Tuple[plt.Figure, Dict]:
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
        
        # Get and process streams
        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()
        
        # Initialize results dictionary
        results_dict = {}
        baz_estimated = {}
        
        # Create figure if plotting
        if plot:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 8, figure=fig)
            
            # Create subplots
            ax1 = fig.add_subplot(gs[0, :8])  # Love wave BAZ
            ax11 = fig.add_subplot(gs[0, 7:])  # Love wave histogram
            ax11.set_axis_off()
            
            ax2 = fig.add_subplot(gs[1, :8])  # Rayleigh wave BAZ
            ax22 = fig.add_subplot(gs[1, 7:])  # Rayleigh wave histogram
            ax22.set_axis_off()
            
            ax3 = fig.add_subplot(gs[2, :8])  # Tangent BAZ
            ax33 = fig.add_subplot(gs[2, 7:])  # Tangent histogram
            ax33.set_axis_off()
            
            # Create color map
            cmap = plt.get_cmap("viridis", 10)
        
        # Plot settings
        font = 12
        deltaa = 10
        angles1 = arange(0, 365, deltaa)
        angles2 = arange(0, 365, 1)
        t1, t2 = 0, rot[0].stats.endtime - rot[0].stats.starttime
        
        # Process each wave type
        for wave_type, label in [('love', 'Love'), ('rayleigh', 'Rayleigh'), ('tangent', 'Tangent')]:
            
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
                # Plot results for each wave type
                if wave_type == 'love':
                    scatter = ax1.scatter(
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
                    scatter = ax2.scatter(
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
                    ax3.scatter(
                        wave_results['twin_center'][wave_results['cc_max'] > cc_threshold], 
                        wave_results['baz_max'][wave_results['cc_max'] > cc_threshold],
                        c='tab:blue',
                        s=70,
                        alpha=0.7,
                        edgecolors="k",
                        lw=1,
                        zorder=3
                    )
                    ax3.scatter(
                        wave_results['twin_center'][wave_results['cc_max'] < -cc_threshold], 
                        wave_results['baz_max'][wave_results['cc_max'] < -cc_threshold],
                        c='tab:orange',
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
            else:
                baz_estimated[wave_type] = nan
            
            print(f"\nEstimated BAZ {label} = {baz_estimated[wave_type]}° (CC ≥ {cc_threshold})")
    
        if plot:

            # add histograms and KDEs to subplots
            for ax, ax_hist, label in [(ax1, ax11, "love"), (ax2, ax22, "rayleigh"), (ax3, ax33, "tangent")]:
                if len(baz_filtered) > 0:
                    ax_hist.hist(results_dict[label]['backazimuth'], 
                                bins=len(angles1)-1,
                                range=[min(angles1), max(angles1)],
                                weights=results_dict[label]['correlation'],
                                orientation="horizontal", density=True, color="grey")
                    if len(baz_filtered) > 5:
                        ax_hist.plot(results_dict[label]['kde'],
                                    results_dict[label]['kde_angles'],
                                    color='k', lw=3)
                    ax_hist.yaxis.tick_right()
                    ax_hist.invert_xaxis()
                    ax_hist.set_ylim(-5, 365)

            # Add theoretical BAZ if provided
            if baz_theo is not None:
                for ax in [ax1, ax2, ax3]:
                    ax.plot([t1, t2], [baz_theo, baz_theo], color='k', ls='--', label='Theoretical BAZ')
                    ax.fill_between([t1, t2], 
                                  baz_theo-baz_theo_margin,
                                  baz_theo+baz_theo_margin,
                                  color='grey', alpha=0.3, zorder=1)
            
            # Configure axes
            for ax in [ax1, ax2, ax3]:
                ax.set_ylim(-5, 365)
                ax.set_yticks(range(0, 360+60, 60))
                ax.grid(True, alpha=0.3)
                ax.set_xlim(t1, t2*1.15)
                if minors:
                    ax.minorticks_on()
            
            # Add labels
            ax1.set_title(f"Love Wave BAZ (estimated = {baz_estimated['love']}°)", fontsize=font)
            ax2.set_title(f"Rayleigh Wave BAZ (estimated = {baz_estimated['rayleigh']}°)", fontsize=font)
            ax3.set_title(f"Tangent BAZ (estimated = {baz_estimated['tangent']}°)", fontsize=font)
            ax3.set_xlabel("Time (s)", fontsize=font)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_ylabel("BAZ (°)", fontsize=font)
            
            # Add colorbar
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
    def fit_gaussian_to_kde(angles: array, kde_values: array) -> Dict:
        """
        Fit a Gaussian to KDE values with mean fixed at KDE maximum.
        
        Parameters
        ----------
        angles : array
            Array of angle values (x-axis)
        kde_values : array
            Array of KDE probability density values (y-axis)
            
        Returns
        -------
        Dict
            Dictionary containing the fit parameters:
            - mean: center of the Gaussian (fixed at KDE maximum)
            - std: standard deviation
            - amplitude: peak height
            - r_squared: R-squared value of the fit
        """
        from scipy.optimize import curve_fit
        import numpy as np
        
        # Fix mean at KDE maximum
        mean = angles[np.argmax(kde_values)]
        
        def gaussian(x, amplitude, std):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        
        # Initial parameter guesses
        p0 = [np.max(kde_values),  # amplitude
              30.0]  # initial std guess
              
        # Fit the Gaussian
        popt, _ = curve_fit(gaussian, angles, kde_values, p0=p0)
        
        # Calculate R-squared
        residuals = kde_values - gaussian(angles, *popt)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((kde_values - np.mean(kde_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'mean': mean,
            'std': abs(popt[1]),  # ensure positive std
            'amplitude': popt[0],
            'r_squared': r_squared
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

    def plot_waveform_cc(self, runit: str=r"rad/s", tunit: str=r"m/s$^2$", wave_type: str="both",
                         twin_sec: int=5, twin_overlap: float=0.5, unitscale: str="nano", t1: UTCDateTime=None, t2: UTCDateTime=None) -> plt.Figure:

        """
        Plot waveform cross-correlation.

        Parameters:
        -----------
        wave_type : str
            Wave type: "love", "rayleigh", or "both"
        runit : str
            Unit for rotation rate
        tunit : str
            Unit for acceleration
        twin_sec : int
            Time window length
        twin_overlap : float
            Time window overlap
        unitscale : str
            Unit scale: "nano" or "micro"
        t1 : UTCDateTime
            Start time
        t2 : UTCDateTime
            End time
        Returns:
        --------
        fig : plt.Figure
            Figure object

        """
        from obspy.signal.cross_correlation import correlate
        from obspy.signal.rotate import rotate_ne_rt
        from numpy import linspace, ones, array
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator

        def _cross_correlation_windows(arr1: array, arr2: array, dt: float, Twin: float, overlap: float=0, lag: int=0, demean: bool=True, plot: bool=False) -> Tuple[array, array]:

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
                shift, val = xcorr_max(ccf, abs_max=False)
                cc.append(val)

            return array(times), array(cc)

        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()

        if t1 is not None and t2 is not None:
            rot = rot.trim(t1, t2)
            acc = acc.trim(t1, t2)

        # get backazimuth and distance
        baz = self.event_info['backazimuth']
        distance = self.event_info['distance_km']

        # get frequency range if filter has been applied
        fmin = self.fmin if self.fmin is not None else None
        fmax = self.fmax if self.fmax is not None else None

        # get sampling rate
        dt = rot[0].stats.delta

        # define polarity
        pol = self.pol_dict.copy()
        pol.update({"HR":1,"HT":1,"JR":1,"JT":1})

        # Change number of rows based on wave type
        if wave_type == "both":
            Nrow, Ncol = 2, 1
            fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5*Nrow), sharex=True)
            plt.subplots_adjust(hspace=0.1)
            ax = axes  # axes is already an array for multiple subplots
        else:
            Nrow, Ncol = 1, 1
            fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5), sharex=True)
            ax = [axes]  # wrap single axes in list for consistent indexing
        
        # define scaling factors
        mu = r"$\mu$"
        if unitscale == "nano":
            acc_scaling, acc_unit = 1e6, f"{mu}{tunit}"
            rot_scaling, rot_unit = 1e9, f"n{runit}"
        elif unitscale == "micro":
            acc_scaling, acc_unit = 1e3, f"m{tunit}"
            rot_scaling, rot_unit = 1e6, f"{mu}{runit}"

        # define linewidth and fontsize
        lw = 1
        font = 12

        cc = []
        cc_all = []

        # Get vertical and rotated components
        if wave_type == "both" or wave_type == "love":
            # get vertical component
            rot_z = rot.select(channel="*Z")[0].data
            # rotate components
            acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data, acc.select(channel="*E")[0].data, baz)
            # apply scaling
            rot_z *= rot_scaling
            acc_r *= acc_scaling
            acc_t *= acc_scaling
            # calculate max values
            acc_r_max = max([abs(min(acc_r)), abs(max(acc_r))])
            acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])
            rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
            # update polarity
            # rot0, acc0, rot0_lbl, acc0_lbl = pol['JZ']*rot_z, pol['HT']*acc_t, f"{pol['JZ']}x ROT-Z", f"{pol['HT']}x ACC-T"
            rot0, acc0, rot0_lbl, acc0_lbl = rot_z, acc_t, f"{pol['JZ']}x ROT-Z", f"{pol['HT']}x ACC-T"
            # calculate cross-correlation
            tt0, cc0 = _cross_correlation_windows(rot0, acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
            cc.append(cc0)
            cc_all.append(max(correlate(rot0, acc0, 0, demean=True, normalize='naive', method='fft')))

        if wave_type == "both" or wave_type == "rayleigh":
            # get vertical component
            acc_z = acc.select(channel="*Z")[0].data
            # rotate components
            rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data, rot.select(channel="*E")[0].data, baz)
            # apply scaling
            acc_z *= acc_scaling
            rot_r *= rot_scaling
            rot_t *= rot_scaling
            # calculate max values
            acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
            rot_r_max = max([abs(min(rot_r)), abs(max(rot_r))])
            rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])
            # update polarity
            # rot1, acc1, rot1_lbl, acc1_lbl = pol['JT']*rot_t, pol['HZ']*acc_z, f"{pol['JT']}x ROT-T", f"{pol['HZ']}x ACC-Z"
            rot1, acc1, rot1_lbl, acc1_lbl = rot_t, acc_z, f"{pol['JT']}x ROT-T", f"{pol['HZ']}x ACC-Z"
            # calculate cross-correlation
            tt1, cc1 = _cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
            cc.append(cc1)
            cc_all.append(max(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='fft')))

        # rot2, acc2, rot2_lbl, acc2_lbl = pol['JZ']*rot_z, pol['HR']*acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
        rot2, acc2, rot2_lbl, acc2_lbl = rot_z, acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
        # tt2, cc2 = _cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

        cmap = plt.get_cmap("coolwarm", 12)

        if wave_type == "love":
            ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
            ax00 = ax[0].twinx()
            ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
            ax01 = ax[0].twinx()
            cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, label="")

            ax[0].set_ylim(-rot_z_max, rot_z_max)
            ax00.set_ylim(-acc_t_max, acc_t_max)
            ax01.set_ylim(-1, 1)
            ax01.yaxis.set_visible(False)

            twinaxs = [ax00]
            cms = [cm1]

        elif wave_type == "rayleigh":
            ax[0].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
            ax11 = ax[0].twinx()
            ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
            ax12 = ax[0].twinx()
            cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, label="")

            ax[0].set_ylim(-rot_t_max, rot_t_max)
            ax11.set_ylim(-acc_z_max, acc_z_max)
            ax12.set_ylim(-1, 1)
            ax12.yaxis.set_visible(False)

            twinaxs = [ax11]
            cms = [cm2]

        elif wave_type == "both":
            # First subplot
            ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
            ax00 = ax[0].twinx()
            ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
            ax01 = ax[0].twinx()
            cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, label="")

            ax[0].set_ylim(-rot_z_max, rot_z_max)
            ax00.set_ylim(-acc_t_max, acc_t_max)
            ax01.set_ylim(-1, 1)
            ax01.yaxis.set_visible(False)

            # Second subplot
            ax[1].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
            ax11 = ax[1].twinx()
            ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
            ax12 = ax[1].twinx()
            cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, label="")

            ax[1].set_ylim(-rot_t_max, rot_t_max)
            ax11.set_ylim(-acc_z_max, acc_z_max)
            ax12.set_ylim(-1, 1)
            ax12.yaxis.set_visible(False)

            twinaxs = [ax00, ax11]
            cms = [cm1, cm2]

        # Sync twinx axes
        ax[0].set_yticks(linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-1], len(ax[0].get_yticks())))
        twinaxs[0].set_yticks(linspace(twinaxs[0].get_yticks()[0], twinaxs[0].get_yticks()[-1], len(ax[0].get_yticks())))

        if wave_type == "both":
            ax[1].set_yticks(linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-1], len(ax[1].get_yticks())))
            twinaxs[1].set_yticks(linspace(twinaxs[1].get_yticks()[0], twinaxs[1].get_yticks()[-1], len(ax[1].get_yticks())))

        # Set labels and grid
        rot_rate_label = r"$\dot{\Omega}$"
        if wave_type == "both":
            names = ["love", "rayleigh"]
        else:
            names = [wave_type]

        for i, wt in zip(range(Nrow), names):
            ax[i].legend(loc=1, ncols=4)
            ax[i].grid(which="both", alpha=0.5)
            ax[i].set_ylabel(f"{rot_rate_label} ({rot_unit})", fontsize=font)
            ax[i].text(0.05, 0.9,
                       f"{wt.capitalize()}: CC={cc_all[i]:.2f}",
                       ha='left', va='top', 
                       transform=ax[i].transAxes, 
                       fontsize=font-1,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
                       )

        for _ax in twinaxs:
            _ax.legend(loc=1, bbox_to_anchor=(1, 0.9))
            _ax.set_ylabel(f"$a$ ({acc_unit})", fontsize=font)

        # Add colorbar
        cax = ax[Nrow-1].inset_axes([0.8, -0.25, 0.2, 0.1], transform=ax[Nrow-1].transAxes)

        # Create a ScalarMappable for the colorbar
        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, location="bottom", orientation="horizontal")

        cbar.set_label("Cross-Correlation Value", fontsize=font-1, loc="left", labelpad=-55, color="k")

        # Set limits for scatter plots
        for cm in cms:
            cm.set_clim(-1, 1)

        # set subticks for x axis
        for a in ax:
            a.xaxis.set_minor_locator(AutoMinorLocator())

        # Add xlabel to bottom subplot
        ax[Nrow-1].set_xlabel("Time (s)", fontsize=font)

        # Set title
        tbeg = acc[0].stats.starttime
        title = f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC"
        title += f" | {wave_type}"
        title += f" | f = {fmin}-{fmax} Hz"
        if baz is not None:
            title += f"  |  BAz = {round(baz, 1)}°"
        if distance is not None:
            title += f"  |  ED = {round(distance, 0)} km"
        title += f"  |  T = {twin_sec}s ({int(100*twin_overlap)}%)"
        ax[0].set_title(title)

        # plt.show()
        return fig

    # @staticmethod
    # def plot_waveform_cc(rot0: Stream, acc0: Stream, baz: float, fmin: Optional[float]=None, fmax: Optional[float]=None, wave_type: str="both",
    #                      pol_dict: Union[None, Dict]=None, distance: Union[None, float]=None, runit: str=r"rad/s", tunit: str=r"m/s$^2$",
    #                      twin_sec: int=5, twin_overlap: float=0.5, unitscale: str="nano") -> plt.Figure:

    #     """
    #     Plot waveform cross-correlation.

    #     Parameters:
    #     -----------
    #     rot0 : Stream
    #         Rotation rate stream
    #     acc0 : Stream
    #         Acceleration stream
    #     baz : float
    #         Backazimuth
    #     fmin : float or None
    #         Minimum frequency for bandpass filter
    #     fmax : float or None
    #         Maximum frequency for bandpass filter
    #     wave_type : str
    #         Wave type: "love", "rayleigh", or "both"
    #     pol_dict : dict or None
    #         Polarity dictionary
    #     distance : float or None
    #         Distance
    #     runit : str
    #         Unit for rotation rate
    #     tunit : str
    #         Unit for acceleration
    #     twin_sec : int
    #         Time window length
    #     twin_overlap : float
    #         Time window overlap
    #     unitscale : str
    #         Unit scale: "nano" or "micro"

    #     Returns:
    #     --------
    #     fig : plt.Figure
    #         Figure object

    #     """
    #     from obspy.signal.cross_correlation import correlate
    #     from obspy.signal.rotate import rotate_ne_rt
    #     from numpy import linspace, ones, array
    #     import matplotlib.pyplot as plt
    #     from matplotlib.ticker import AutoMinorLocator

    #     def _cross_correlation_windows(arr1: array, arr2: array, dt: float, Twin: float, overlap: float=0, lag: int=0, demean: bool=True, plot: bool=False) -> Tuple[array, array]:

    #         from obspy.signal.cross_correlation import correlate, xcorr_max
    #         from numpy import arange, array, roll

    #         N = len(arr1)
    #         n_interval = int(Twin/dt)
    #         n_overlap = int(overlap*Twin/dt)

    #         # time = arange(0, N*dt, dt)

    #         times, samples = [], []
    #         n1, n2 = 0, n_interval
    #         while n2 <= N:
    #             samples.append((n1, n2))
    #             times.append(int(n1+(n2-n1)/2)*dt)
    #             n1 = n1 + n_interval - n_overlap
    #             n2 = n2 + n_interval - n_overlap

    #         cc = []
    #         for _n, (n1, n2) in enumerate(samples):

    #             _arr1 = roll(arr1[n1:n2], lag)
    #             _arr2 = arr2[n1:n2]
    #             ccf = correlate(_arr1, _arr2, 0, demean=demean, normalize='naive', method='fft')
    #             shift, val = xcorr_max(ccf, abs_max=False)
    #             cc.append(val)

    #         return array(times), array(cc)

    #     rot = rot0.copy()
    #     acc = acc0.copy()

    #     # get sampling rate
    #     dt = rot[0].stats.delta

    #     # define polarity
    #     pol = {"HZ":1,"HN":1,"HE":1,"HR":1,"HT":1,
    #            "JZ":1,"JN":1,"JE":1,"JR":1,"JT":1,
    #           }
        
    #     # update polarity dictionary
    #     if pol_dict is not None:
    #         for k in pol_dict.keys():
    #             pol[k] = pol_dict[k]

    #     # Change number of rows based on wave type
    #     if wave_type == "both":
    #         Nrow, Ncol = 2, 1
    #         fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5*Nrow), sharex=True)
    #         plt.subplots_adjust(hspace=0.1)
    #         ax = axes  # axes is already an array for multiple subplots
    #     else:
    #         Nrow, Ncol = 1, 1
    #         fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5), sharex=True)
    #         ax = [axes]  # wrap single axes in list for consistent indexing
        
    #     # define scaling factors
    #     mu = r"$\mu$"
    #     if unitscale == "nano":
    #         acc_scaling, acc_unit = 1e6, f"{mu}{tunit}"
    #         rot_scaling, rot_unit = 1e9, f"n{runit}"
    #     elif unitscale == "micro":
    #         acc_scaling, acc_unit = 1e3, f"m{tunit}"
    #         rot_scaling, rot_unit = 1e6, f"{mu}{runit}"

    #     # define linewidth and fontsize
    #     lw = 1
    #     font = 12

    #     cc = []
    #     cc_all = []

    #     # Get vertical and rotated components
    #     if wave_type == "both" or wave_type == "love":
    #         # get vertical component
    #         rot_z = rot.select(channel="*Z")[0].data
    #         # rotate components
    #         acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data, acc.select(channel="*E")[0].data, baz)
    #         # apply scaling
    #         rot_z *= rot_scaling
    #         acc_r *= acc_scaling
    #         acc_t *= acc_scaling
    #         # calculate max values
    #         acc_r_max = max([abs(min(acc_r)), abs(max(acc_r))])
    #         acc_t_max = max([abs(min(acc_t)), abs(max(acc_t))])
    #         rot_z_max = max([abs(min(rot_z)), abs(max(rot_z))])
    #         # update polarity
    #         rot0, acc0, rot0_lbl, acc0_lbl = pol['JZ']*rot_z, pol['HT']*acc_t, f"{pol['JZ']}x ROT-Z", f"{pol['HT']}x ACC-T"
    #         # calculate cross-correlation
    #         tt0, cc0 = _cross_correlation_windows(rot0, acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
    #         cc.append(cc0)
    #         cc_all.append(max(correlate(rot0, acc0, 0, demean=True, normalize='naive', method='fft')))

    #     if wave_type == "both" or wave_type == "rayleigh":
    #         # get vertical component
    #         acc_z = acc.select(channel="*Z")[0].data
    #         # rotate components
    #         rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data, rot.select(channel="*E")[0].data, baz)
    #         # apply scaling
    #         acc_z *= acc_scaling
    #         rot_r *= rot_scaling
    #         rot_t *= rot_scaling
    #         # calculate max values
    #         acc_z_max = max([abs(min(acc_z)), abs(max(acc_z))])
    #         rot_r_max = max([abs(min(rot_r)), abs(max(rot_r))])
    #         rot_t_max = max([abs(min(rot_t)), abs(max(rot_t))])
    #         # update polarity
    #         rot1, acc1, rot1_lbl, acc1_lbl = pol['JT']*rot_t, pol['HZ']*acc_z, f"{pol['JT']}x ROT-T", f"{pol['HZ']}x ACC-Z"
    #         # calculate cross-correlation
    #         tt1, cc1 = _cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
    #         cc.append(cc1)
    #         cc_all.append(max(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='fft')))

    #     # rot2, acc2, rot2_lbl, acc2_lbl = pol['JZ']*rot_z, pol['HR']*acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
    #     # tt2, cc2 = _cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

    #     cmap = plt.get_cmap("coolwarm", 12)

    #     if wave_type == "love":
    #         ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
    #         ax00 = ax[0].twinx()
    #         ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
    #         ax01 = ax[0].twinx()
    #         cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, label="")

    #         ax[0].set_ylim(-rot_z_max, rot_z_max)
    #         ax00.set_ylim(-acc_t_max, acc_t_max)
    #         ax01.set_ylim(-1, 1)
    #         ax01.yaxis.set_visible(False)

    #         twinaxs = [ax00]
    #         cms = [cm1]

    #     elif wave_type == "rayleigh":
    #         ax[0].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
    #         ax11 = ax[0].twinx()
    #         ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
    #         ax12 = ax[0].twinx()
    #         cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, label="")

    #         ax[0].set_ylim(-rot_t_max, rot_t_max)
    #         ax11.set_ylim(-acc_z_max, acc_z_max)
    #         ax12.set_ylim(-1, 1)
    #         ax12.yaxis.set_visible(False)

    #         twinaxs = [ax11]
    #         cms = [cm2]

    #     elif wave_type == "both":
    #         # First subplot
    #         ax[0].plot(rot.select(channel="*Z")[0].times(), rot0, label=rot0_lbl, color="tab:red", lw=lw, zorder=3)
    #         ax00 = ax[0].twinx()
    #         ax00.plot(acc.select(channel="*Z")[0].times(), acc0, label=acc0_lbl, color="black", lw=lw)
    #         ax01 = ax[0].twinx()
    #         cm1 = ax01.scatter(tt0, ones(len(tt0))*-0.9, c=cc0, alpha=abs(cc0), cmap=cmap, label="")

    #         ax[0].set_ylim(-rot_z_max, rot_z_max)
    #         ax00.set_ylim(-acc_t_max, acc_t_max)
    #         ax01.set_ylim(-1, 1)
    #         ax01.yaxis.set_visible(False)

    #         # Second subplot
    #         ax[1].plot(rot.select(channel="*N")[0].times(), rot1, label=rot1_lbl, color="tab:red", lw=lw, zorder=3)
    #         ax11 = ax[1].twinx()
    #         ax11.plot(acc.select(channel="*Z")[0].times(), acc1, label=acc1_lbl, color="black", lw=lw)
    #         ax12 = ax[1].twinx()
    #         cm2 = ax12.scatter(tt1, ones(len(tt1))*-0.9, c=cc1, alpha=abs(cc1), cmap=cmap, label="")

    #         ax[1].set_ylim(-rot_t_max, rot_t_max)
    #         ax11.set_ylim(-acc_z_max, acc_z_max)
    #         ax12.set_ylim(-1, 1)
    #         ax12.yaxis.set_visible(False)

    #         twinaxs = [ax00, ax11]
    #         cms = [cm1, cm2]

    #     # Sync twinx axes
    #     ax[0].set_yticks(linspace(ax[0].get_yticks()[0], ax[0].get_yticks()[-1], len(ax[0].get_yticks())))
    #     twinaxs[0].set_yticks(linspace(twinaxs[0].get_yticks()[0], twinaxs[0].get_yticks()[-1], len(ax[0].get_yticks())))

    #     if wave_type == "both":
    #         ax[1].set_yticks(linspace(ax[1].get_yticks()[0], ax[1].get_yticks()[-1], len(ax[1].get_yticks())))
    #         twinaxs[1].set_yticks(linspace(twinaxs[1].get_yticks()[0], twinaxs[1].get_yticks()[-1], len(ax[1].get_yticks())))

    #     # Set labels and grid
    #     rot_rate_label = r"$\dot{\Omega}$"
    #     if wave_type == "both":
    #         names = ["love", "rayleigh"]
    #     else:
    #         names = [wave_type]

    #     for i, wt in zip(range(Nrow), names):
    #         ax[i].legend(loc=1, ncols=4)
    #         ax[i].grid(which="both", alpha=0.5)
    #         ax[i].set_ylabel(f"{rot_rate_label} ({rot_unit})", fontsize=font)
    #         ax[i].text(0.05, 0.9,
    #                    f"{wt.capitalize()}: CC={cc_all[i]:.2f}",
    #                    ha='left', va='top', 
    #                    transform=ax[i].transAxes, 
    #                    fontsize=font-1,
    #                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
    #                    )

    #     for _ax in twinaxs:
    #         _ax.legend(loc=1, bbox_to_anchor=(1, 0.9))
    #         _ax.set_ylabel(f"$a$ ({acc_unit})", fontsize=font)

    #     # Add colorbar
    #     cax = ax[Nrow-1].inset_axes([0.8, -0.25, 0.2, 0.1], transform=ax[Nrow-1].transAxes)

    #     # Create a ScalarMappable for the colorbar
    #     norm = plt.Normalize(-1, 1)
    #     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    #     sm.set_array([])
    #     cbar = plt.colorbar(sm, cax=cax, location="bottom", orientation="horizontal")

    #     cbar.set_label("Cross-Correlation Value", fontsize=font-1, loc="left", labelpad=-55, color="k")

    #     # Set limits for scatter plots
    #     for cm in cms:
    #         cm.set_clim(-1, 1)

    #     # set subticks for x axis
    #     for a in ax:
    #         a.xaxis.set_minor_locator(AutoMinorLocator())

    #     # Add xlabel to bottom subplot
    #     ax[Nrow-1].set_xlabel("Time (s)", fontsize=font)

    #     # Set title
    #     tbeg = acc[0].stats.starttime
    #     title = f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC"
    #     title += f" | {wave_type}"
    #     title += f" | f = {fmin}-{fmax} Hz"
    #     if baz is not None:
    #         title += f"  |  BAz = {round(baz, 1)}°"
    #     if distance is not None:
    #         title += f"  |  ED = {round(distance, 0)} km"
    #     title += f"  |  T = {twin_sec}s ({int(100*twin_overlap)}%)"
    #     ax[0].set_title(title)

    #     # plt.show()
    #     return fig

    @staticmethod
    def compute_cwt(times: array, data: array, dt: float, datalabel: str="data", log: bool=False, 
                    period: bool=False, tscale: str='sec', scale_value: float=2, 
                    ymax: Union[float, None]=None, normalize: bool=True, plot: bool=False) -> Dict:
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
            Maximum y-axis limit
        normalize : bool
            Normalize wavelet power if True
        plot : bool
            Generate diagnostic plot if True
            
        Returns:
        --------
        Dict : CWT analysis results
        """
        from pycwt import cwt, Morlet
        from numpy import std, nanmean, nan, nansum, nanmax, polyfit, polyval, array, reshape, nanpercentile, ones
        
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
        dj = 1/12
        J = int(7/dj)
        
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

    @staticmethod
    def plot_cwt_all(rot: Stream, acc: Stream, cwt_output: Dict, clog: bool=False, 
                     ylim: Union[float, None]=None) -> plt.Figure:
        """
        Plot continuous wavelet transform analysis for all components of rotation and translation
        
        Parameters:
        -----------
        rot : Stream
            Rotation rate stream
        acc : Stream
            Acceleration stream
        cwt_output : Dict
            Dictionary containing CWT results for each component
        clog : bool
            Use logarithmic colorscale if True
        ylim : float or None
            Upper frequency limit for plotting
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        # Plot settings
        tscale = 1
        font = 12
        cmap = plt.get_cmap("viridis")
        rot_scale = 1e6
        acc_scale = 1e3

        # Count total components and calculate needed subplots
        n_panels = len(cwt_output.keys())
        n_components = len(rot) + len(acc)
        
        # Create figure with GridSpec
        # Each component needs 2 rows - one for waveform and one for CWT
        fig = plt.figure(figsize=(15, 4*n_panels))
        gs = GridSpec(2*n_panels, 1, figure=fig, height_ratios=[1, 3]*n_panels, hspace=0.3)

        # Component mapping
        components = []
        for tr in rot:
            components.append((tr.stats.channel, 'Rotation'))
        for tr in acc:
            components.append((tr.stats.channel, 'Translation'))
        
        # Set colormap limits
        if clog:
            vmin, vmax, norm = 0.01, 1, "log"
        else:
            vmin, vmax, norm = 0.0, 0.9, None
            
        # Plot each component
        for i, (comp, data_type) in enumerate(components):
            wave_ax = fig.add_subplot(gs[2*i])
            cwt_ax = fig.add_subplot(gs[2*i+1])
            
            # Get data and scale
            if data_type == 'Rotation':
                tr = rot.select(channel=f"*{comp}")[0]
                data = tr.data * rot_scale
                unit = r"$\mu$rad"
                label = f"$\Omega_{comp[-1]}$"
            else:
                tr = acc.select(channel=f"*{comp}")[0]
                data = tr.data * acc_scale
                unit = r"mm/s$^2$"
                label = f"$a_{comp[-1]}$"
            
            # Get times from the current trace instead of rotation stream
            times = tr.times() * tscale
            
            # Plot waveform
            wave_ax.plot(times, data, color="k", label=label, lw=1)
            wave_ax.set_xlim(min(times), max(times))
            wave_ax.legend(loc=1)
            wave_ax.set_xticklabels([])
            wave_ax.set_ylabel(f"{label}\n({unit})", fontsize=font)
            wave_ax.grid(True, alpha=0.3)
            
            # Plot CWT
            key = f"{comp}"
            im = cwt_ax.pcolormesh(
                cwt_output[key]['times'] * tscale,
                cwt_output[key]['frequencies'],
                cwt_output[key]['cwt_power'],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                rasterized=True
            )
            
            # Add cone of influence
            cwt_ax.plot(
                cwt_output[key]['times'] * tscale,
                cwt_output[key]['cone'],
                color="white",
                ls="--",
                alpha=0.7
            )
            cwt_ax.fill_between(
                cwt_output[key]['times'] * tscale,
                cwt_output[key]['cone'],
                min(cwt_output[key]['frequencies']) * np.ones(len(cwt_output[key]['cone'])),
                color="white",
                alpha=0.2
            )
            
            # Set frequency limits
            if ylim is None:
                cwt_ax.set_ylim(min(cwt_output[key]['frequencies']),
                                max(cwt_output[key]['frequencies']))
            else:
                cwt_ax.set_ylim(min(cwt_output[key]['frequencies']), ylim)
            
            cwt_ax.set_yscale('log')
            cwt_ax.set_ylabel("Frequency (Hz)", fontsize=font)
            cwt_ax.grid(True, alpha=0.3)
            
            # Only add xlabel to bottom subplot
            if i == len(components) - 1:
                cwt_ax.set_xlabel(f"Time (s) from {rot[0].stats.starttime.date} "
                                f"{str(rot[0].stats.starttime.time).split('.')[0]} UTC",
                                fontsize=font)
            
            # Add subplot labels
            wave_ax.text(.005, .97, f"({chr(97+i*2)})", ha='left', va='top',
                         transform=wave_ax.transAxes, fontsize=font+2)
            cwt_ax.text(.005, .97, f"({chr(98+i*2)})", ha='left', va='top',
                        transform=cwt_ax.transAxes, fontsize=font+2, color="w")
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.7])
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.set_label("Normalized CWT Power", fontsize=font)
        
        plt.subplots_adjust(right=0.9)
        return fig

    @staticmethod
    def plot_spectra_comparison_fill(rot: Stream, acc: Stream, fmin: Union[float, None]=None, fmax: Union[float, None]=None, 
                                   ylog: bool=False, xlog: bool=False, fill: bool=False) -> plt.Figure:
        """
        Plot power spectral density comparison between rotation and acceleration data with horizontal layout
        
        Parameters:
        -----------
        rot : Stream
            Rotation rate stream
        acc : Stream
            Acceleration stream
        fmin : float or None
            Minimum frequency for bandpass filter
        fmax : float or None
            Maximum frequency for bandpass filter
        ylog : bool
            Use logarithmic y-axis scale if True
        xlog : bool
            Use logarithmic x-axis scale if True
        fill : bool
            Fill the area under curves if True
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import multitaper as mt
        from numpy import reshape, max, min
        
        def _multitaper_psd(arr: array, dt: float, n_win: int=5, time_bandwidth: float=4.0) -> Tuple[array, array]:
            """Calculate multitaper power spectral density"""
            out_psd = mt.MTSpec(arr, nw=time_bandwidth, kspec=n_win, dt=dt, iadapt=2)
            _f, _psd = out_psd.rspec()
            return reshape(_f, _f.size), reshape(_psd, _psd.size)

        # Calculate PSDs for each component
        Tsec = 5
        components = [
            ('Z', '*Z'), ('N', '*N'), ('E', '*E')
        ]
        psds = {}
        for comp_name, comp_pattern in components:
            f1, psd1 = _multitaper_psd(
                rot.select(channel=comp_pattern)[0].data, 
                rot[0].stats.delta,
                n_win=Tsec
            )
            f2, psd2 = _multitaper_psd(
                acc.select(channel=comp_pattern)[0].data, 
                acc[0].stats.delta,
                n_win=Tsec
            )
            psds[comp_name] = {'rot': (f1, psd1), 'acc': (f2, psd2)}

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.3)

        # Plot settings
        font = 12
        lw = 1
        rot_color = "darkred"
        acc_color = "black"
        alpha = 0.5 if fill else 1.0

        # Add title with time information
        title = f"{rot[0].stats.starttime.date} {str(rot[0].stats.starttime.time).split('.')[0]} UTC"
        if fmin is not None and fmax is not None:
            title += f" | {fmin}-{fmax} Hz"
        fig.suptitle(title, fontsize=font+2, y=1.02)

        # Plot each component
        for i, (comp_name, comp_data) in enumerate(psds.items()):
            # Get component labels
            rot_label = f"{rot[0].stats.station}.{rot.select(channel=f'*{comp_name}')[0].stats.channel}"
            acc_label = f"{acc[0].stats.station}.{acc.select(channel=f'*{comp_name}')[0].stats.channel}"
            
            if fill:
                # Plot with fill
                axes[i].fill_between(
                    comp_data['rot'][0],
                    comp_data['rot'][1],
                    lw=lw,
                    label=rot_label,
                    color=rot_color,
                    alpha=alpha,
                    zorder=3
                )
                ax2 = axes[i].twinx()
                ax2.fill_between(
                    comp_data['acc'][0],
                    comp_data['acc'][1],
                    lw=lw,
                    label=acc_label,
                    color=acc_color,
                    alpha=alpha,
                    zorder=2
                )
            else:
                # Plot lines
                axes[i].plot(
                    comp_data['rot'][0],
                    comp_data['rot'][1],
                    lw=lw,
                    label=rot_label,
                    color=rot_color,
                    ls="-",
                    zorder=3
                )
                ax2 = axes[i].twinx()
                ax2.plot(
                    comp_data['acc'][0],
                    comp_data['acc'][1],
                    lw=lw,
                    label=acc_label,
                    color=acc_color,
                    zorder=2
                )
            
            # Configure axes
            axes[i].legend(loc=1, ncols=4)
            if xlog:
                axes[i].set_xscale("log")
            if ylog:
                axes[i].set_yscale("log")
                ax2.set_yscale("log")
            
            # axes[i].grid(which="both", alpha=0.5)
            axes[i].tick_params(axis='y', colors=rot_color)
            axes[i].set_ylim(bottom=0)
            ax2.set_ylim(bottom=0)
            
            # Set frequency limits
            xlim_right = fmax if fmax else rot[0].stats.sampling_rate * 0.5
            axes[i].set_xlim(left=fmin, right=xlim_right)
            ax2.set_xlim(left=fmin, right=xlim_right)
            axes[i].set_xlabel("Frequency (Hz)", fontsize=font)

            # Set legends
            ax2.legend(loc=2)

            # For the last panel (E component), don't create new y-axis ticks on the right
            if i == 2:
                ax2.set_ylabel(r"PSD (m$^2$/s$^4$/Hz)", fontsize=font)
            if i == 0:
                axes[i].set_ylabel(r"PSD (rad$^2$/s$^2$/Hz)", fontsize=font, color=rot_color)
            
            # Add component label
            axes[i].set_title(f"Component {comp_name}", fontsize=font)

        # Adjust layout to accommodate supertitle
        plt.subplots_adjust(top=0.90)
        
        return fig

    @staticmethod
    def plot_cwt(st: Stream, cwt_output: Dict, clog: bool=False, 
                ylim: Union[float, None]=None, scale: float=1e6) -> plt.Figure:
        """
        Plot continuous wavelet transform analysis for all components of rotation and translation
        
        Parameters:
        -----------
        st : Stream
            Stream of data to plot
        cwt_output : Dict
            Dictionary containing CWT results for each component
        clog : bool
            Use logarithmic colorscale if True
        ylim : float or None
            Upper frequency limit for plotting
        scale : float
            Scale factor for data
            
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.gridspec import GridSpec
        
        # Plot settings
        tscale = 1
        font = 12
        cmap = plt.get_cmap("viridis")

        # decide if rotation or translation data 
        if "J" in st[0].stats.channel:
            if scale == 1e9:
                unit = r"nrad"
            elif scale == 1e6:
                unit = r"$\mu$rad"
            elif scale == 1e3:
                unit = r"mrad"
            else:
                unit = r"rad"
                scale = 1
                print(f"WARNING: unknown scale factor (1e3, 1e6, 1e9): {scale}. Using 1 for scale")
        else:
            if scale == 1e9:
                unit = r"nm/s$^2$"
            elif scale == 1e6:
                unit = r"mm/s$^2$"
            elif scale == 1e3:
                unit = r"m/s$^2$"
            else:
                unit = r"m/s$^2$"
                print(f"WARNING: unknown scale factor (1e3, 1e6, 1e9): {scale}. Using 1 for scale")
                scale = 1


        # Create figure with GridSpec
        # Each component needs 2 rows - one for waveform and one for CWT
        fig = plt.figure(figsize=(15, 4))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 3], hspace=0.3)

        # Set colormap limits
        if clog:
            vmin, vmax, norm = 0.01, 1, "log"
        else:
            vmin, vmax, norm = 0.0, 0.9, None
            
        # Plot each component
        wave_ax = fig.add_subplot(gs[0])
        cwt_ax = fig.add_subplot(gs[1])
            
        # Get data and scale
        tr = st.copy()[0]
        data = tr.data * scale
        label = f"$\Omega_{tr.stats.channel[-1]}$"
        key = f"{tr.stats.channel}"
        
        # Get times from the current trace instead of rotation stream
        times = tr.times() * tscale
        
        # Plot waveform
        wave_ax.plot(times, data, color="k", label=label, lw=1)
        wave_ax.set_xlim(min(times), max(times))
        wave_ax.legend(loc=1)
        wave_ax.set_xticklabels([])
        wave_ax.set_ylabel(f"{label}\n({unit})", fontsize=font)
        wave_ax.grid(True, alpha=0.3)
        
        # Plot CWT
        im = cwt_ax.pcolormesh(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['frequencies'],
            cwt_output[key]['cwt_power'],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            norm=norm,
            rasterized=True
        )
        
        # Add cone of influence
        cwt_ax.plot(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['cone'],
            color="white",
            ls="--",
            alpha=0.7
        )
        cwt_ax.fill_between(
            cwt_output[key]['times'] * tscale,
            cwt_output[key]['cone'],
            min(cwt_output[key]['frequencies']) * np.ones(len(cwt_output[key]['cone'])),
            color="white",
            alpha=0.2
        )
        
        # Set frequency limits
        if ylim is None:
            cwt_ax.set_ylim(min(cwt_output[key]['frequencies']),
                            max(cwt_output[key]['frequencies']))
        else:
            cwt_ax.set_ylim(min(cwt_output[key]['frequencies']), ylim)
        
        cwt_ax.set_yscale('log')
        cwt_ax.set_ylabel("Frequency (Hz)", fontsize=font)
        cwt_ax.grid(True, alpha=0.3)

        # Only add xlabel to bottom subplot
        cwt_ax.set_xlabel(
            f"Time (s) from {st[0].stats.starttime.date} "
            f"{str(st[0].stats.starttime.time).split('.')[0]} UTC",
            fontsize=font
        )
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.7])
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.set_label("Normalized CWT Power", fontsize=font)
        
        plt.subplots_adjust(right=0.9)
        return fig

    def plot_backazimuth_results(self, baz_results: Dict, wave_type: str='love', 
                                baz_theo: float=None, baz_theo_margin: float=10, unitscale: str='nano',
                                cc_threshold: float=None, minors: bool=True, cc_method: str='mid') -> plt.Figure:
        """
        Plot backazimuth estimation results
        
        Parameters:
        -----------
        baz_results : Dict
            Dictionary containing backazimuth results
        wave_type : str
            Type of wave ('love' or 'rayleigh')
        baz_theo : float, optional
            Theoretical backazimuth in degrees
        baz_theo_margin : float, optional
            Margin around theoretical backazimuth in degrees
        cc_threshold : float, optional
            Minimum cross-correlation coefficient threshold
        minors : bool, optional
            Add minor ticks to axes if True
        cc_method : str
            Type of cc to choose ('mid' or 'max')
        unitscale : str
            Unit scale for rotation rate ('nano' or 'micro')
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the plot
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from obspy.signal.rotate import rotate_ne_rt
        import numpy as np
        from numpy import arange, histogram, average, cov
        import scipy.stats as sts

        # Create figure with GridSpec
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(4, 8, figure=fig, hspace=0.2)
        
        # Create subplots
        ax_wave = fig.add_subplot(gs[0:2, :])  # Waveform panel
        ax_baz = fig.add_subplot(gs[2:3, :])  # Backazimuth panel
        ax_hist = fig.add_subplot(gs[2:3, 7:])  # Histogram panel
        ax_hist.set_axis_off()
        
        # Plot settings
        font = 12
        lw = 1.0
        if unitscale == 'nano':
            rot_scale, rot_unit = 1e9, f"n{self.runit}"
            trans_scale, trans_unit = 1e6, f"{self.mu}{self.tunit}"
        elif unitscale == 'micro':
            rot_scale, rot_unit = 1e6, f"{self.mu}{self.runit}"
            trans_scale, trans_unit = 1e3, f"m{self.tunit}"

        
        # Get streams and apply filtering if needed
        rot = self.get_stream("rotation").copy()
        acc = self.get_stream("translation").copy()

        # Get components
        if wave_type == "love":
            hn = acc.select(channel="*HN")[0].data
            he = acc.select(channel="*HE")[0].data
            jz = rot.select(channel="*JZ")[0].data
        elif wave_type == "rayleigh":
            hz = acc.select(channel="*HZ")[0].data
            je = rot.select(channel="*JE")[0].data
            jn = rot.select(channel="*JN")[0].data
        else:
            raise ValueError(f"Invalid wave_type: {wave_type}. Use 'love' or 'rayleigh'.")
        
        # Rotate to radial-transverse
        if baz_theo is not None:
            if wave_type == "love":
                hr, ht = rotate_ne_rt(hn, he, baz_theo)
            elif wave_type == "rayleigh":
                jr, jt = rotate_ne_rt(jn, je, baz_theo)
        else:
            print("No theoretical backazimuth provided")
            return

        # get times
        time = baz_results['twin_center']

        # select maximal or mid approach results
        if cc_method == 'mid':
            ccc = baz_results['cc_mid']
            baz = baz_results['baz_mid']
        elif cc_method == 'max':
            ccc = baz_results['cc_max']
            baz = baz_results['baz_max']
        
        # apply cc threshold if provided
        if cc_threshold is not None:
            mask = ccc > cc_threshold
            time = time[mask]
            baz = baz[mask]
            cc = ccc[mask]

        # Plot transverse components
        times = acc.select(channel="*HZ")[0].times()

        if wave_type == "love":

            # Plot translational data
            ax_wave.plot(times, ht*trans_scale, 'black', label=f"{self.tra_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[-1][:-1]}T", lw=lw)
            ax_wave.set_ylim(-max(abs(ht*trans_scale)), max(abs(ht*trans_scale)))

            # Add rotational data on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, jz*rot_scale, 'darkred', label=f"{self.rot_seed[0].split('.')[1]}.{self.rot_seed[0].split('.')[-1][:-1]}Z", lw=lw)
            ax_wave2.set_ylim(-max(abs(jz*rot_scale)), max(abs(jz*rot_scale)))
    
        elif wave_type == "rayleigh":
            ax_wave.plot(times, hz*trans_scale, 'black', label=f"{self.tra_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[-1][:-1]}Z", lw=lw)
            ax_wave.set_ylim(-max(abs(hz*trans_scale)), max(abs(hz*trans_scale)))

            # Add rotational data on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, jt*rot_scale, 'darkred', label=f"{self.rot_seed[0].split('.')[1]}.{self.rot_seed[0].split('.')[-1][:-1]}T", lw=lw)
            ax_wave2.set_ylim(-max(abs(jt*rot_scale)), max(abs(jt*rot_scale)))
            
        # Configure waveform axes
        # ax_wave.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
        ax_wave.legend(loc=1)
        ax_wave.set_ylabel(f"Acceleration ({trans_unit})", fontsize=font)
        ax_wave2.tick_params(axis='y', colors="darkred")
        ax_wave2.set_ylabel(f"Rotation rate ({rot_unit})", color="darkred", fontsize=font)
        ax_wave2.legend(loc=4)
        
        # Plot backazimuth estimates
        cmap = plt.get_cmap("viridis", 10)
        scatter = ax_baz.scatter(time, baz,
                               c=cc, s=50, cmap=cmap,
                               edgecolors="k", lw=1, vmin=0, vmax=1, zorder=2)
        
        # Configure backazimuth axis
        ax_baz.set_ylim(-5, 365)
        ax_baz.set_yticks(range(0, 360+60, 60))
        ax_baz.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
        ax_baz.set_ylabel(f"{wave_type.capitalize()} BAz (°)", fontsize=font)
        
        # Add theoretical backazimuth
        ax_baz.plot([min(times), max(times)], [baz_theo, baz_theo],
                    color='k', ls='--', label='Theoretical BAz', zorder=1)
        ax_baz.fill_between([baz_theo-baz_theo_margin, baz_theo+baz_theo_margin],
                           [min(times), min(times)],
                           color='grey', alpha=0.5, zorder=1)

        # Compute statistics
        deltaa = 10
        angles1 = arange(0, 365, deltaa)
    
        # Compute histogram
        hist = histogram(baz,
                         bins=len(angles1)-1,
                         range=[min(angles1), max(angles1)], 
                         weights=cc, 
                         density=True)

        # get kde stats
        try:
            kde_stats = self.get_kde_stats(baz, cc, _baz_steps=0.5, Ndegree=60, plot=False)
            # get max and std
            baz_max = kde_stats['baz_estimate']
            baz_std = kde_stats['kde_dev']
            print(f"baz_max = {baz_max}, baz_std = {baz_std}")
            got_kde = True
        except:
            got_kde = False

        # Add histogram
        # ax_hist2.plot(kernel_density(np.linspace(0, 360, 100)), np.linspace(0, 360, 100), 'k-', lw=2)
        ax_hist.hist(baz, bins=len(angles1)-1, range=[min(angles1), max(angles1)],
                     weights=cc, orientation="horizontal", density=True, color="grey")
        if got_kde:
            ax_hist.plot(kde_stats['kde_values'],
                        kde_stats['kde_angles'],
                        c="k",
                        lw=2,
                        label='KDE'
                        )
        ax_hist.set_ylim(-5, 365)
        ax_hist.invert_xaxis()
        ax_hist.set_axis_off()
        
        # Add colorbar
        cbar_ax = ax_baz.inset_axes([1.02, 0., 0.02, 1])
        cb = plt.colorbar(scatter, cax=cbar_ax)
        cb.set_label("CC coefficient", fontsize=font)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels([0, 0.5, 1])

        # Add title and labels
        title = f"{self.tbeg.date} {str(self.tbeg.time).split('.')[0]} UTC"
        title += f" | {wave_type.capitalize()} Waves"
        if self.fmin is not None and self.fmax is not None:
            title += f" | f = {self.fmin}-{self.fmax} Hz"
        if cc_threshold is not None:
            title += f" | CC > {cc_threshold}"
        if baz_theo is not None:
            title += f" | Theo. BAz = {round(baz_theo, 1)}°"
        if baz_results['parameters']['baz_win_sec'] is not None:
            title += f" | T = {baz_results['parameters']['baz_win_sec']} s ({baz_results['parameters']['baz_win_overlap']*100}%)"
        fig.suptitle(title, fontsize=font+2, y=0.93)
        
        ax_baz.set_xlabel("Time (s)", fontsize=font)

        # Adjust x-axis limits
        ax_wave.set_xlim(min(times), max(times)+0.15*max(times))
        ax_wave2.set_xlim(min(times), max(times)+0.15*max(times))
        ax_baz.set_xlim(min(times), max(times)+0.15*max(times))

        # Add minor ticks
        if minors:
            ax_wave.minorticks_on()
            ax_baz.minorticks_on()
            ax_wave2.minorticks_on()

        return fig

    def plot_velocities(self, velocity_results: Dict, vmax: float=None, minors: bool=True, cc_threshold: float=None) -> plt.Figure:
        """
        Plot waveforms and velocity estimates
        
        Parameters:
        -----------
        velocity_results : Dict
            Results dictionary from compute_velocities
        vmax : float or None
            Maximum velocity for plot scaling
        minors : bool
            Add minor ticks to axes if True
        cc_threshold : float, optional
            Minimum cross-correlation coefficient threshold
        Returns:
        --------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy as np
        
        wave_type = velocity_results['parameters']['wave_type'].lower()

        # Create figure
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(4, 8, figure=fig, hspace=0.2)
        
        # Create subplots
        ax_wave = fig.add_subplot(gs[0:2, :7])  # Waveform panel
        ax_vel = fig.add_subplot(gs[2:4, :7])   # Velocity panel
        
        # Plot settings
        font = 12
        lw = 1.0
        rot_scale, rot_unit = 1e9, f"n{self.runit}"
        tra_scale, tra_unit = 1e6, f"{self.mu}{self.tunit}"
        
        # Get time vector
        times = np.arange(len(self.st[0])) / self.sampling_rate
        
        # get streams
        acc = self.get_stream("translation").copy()
        rot = self.get_stream("rotation").copy()

        # scale waveforms
        for tr in acc:
            tr.data *= tra_scale
        for tr in rot:
            tr.data *= rot_scale

        # rotate waveforms
        if wave_type == "love":
            rot_z = 2*rot.select(channel="*Z")[0].data # times two for velocity scaling (plotting only)
            acc_r, acc_t = rotate_ne_rt(acc.select(channel="*N")[0].data,
                                        acc.select(channel="*E")[0].data,
                                        velocity_results['parameters']['baz'])
            

        elif wave_type == "rayleigh":
            acc_z = acc.select(channel="*Z")[0].data
            rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data,
                                        rot.select(channel="*E")[0].data,
                                        velocity_results['parameters']['baz'])

        # prepare mask
        if cc_threshold is not None:
            mask = velocity_results['ccoef'] > cc_threshold
        else:
            mask = velocity_results['ccoef'] >= 0

        # Plot waveforms based on wave type
        if  wave_type == 'love':

            # Plot transverse acceleration
            ax_wave.plot(times, acc_t, 'black', 
                        label=f"{self.tra_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[3][0]}HT", lw=lw)
            
            # Plot vertical rotation on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, rot_z, 'darkred',
                         label=f"2x {self.rot_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[3][0]}JZ", lw=lw)
            
        elif wave_type == 'rayleigh':

            # Plot vertical acceleration
            ax_wave.plot(times, acc_z, 'black',
                        label=f"{self.tra_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[3][0]}HZ", lw=lw)
            
            # Plot transverse rotation on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, rot_t, 'darkred',
                         label=f"{self.rot_seed[0].split('.')[1]}.{self.tra_seed[0].split('.')[3][0]}JT", lw=lw)

        # Configure waveform axes
        ax_wave.grid(True, which='both', ls='--', alpha=0.3)
        ax_wave.legend(loc=1)
        ax_wave.set_ylabel(f"acceleration ({tra_unit})", fontsize=font)
        ax_wave2.tick_params(axis='y', colors="darkred")
        ax_wave2.set_ylabel(f"rotation rate ({rot_unit})", color="darkred", fontsize=font)
        ax_wave2.legend(loc=4)

        self.sync_twin_axes(ax_wave, ax_wave2)
        
        # Plot velocities
        cmap = plt.get_cmap("viridis", 10)
        scatter = ax_vel.scatter(velocity_results['time'][mask], 
                               velocity_results['velocity'][mask],
                               c=velocity_results['ccoef'][mask], 
                               cmap=cmap, s=70, alpha=1.0,
                               vmin=0, vmax=1, edgecolors="k", lw=1, zorder=2)
        
        # Add error bars
        ax_vel.errorbar(velocity_results['time'][mask], 
                       velocity_results['velocity'][mask],
                       xerr=velocity_results['terr'][mask],
                       color='black', alpha=0.4, ls='none', zorder=1)
        
        # Configure velocity axis
        ax_vel.set_ylabel("phase velocity (m/s)", fontsize=font)
        ax_vel.set_xlabel("time (s)", fontsize=font)
        ax_vel.set_ylim(bottom=0)
        if vmax is not None:
            ax_vel.set_ylim(top=vmax)
        ax_vel.grid(True, which='both', ls='--', alpha=0.3)
        
        for a in [ax_vel, ax_wave]:
            a.set_xlim(0, times.max())

        if minors:
            ax_wave.minorticks_on()
            ax_vel.minorticks_on()
            ax_wave2.minorticks_on()
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cb = plt.colorbar(scatter, cax=cbar_ax)
        cb.set_label('cross-correlation coefficient', fontsize=font)
        
        # Add title
        title = f"{velocity_results['parameters']['wave_type'].capitalize()} Waves"
        title += (f" | {self.tbeg.date} {str(self.tbeg.time).split('.')[0]} UTC"
                 f" | {self.fmin}-{self.fmax} Hz"
                 f" | T = {velocity_results['parameters']['win_time_s']:.1f} s"
                 f" | {velocity_results['parameters']['overlap']*100:.0f}% overlap")
        if cc_threshold is not None:
            title += f" | cc > {cc_threshold}"
        fig.suptitle(title, fontsize=font+2, y=0.95)
        
        # plt.tight_layout()
        plt.show()
        return fig

    def plot_optimization_results(self, params: Dict, wave_type: str='love', vel_max_threshold: float=5000, cc_threshold: float=0.8, baz_theo: float=None) -> plt.Figure:
        """
        Plot optimization results including frequency bands, backazimuth, and velocities
        
        Parameters:
        -----------
        params : Dict
            Dictionary containing optimization results from optimize_parameters()
        wave_type : str
            Type of wave to analyze ('love' or 'rayleigh')
        vel_max_threshold : float
            Maximum velocity threshold in m/s. Points above this will be plotted in grey
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object containing the plots
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import BoundaryNorm, ListedColormap
        
        font = 12

        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, figure=fig, 
                    width_ratios=[1, 0.05], 
                    height_ratios=[1, 1, 1], 
                    hspace=0.25)
        
        # Convert data to arrays if needed
        times = np.array(params['times'])
        freqs = np.array(params['frequency']['center'])
        baz = np.array(params['backazimuth']['optimal'])
        vel = np.array(params['velocity'])
        cc = np.array(params['cross_correlation']['optimal'])
        
        # Create velocity mask
        vel_mask = vel <= vel_max_threshold
        
        # Set colorbar parameters
        vmin, vmax, vstep = cc_threshold, 1.0, 0.01
        levels = np.arange(vmin, vmax + vstep, vstep)  # steps of 0.01
        
        # Create discrete colormap
        n_bins = len(levels) - 1
        viridis = plt.cm.get_cmap('viridis')
        colors = viridis(np.linspace(0, 1, n_bins))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=len(levels)-1)
        
        # 1. Plot frequency band optimization
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(times[~vel_mask], freqs[~vel_mask], 
                    color='grey', alpha=0.3, label=f'v > {vel_max_threshold} m/s')
        sc1 = ax1.scatter(times[vel_mask], freqs[vel_mask], 
                        c=cc[vel_mask], cmap=cmap, alpha=0.7,
                        norm=norm, edgecolor='black')
        ax1.set_ylabel('Frequency (Hz)', fontsize=font)
        ax1.set_yscale('log')
        ax1.set_title('Frequency Band Optimization', fontsize=font)
        ax1.grid(which='both', zorder=0, alpha=0.5)
        ax1.legend()
        ax1.set_ylim(freqs.min()-0.1*freqs.max(), freqs.max()+0.1*freqs.max())
        
        # 2. Plot backazimuth results
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(times[~vel_mask], baz[~vel_mask], 
                    color='grey', alpha=0.3, label=f'v > {vel_max_threshold} m/s')
        sc2 = ax2.scatter(times[vel_mask], baz[vel_mask],
                        c=cc[vel_mask], cmap=cmap, alpha=0.7,
                        norm=norm, edgecolor='black')
        # add theoretical baz if available
        if self.baz_theo is not None:
            ax2.plot([times.min(), times.max()], [self.baz_theo, self.baz_theo],
                     color='black', ls='--', zorder=1, linewidth=2, label='Theoretical BAz')
        elif baz_theo is not None:
            ax2.plot([times.min(), times.max()], [baz_theo, baz_theo],
                     color='black', ls='--', zorder=1, linewidth=2, label='Theoretical BAz')
        ax2.set_ylabel('Backazimuth (°)', fontsize=font)
        ax2.grid(which='both', zorder=0, alpha=0.5)
        ax2.set_title('Optimal Backazimuth', fontsize=font)
        ax2.set_ylim(0, 360)
        ax2.legend()
        ax2.set_yticks([0, 90, 180, 270, 360])
        
        # 3. Plot velocity results
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.scatter(times[~vel_mask], vel[~vel_mask], 
                    color='grey', alpha=0.3)
        sc3 = ax3.scatter(times[vel_mask], vel[vel_mask],
                        c=cc[vel_mask], cmap=cmap, alpha=0.7,
                        norm=norm, edgecolor='black')
        ax3.set_xlabel('Time (s)', fontsize=font)
        ax3.set_ylabel('Velocity (m/s)', fontsize=font)
        ax3.grid(which='both', zorder=0, alpha=0.5)
        ax3.set_title('Phase Velocity', fontsize=font)
        ax3.set_ylim(0, vel_max_threshold)
        
        for ax in [ax1, ax2, ax3]:
            ax.minorticks_on()
            ax.set_xlim(times.min()-0.01*times.max(), times.max()+0.01*times.max())
        
        # Add vertical colorbar
        cax = fig.add_subplot(gs[:, 1])
        cbar = plt.colorbar(sc3, cax=cax, ticks=levels)
        cbar.set_label('Cross-correlation Coefficient', fontsize=font)
        
        plt.suptitle(f'{wave_type.capitalize()} Wave Analysis Results', y=0.95)
        plt.tight_layout()
        
        # Print statistics for values below threshold
        print("\nOptimal Parameters (v ≤ {vel_max_threshold} m/s):")
        print(f"Mean Velocity: {np.nanmean(vel[vel_mask]):.1f} m/s")
        print(f"Mean Backazimuth: {np.nanmean(baz[vel_mask]):.1f}°")
        print(f"Mean Cross-correlation: {np.nanmean(cc[vel_mask]):.3f}")
        print(f"Number of points: {np.sum(vel_mask)} / {len(vel_mask)}")
        
        return fig

    def plot_velocity_method_comparison(self, love_velocities_ransac: Dict, love_velocities_odr: Dict, 
                           cc_threshold: float = 0.75, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create comparison plot of RANSAC and ODR velocity estimates
        
        Parameters
        ----------
        love_velocities_ransac : Dict
            Dictionary containing RANSAC velocity results
        love_velocities_odr : Dict
            Dictionary containing ODR velocity results  
        cc_threshold : float, optional
            Cross-correlation threshold for filtering (default: 0.75)
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches (default: (12, 8))
        
        Returns:
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.lines import Line2D
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(3, 2, width_ratios=[1, 0.03], height_ratios=[1, 1, 1], 
                          hspace=0.2, wspace=0.05)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        cax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        cax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

        # Create mask based on correlation threshold
        mask = love_velocities_ransac['ccoef'] > cc_threshold

        # Create colormap
        vmin, vmax, vstep = 0.5, 1.0, 0.05
        levels = np.arange(vmin, vmax + vstep, vstep)
        n_bins = len(levels) - 1
        viridis = plt.cm.get_cmap('viridis')
        colors = viridis(np.linspace(0, 1, n_bins))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(levels, ncolors=len(levels)-1)

        # Plot RANSAC velocities
        cm1 = ax1.scatter(love_velocities_ransac['time'][mask],
                          love_velocities_ransac['velocity'][mask],
                          c=love_velocities_ransac['ccoef'][mask],
                          cmap=cmap, norm=norm, label='RANSAC', alpha=0.9,
                          edgecolors='black', linewidths=0.5)
        plt.colorbar(cm1, cax=cax1, label='Cross-Correlation Coefficient')

        # Plot ODR velocities
        cm2 = ax2.scatter(love_velocities_odr['time'][mask],
                          love_velocities_odr['velocity'][mask],
                          c=love_velocities_odr['ccoef'][mask],
                          cmap=cmap, norm=norm, label='ODR', alpha=0.9, zorder=2,
                          edgecolors='black', linewidths=0.5)
        plt.colorbar(cm2, cax=cax2, label='Cross-Correlation Coefficient')

        # Plot velocity differences
        ax3.plot([love_velocities_odr['time'][mask], love_velocities_odr['time'][mask]],
                 [np.zeros(len(love_velocities_odr['time'][mask])), 
                  love_velocities_odr['velocity'][mask] - love_velocities_ransac['velocity'][mask]],
                 alpha=0.9, color='black', zorder=2)

        # Add reference line for differences
        ax3.plot([love_velocities_odr['time'][mask][0], love_velocities_odr['time'][mask][0]],
                 [love_velocities_odr['time'][mask][0], 
                  love_velocities_odr['velocity'][mask][0] - love_velocities_ransac['velocity'][mask][0]],
                 label='ODR - RANSAC', alpha=0.9, color='black', zorder=2)

        # Create custom legend elements
        legend_elements1 = [Line2D([0], [0], marker='.', color='w', markerfacecolor=viridis(0.5),
                                  label='RANSAC', markersize=10)]
        legend_elements2 = [Line2D([0], [0], marker='.', color='w', markerfacecolor=viridis(0.5),
                                  label='ODR', markersize=10)]
        legend_elements3 = [Line2D([0], [0], marker='.', color='w', markerfacecolor='black',
                                  label='ODR - RANSAC', markersize=10)]

        # Add legends
        ax1.legend(handles=legend_elements1)
        ax2.legend(handles=legend_elements2)
        ax3.legend(handles=legend_elements3)

        # Configure axes
        for ax in [ax1, ax2, ax3]:
            ax.set_ylabel('Phase Velocity (m/s)')
            ax.grid(which='both', zorder=0, alpha=0.3)
            ax.minorticks_on()

        # Set y-limits
        for ax in [ax1, ax2]:
            ax.set_ylim(0, 6000)

        # Configure difference plot
        ax3.set_ylabel('Velocity Difference (m/s)')
        max_diff = np.nanmax(abs(love_velocities_odr['velocity'][mask] - 
                                love_velocities_ransac['velocity'][mask]))
        ax3.set_ylim(-max_diff, max_diff)

        # Add title and adjust layout
        plt.suptitle('Love Wave Velocity Estimates: RANSAC vs ODR', y=0.92)
        plt.tight_layout()

        return fig

    # OLD
    def plot_velocities_win(self, results_velocities: Dict, cc_threshold: float = 0.0, 
                           baz_theo: Union[float, None] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot Love wave velocity and backazimuth estimates with correlation coefficient coloring
        
        Parameters
        ----------
        results_velocities : Dict
            Dictionary containing velocity analysis results with keys:
            'time', 'backazimuth', 'velocity', 'ccoef'
        cc_threshold : float, optional
            Cross-correlation threshold for filtering (default: 0.0)
        baz_theo : float, optional
            Theoretical backazimuth to plot as reference line (default: None)
        figsize : Tuple[int, int], optional
            Figure size (width, height) in inches (default: (12, 8))
            
        Returns:
        -------
        matplotlib.figure.Figure
            The generated figure
        """
        from numpy import array

        # Convert arrays if needed
        times = array(results_velocities['time'])
        baz = array(results_velocities['backazimuth'])
        vel = array(results_velocities['velocity'])
        cc = array(results_velocities['ccoef'])

        # Apply threshold mask
        mask = cc > cc_threshold

        # Create figure with space for colorbar
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, width_ratios=[15, 0.5], hspace=0.1)

        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        cax = fig.add_subplot(gs[:, 1])  # colorbar axis

        # Plot backazimuth estimates
        sc1 = ax1.scatter(results_velocities['time'][mask], 
                         results_velocities['backazimuth'][mask], 
                         c=results_velocities['ccoef'][mask], 
                         cmap='viridis', 
                         alpha=1, 
                         label='Estimated BAZ', 
                         zorder=2, 
                         vmin=0, 
                         vmax=1,
                         edgecolors='black', 
                         linewidths=0.5)
        
        # Add theoretical backazimuth line if provided
        if baz_theo is not None:
            ax1.axhline(y=baz_theo, color='r', ls='--', label='Theoretical BAZ', zorder=0)

        # Plot velocity estimates
        sc2 = ax2.scatter(results_velocities['time'][mask], 
                         results_velocities['velocity'][mask], 
                         c=results_velocities['ccoef'][mask], 
                         cmap='viridis', 
                         alpha=1, 
                         label='Phase Velocity', 
                         zorder=2, 
                         vmin=0, 
                         vmax=1, 
                         edgecolors='black', 
                         linewidths=0.5)

        # Configure axes
        ax1.set_ylim(0, 360)
        ax1.set_ylabel('Backazimuth (°)')
        ax1.grid(which='both', zorder=0, alpha=0.5)
        ax1.legend()

        ax2.set_ylim(0, 5000)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(which='both', zorder=0, alpha=0.5)
        ax2.legend()

        # Add minor ticks
        for ax in [ax1, ax2]:
            ax.minorticks_on()

        # Add colorbar
        cb = plt.colorbar(sc1, cax=cax, label='Cross-Correlation Coefficient')

        plt.tight_layout()
        return fig

    # OLD
    def plot_backazimuth_map(self, results, event_info=None, map_projection='orthographic', 
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
            colors = {'love': 'darkblue', 'rayleigh': 'red', 'tangent': 'purple'}
            
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
                            linestyle=':', label=f'Theoretical: {theo_baz:.1f}°', alpha=0.9,
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
                            label=f'{wave_type.upper()}: {baz_deg:.1f}°', alpha=0.8,
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

        def _create_map_subplot(fig, gridspec, projection):
            """Create map subplot with appropriate projection"""
            try:
                import cartopy.crs as ccrs
                if projection == 'orthographic':
                    ax = fig.add_subplot(gridspec, projection=ccrs.Orthographic())
                else:
                    ax = fig.add_subplot(gridspec, projection=ccrs.PlateCarree())
                return ax
            except ImportError:
                return fig.add_subplot(gridspec)

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
            ax_map = _create_map_subplot(fig, gs[0, 0], map_projection)
            ax_hist = fig.add_subplot(gs[0, 1])
            hist_axes = [ax_hist]
        elif num_wave_types == 2:
            gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
            ax_map = _create_map_subplot(fig, gs[:, 0], map_projection)
            ax_hist1 = fig.add_subplot(gs[0, 1])
            ax_hist2 = fig.add_subplot(gs[1, 1])
            hist_axes = [ax_hist1, ax_hist2]
        else:  # 3 wave types
            gs = GridSpec(3, 2, figure=fig, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
            ax_map = _create_map_subplot(fig, gs[:, 0], map_projection)
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
            baz_mean = np.average(baz, weights=cc)
            baz_std = np.sqrt(np.average((baz - baz_mean)**2, weights=cc))
            baz_max = baz_estimates.get(wave_type, baz_mean)
            
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
            
            # Mark estimated maximum
            ax.axvline(baz_max, color='black', linestyle='--', linewidth=2, 
                    label=f'Est: {baz_max:.1f}°')
            
            # Mark theoretical BAZ if available
            if event_info and 'backazimuth' in event_info:
                ax.axvline(event_info['backazimuth'], color='green', 
                        linestyle=':', linewidth=3, label=f"Theo: {event_info['backazimuth']:.1f}°")
                
                # Calculate deviation
                dev = abs(baz_max - event_info['backazimuth'])
                if dev > 180:
                    dev = 360 - dev
                
                # Add statistics text
                stats_text = (f"Mean: {baz_mean:.1f}° ± {baz_std:.1f}°\n"
                            f"Max: {baz_max:.1f}°\n"
                            f"Deviation: {dev:.1f}°")
            else:
                stats_text = (f"Mean: {baz_mean:.1f}° ± {baz_std:.1f}°\n"
                            f"Max: {baz_max:.1f}°")
            
            # Add statistics text box
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
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
        if hasattr(self, 'tbeg') and self.tbeg:
            title += f" - {self.tbeg.date} {str(self.tbeg.time)[:9]} UTC"
            title += f" | T = {results['parameters']['baz_win_sec']}s ({results['parameters']['baz_win_overlap']}%)"
            title += f" | CC > {results['parameters']['cc_threshold']}"
            title += f" | f = {self.fmin} - {self.fmax} Hz"
        fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        return fig

    def plot_waveforms(self, equal_scale=False, figsize=(12, 10), time_scale="seconds", ymin=None, ymax=None, ybounds=None):
        """
        Plot all waveforms in the stream in subplots above each other.
        
        Parameters
        ----------
        equal_scale : bool, optional
            If True, all subplots share the same y-axis scale. Default is False.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (12, 8).
        time_scale : str, optional
            Time scale for time labels. Default is "seconds".
        ymin : float or dict, optional
            Minimum y-axis value. Can be single float for all plots or dict with channel keys
        ymax : float or dict, optional
            Maximum y-axis value. Can be single float for all plots or dict with channel keys
        ybounds : dict, optional
            Minimum and maximum y-axis values as dict with channel keys.
        Returns
        -------
        fig, axs : matplotlib figure and axes objects
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import AutoMinorLocator
        
        if not hasattr(self, 'st') or self.st is None:
            raise ValueError("No stream data available in sixdegrees object")
        
        # Create figure and subplots
        n_traces = len(self.st)

        fig, axs = plt.subplots(n_traces, 1, figsize=figsize, sharex=True)

        if n_traces == 1:
            axs = [axs]
        
        # Get global min/max if equal_scale is True and no ymin/ymax provided
        if equal_scale and ymin is None and ymax is None:
            global_min = min(tr.data.min() for tr in self.st)
            global_max = max(tr.data.max() for tr in self.st)
            y_range = global_max - global_min
            y_margin = y_range * 0.1
            y_lims = (global_min - y_margin, global_max + y_margin)
        
        # Plot each trace
        for i, tr in enumerate(self.st):
            # Get time vector based on time_scale
            if time_scale == "seconds":
                times = tr.times(reftime=self.tbeg)
                xlabel = "Time (s)"
            elif time_scale == "minutes":
                times = tr.times(reftime=self.tbeg) / 60
                xlabel = "Time (min)"
            elif time_scale == "hours":
                times = tr.times(reftime=self.tbeg) / 3600
                xlabel = "Time (h)"
            elif time_scale == "days":
                times = tr.times(reftime=self.tbeg) / 86400
                xlabel = "Time (d)"
            
            # Plot waveform
            line = axs[i].plot(times, tr.data, 'k-', linewidth=0.5)[0]
            
            # Set y-limits
            if equal_scale and ymin is None and ymax is None:
                axs[i].set_ylim(y_lims)
            else:
                # Handle individual channel limits
                y_min_val = ymin[tr.stats.channel] if isinstance(ymin, dict) else ymin
                y_max_val = ymax[tr.stats.channel] if isinstance(ymax, dict) else ymax
                
                if y_min_val is None or y_max_val is None:
                    data_min, data_max = tr.data.min(), tr.data.max()
                    y_range = data_max - data_min
                    y_margin = y_range * 0.1
                    y_min_val = data_min - y_margin if y_min_val is None else y_min_val
                    y_max_val = data_max + y_margin if y_max_val is None else y_max_val
                
                if ybounds is not None and tr.stats.channel in ybounds:
                    if y_min_val < ybounds[tr.stats.channel][0]:
                        y_min_val = ybounds[tr.stats.channel][0]
                    if y_max_val > ybounds[tr.stats.channel][1]:
                        y_max_val = ybounds[tr.stats.channel][1]
                
                axs[i].set_ylim(y_min_val, y_max_val)
            
            # Format y-label with units
            if "J" in tr.stats.channel:
                    unit = self.runit
            else:
                unit = self.tunit

            axs[i].set_ylabel(unit, rotation=0, ha='right', va='center')
            
            # Add channel info as legend
            network = tr.stats.network
            station = tr.stats.station
            location = tr.stats.location
            channel = tr.stats.channel
            label = f"{network}.{station}.{location}.{channel}"
            axs[i].legend([line], [label], loc='upper right', frameon=False)
            
            # Add minor ticks
            axs[i].yaxis.set_minor_locator(AutoMinorLocator())
            axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            
            # Remove top, right, and bottom spines (except for last subplot)
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            if i < n_traces - 1:
                axs[i].spines['bottom'].set_visible(False)
                axs[i].xaxis.set_visible(False)
            
            # Add grid
            # axs[i].grid(True, which='major', alpha=0.3)
            # axs[i].grid(True, which='minor', alpha=0.1)
        
        # Set x-label on bottom subplot
        axs[-1].set_xlabel(xlabel)
        
        # Add title with time period and filter info if available
        title_parts = []
        
        # Add time period
        start_time = min(tr.stats.starttime for tr in self.st)
        end_time = max(tr.stats.endtime for tr in self.st)
        time_period = f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        title_parts.append(time_period)
        
        # Add filter info if available
        filter_info = ""
        if hasattr(self, 'fmin') and hasattr(self, 'fmax'):
            if self.fmin is not None and self.fmax is not None:
                filter_info = f"Filter: {self.fmin:.3f} - {self.fmax:.3f} Hz"
            elif self.fmin is not None:
                filter_info = f"Filter: > {self.fmin:.3f} Hz"
            elif self.fmax is not None:
                filter_info = f"Filter: < {self.fmax:.3f} Hz"
            title_parts.append(filter_info)
        
        if title_parts:
            fig.suptitle(' | '.join(title_parts), y=0.99)
        
        # Adjust layout
        plt.tight_layout()
        plt.show();

        return fig