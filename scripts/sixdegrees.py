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
import os
import matplotlib.pyplot as plt
import numpy as np
import os.path  # Add explicit import for os.path

from typing import Dict, List, Tuple, Union, Optional, Any
from obspy import UTCDateTime, Stream, Inventory
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

        # predefine theoretical baz
        self.baz_theo = None
        # define data source (local SDS or online FDSN)
        if 'data_source' in conf.keys():
            self.data_source = conf['data_source']  # 'sds' or 'fdsn'
        else:
            self.data_source = 'fdsn'  # default to local SDS archive

        # define startime
        if 'tbeg' in conf.keys():
            self.tbeg = UTCDateTime(conf['tbeg'])
        else:
            print("-> no starttime given!")

        # define endtime
        if 'tend' in conf.keys():
            self.tend = UTCDateTime(conf['tend'])
        else:
            print("-> no starttime given!")

        # set verbose (print information)
        if 'verbose' in conf.keys():
            self.verbose = conf['verbose']
        else:
            self.verbose = False

        # seed id of stream
        if 'seed' in conf.keys():
            self.net, self.sta, self.loc, self.cha = conf['seed'].split('.')
        else:
            self.net, self.sta, self.loc, self.cha = "XX.XXXX..".split('.')

        # seed id of rotation stream
        if 'rot_seed' in conf.keys():
            self.rot_seed = conf['rot_seed']
        else:
            print("-> no rotation seed id given!")

        # seed id of translation stream
        if 'tra_seed' in conf.keys():
            self.tra_seed = conf['tra_seed']
        else:
            print("-> no translation seed id given!")

        # station coordinates
        if 'station_lon' in conf.keys() and 'station_lat' in conf.keys():
            self.station_longitude = conf['station_lon']
            self.station_latitude = conf['station_lat']
        else:
            self.station_longitude = None
            self.station_latitude = None

        # define project name
        if 'project' in conf.keys():
            self.project = conf['project']
        else:
            self.project = "test"

        # define working directory
        if 'workdir' in conf.keys():
            self.workdir = os.path.normpath(conf['workdir'])
        else:
            self.workdir = os.path.normpath("./")

        # define directory for output data
        if 'path_to_data_out' in conf.keys():
            self.path_to_data_out = os.path.normpath(conf['path_to_data_out'])
        else:
            self.path_to_data_out = os.path.normpath(os.path.join(self.workdir, "output"))

        # define directory for figure output
        if 'path_to_figs_out' in conf.keys():
            self.path_to_figs_out = os.path.normpath(conf['path_to_figs_out'])
        else:
            self.path_to_figs_out = os.path.normpath(os.path.join(self.workdir, "figures"))

        if self.data_source == 'sds':
            # path to SDS file structure for rotation data
            if 'path_to_sds_rot' in conf.keys():
                self.rot_sds = os.path.normpath(conf['path_to_sds_rot'])
            else:
                print("-> no path to SDS file structure for rotation data given!")

            # path to SDS file structure for translaton data
            if 'path_to_sds_tra' in conf.keys():
                self.tra_sds = os.path.normpath(conf['path_to_sds_tra'])
            else:
                print("-> no path to SDS file structure for translaton data given!")

            # path to translation station inventory
            if 'path_to_inv_tra' in conf.keys():
                self.tra_inv = os.path.normpath(conf['path_to_inv_tra'])
            else:
                print("-> no path to translation station inventory given!")
                self.tra_inv = None

            # path to rotation station inventory
            if 'path_to_inv_rot' in conf.keys():
                self.rot_inv = os.path.normpath(conf['path_to_inv_rot'])
            else:
                print("-> no path to rotation station inventory given!")
                self.rot_inv = None

        elif self.data_source == 'fdsn':
            # path to FDSN client for rotation data
            if 'fdsn_client_rot' in conf.keys():
                self.fdsn_client_rot = conf['fdsn_client_rot']
            else:
                print("-> no FDSN client for rotation data given!")

            # path to FDSN client for translation data
            if 'fdsn_client_tra' in conf.keys():
                self.fdsn_client_tra = conf['fdsn_client_tra']
            else:
                print("-> no FDSN client for translation data given!")

        # path to mseed file if using direct file input
        if 'path_to_mseed_file' in conf.keys():
            self.mseed_file = os.path.normpath(conf['path_to_mseed_file'])
        else:
            self.mseed_file = False

        # rotate_zne
        if 'rotate_zne' in conf.keys():
            self.rotate_zne = conf['rotate_zne']
        else:
            self.rotate_zne = False

        # output type for remove response
        self.tra_output = "ACC"

        # polarity dictionary
        self.pol_dict = None

        # Add new attributes
        self.rot_components = None  # Components to rotate from (e.g., 'ZUV')
        self.rot_target = 'ZNE'     # Target components (e.g., 'ZNE')

        # Add ROMY rotation options
        self.use_romy_zne = conf.get('use_romy_zne', False)
        self.keep_z = conf.get('keep_z', True)

    # ____________________________________________________

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
                return self.st0.select(channel="*J*")
            else:
                return self.st.select(channel="*J*")
        elif stream_type == "translation":
            if raw:
                return self.st0.select(channel="*H*")
            else:
                return self.st.select(channel="*H*")
        elif stream_type == "all":
            if raw:
                return self.st0
            else:
                return self.st
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

        #initialize FDSN client for chosen catalog
        client = Client(self.base_catalog)

        try:
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
                print(f"Distance: {self.event_info['distance_deg']:.1f}°")
                print(f"Backazimuth: {self.event_info['backazimuth']:.1f}°")
            
            return self.event_info
            
        except Exception as e:
            print(f"Error getting event information from {self.base_catalog}:")
            if self.verbose:
                print(e)
            return {}

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

        >>> flower, fupper, fcenter = __get_octave_bands(f_min, f_max, fband_type="octave", plot=False)
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
        rot_r, rot_t = rotate_ne_rt(rot.select(channel="*N")[0].data,
                                    rot.select(channel="*E")[0].data,
                                    baz)
        tra_r, tra_t = rotate_ne_rt(tra.select(channel="*N")[0].data,
                                    tra.select(channel="*E")[0].data,
                                    baz)

        tra_z = tra.select(channel="*Z")[0].data
        rot_z = rot.select(channel="*Z")[0].data

        # Compute cross-correlation
        cc = correlate(tra_z, rot_t, len(rot_t), normalize=normalize)
        lag_samples_h, cc_max_h = xcorr_max(cc)
        
        # Convert to time
        lag_time_h = lag_samples_h / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-T & ACC-Z:  lag_time: {lag_time_h}, lag_samples: {lag_samples_h}, cc_max: {cc_max_h}")

        # Compute cross-correlation
        cc = correlate(tra_t, rot_z, len(rot_z), normalize=normalize)
        lag_samples_z, cc_max_z = xcorr_max(cc)
        
        # Convert to time
        lag_time_z = lag_samples_z / self.get_stream("rotation")[0].stats.sampling_rate

        print(f"ROT-Z & ACC-T:  lag_time: {lag_time_z}, lag_samples: {lag_samples_z}, cc_max: {cc_max_z}")
       
        # shift rotataion waveforms
        if correct:
            for tr in rot:
                if tr.stats.channel.endswith("Z"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_z
            
                if tr.stats.channel.endswith("N") or tr.stats.channel.endswith("E"):
                    # tr.data = roll(tr.data, lag_samples)
                    tr.stats.starttime = tr.stats.starttime + lag_time_h
              
        # update and trim raw stream
        if correct:
            # reassign raw stream
            self.st0 = rot + tra
            # trim stream
            self.st0.trim(self.tbeg, self.tend)
            # avoid overlaps
            self.st0 = self.st0.merge(method=1)
            # reassign stream
            self.st = self.st0.copy()

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

    def load_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime]):
        '''
        Load data for translation and rotaion as obspy stream

        @param t1: starttime
        @param t2: endtime
        @type t1: str or UTCDateTime
        @type t2: str or UTCDateTime
        '''

        from obspy import Stream, UTCDateTime, read
        from obspy.clients.fdsn import Client as FDSNClient

        st0 = Stream()

        t1, t2 = UTCDateTime(t1), UTCDateTime(t2)

        if len(self.tra_seed) > 0:
            # add to stream
            st0 += self._load_translation_data(t1, t2)

        if len(self.rot_seed) > 0:
            # add to stream
            st0 += self._load_rotation_data(t1, t2)

        # check if stream has correct length
        if len(st0) < (len(self.tra_seed) + len(self.rot_seed)):
            print(f" -> missing stream data")

       # check if merging is required
        if len(st0) > (len(self.tra_seed) + len(self.rot_seed)):
            st0 = st0.merge(method=1, fill_value=0)

        # Update stream IDs
        for tr in st0:
            tr.stats.network = self.net
            tr.stats.station = self.sta
            tr.stats.location = self.loc

        # assign stream to object
        self.st = st0

        # assign stream as raw stream
        self.st0 = st0

        # Check if all traces have the same sampling rate and add as attribute
        if len(self.st) > 0:
            sampling_rates = set(tr.stats.sampling_rate for tr in self.st)
            if len(sampling_rates) == 1:
                self.sampling_rate = sampling_rates.pop()
            else:
                print(" -> Warning: Not all traces have the same sampling rate!")
                if self.verbose:
                    print(f"Sampling rates found: {sampling_rates}")


    def _load_translation_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime]):
        '''
        Load translation data
        '''

        # Load translation data
        tra = Stream()
        client = FDSNClient(self.fdsn_client_tra)

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
                    if self.verbose:
                        print(f"-> fetching {tseed} data from mseed file")
                    # read directly from mseed file
                    if self.mseed_file is None:
                        raise ValueError("No mseed file path provided")
                    tra0 = read(self.mseed_file)
                    tra += tra0.select(channel=cha)
                    tra = tra.trim(t1-1, t2+1)
                
                else:
                    raise ValueError(f"Unknown data source: {self.data_source}. Use 'sds' or 'fdsn'.")

            except Exception as e:
                print(f" -> loading translational data failed!")
                if self.verbose:
                    print(e)

        # Process translation data
        try:
            # detrend
            tra = tra.detrend("linear")

            # remove response
            if self.tra_inv is not None:
                if self.verbose:
                    print(f"-> translation inventory provided: {self.tra_inv}")

                if self.data_source.lower() == 'mseed_file':
                    if self.verbose:
                        print(f"-> skipping response removal for mseed file")

                elif self.data_source.lower() == 'sds':
                    try:
                        # remove response
                        tra = tra.remove_response(self.tra_inv, output=self.tra_output)
                        if self.verbose:
                            print("-> successfully removed response")
                    except Exception as e:
                        print(f"-> warning: failed to remove response: {str(e)}")

            elif self.data_source.lower() == 'fdsn':
                try:
                    # get inventory from FDSN
                    self.tra_inv = client.get_stations(network=net, station=sta,
                                                        starttime=t1, endtime=t2,
                                                        level="response",
                                                        )
                except Exception as e:
                    print(f"-> warning: failed to get inventory: {str(e)}")

                try:
                    # remove response
                    tra = tra.remove_response(self.tra_inv, output=self.tra_output)
                    if self.verbose:
                        print("-> successfully removed response")

                except Exception as e:
                    print(f"-> warning: failed to remove response: {str(e)}")

            # rotate components
            if not self.data_source.lower() == 'mseed_file':

                # rotate to ZNE
                if self.rotate_zne:
                    if self.verbose:
                        print(f"-> rotating translational data to ZNE")
                    tra = tra.rotate(method="->ZNE", inventory=self.tra_inv)

                # Get station coordinates
                if self.station_latitude is None and self.station_longitude is None:
                    try:
                        coords = self.tra_inv.get_coordinates(self.tra_seed[0])
                        self.station_latitude = coords['latitude']
                        self.station_longitude = coords['longitude']
                    except Exception as e:
                        print(f"-> warning: failed to get station coordinates: {str(e)}")

        except Exception as e:
            print(f" -> removing response failed!")
            if self.verbose:
                print(e)

        if self.verbose:
            print(tra)
        
        return tra

    def _load_rotation_data(self, t1: Union[str, UTCDateTime], t2: Union[str, UTCDateTime]):
        '''
        Load rotation data
        '''
        # Load rotation data
        rot = Stream()
        client = FDSNClient(self.fdsn_client_rot)

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
                    # read directly from mseed file
                    if self.verbose:
                        print(f"-> fetching {rseed} data from mseed file")
                    if self.mseed_file is None:
                        raise ValueError("No mseed file path provided")
                    rot0 = read(self.mseed_file)
                    rot += rot0.select(channel=cha)
                    rot = rot.trim(t1-1, t2+1)
                else:
                    raise ValueError(f"Unknown data source: {self.data_source}. Use 'sds' or 'fdsn'.")
            except Exception as e:
                print(f" -> loading rotational data failed!")
                if self.verbose:
                    print(e)

        # Process rotation data
        try:
            # detrend
            rot = rot.detrend("linear")

            # remove response
            if self.rot_inv is not None:
                if self.verbose:
                    print(f"-> rotation inventory provided: {self.rot_inv}")

                if self.data_source.lower() == 'mseed_file':
                    if self.verbose:
                        print(f"-> skipping sensitivity removal for mseed file")

                elif self.data_source.lower() == 'sds':
                    try:
                        # remove sensitivity
                        rot = rot.remove_sensitivity(self.rot_inv)
                        if self.verbose:
                            print("-> successfully removed sensitivity")
                    except Exception as e:
                        print(f"-> warning: failed to remove sensitivity: {str(e)}")

            elif self.data_source.lower() == 'fdsn':
                try:
                    # get inventory from FDSN
                    self.rot_inv = client.get_stations(network=net, station=sta,
                                                        starttime=t1-1, endtime=t2+1,
                                                        level="response")
                except Exception as e:
                    print(f"-> warning: failed to get inventory: {str(e)}")

                try:
                    # remove sensitivity
                    rot = rot.remove_sensitivity(self.rot_inv)
                    if self.verbose:
                        print("-> successfully removed sensitivity")

                except Exception as e:
                    print(f"-> warning: failed to remove sensitivity: {str(e)}")

        except Exception as e:
            print(f"-> error processing rotation data: {str(e)}")

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
                        # rotate
                        print("rotating")
                        rot = self.rotate_romy_zne(
                            rot, 
                            self.rot_inv,
                            use_components=components,
                            keep_z=self.keep_z
                        )
                    except Exception as e:
                        print(f"-> warning: ROMY ZNE rotation failed: {str(e)}")
            else:
                if self.verbose:
                    print(f"-> rotating rotational data to ZNE")
                # general rotation
                rot = rot.rotate(method="->ZNE", inventory=self.rot_inv)

        # assign station coordinates
        if self.station_latitude is None and self.station_longitude is None:
            coords = self.rot_inv.get_coordinates(self.rot_seed[0])
            self.station_latitude = coords['latitude']
            self.station_longitude = coords['longitude']

        if self.verbose:
            print(rot)
        
        return rot

    def filter_data(self, fmin: float=0.1, fmax: float=0.5, output: bool=False):

        # reset stream to raw stream
        self.st = self.st0.copy()

        # set fmin and fmax
        self.fmin = fmin
        self.fmax = fmax

        # detrend and filter
        self.st = self.st.detrend("linear")

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

        def __get_size(st0: Stream) -> List[int]:
            return [tr.stats.npts for tr in st0]

        # get size of traces
        n_samples = __get_size(self.st)

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
                    print(f"  -> adjusted: {__get_size(self.st)}")

                    if set_interpolate:
                        _times = arange(0, min(__get_size(self.st)), self.st[0].stats.delta)
                        for tr in self.st:
                            tr.data = interp(_times, tr.times(reftime=_tbeg), tr.data)
            else:
                # adjust for difference of one sample
                for tr in self.st:
                    tr.data = tr.data[:min(n_samples)]
                print(f"  -> adjusted: {__get_size(self.st)}")

    def polarity_stream(self, pol_dict: Dict={}, raw: bool=False):
        '''
        Modify polarity of data
        '''
        for tr in self.st:
            if tr.stats.channel[1:] in pol_dict.keys():
                tr.data = tr.data * pol_dict[tr.stats.channel[1:]]
        if raw:
            for tr in self.st0:
                if tr.stats.channel[1:] in pol_dict.keys():
                    tr.data = tr.data * pol_dict[tr.stats.channel[1:]]
        self.pol_dict = pol_dict
        self.pol_applied = True

    def store_as_pickle(self, obj: object, name: str):

        import os, pickle

        ofile = open(name+".pkl", 'wb')
        pickle.dump(obj, ofile)

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

        def filter_stream(stream: Stream, fmin: float, fmax: float) -> Stream:
            stream = stream.detrend('linear')
            stream = stream.taper(max_percentage=0.01)
            stream.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
            stream = stream.detrend('linear')
            return stream

        # Default frequency bands if not provided
        if fbands is None:
            fbands = {
                'fmin': 0.01,
                'fmax': 1.0,
                'octave_fraction': 3
            }

        # Generate frequency bands
        flower, fupper, fcenter = self.get_octave_bands(
            fmin=fbands['fmin'],
            fmax=fbands['fmax'],
            faction_of_octave=fbands['octave_fraction']
        )
        
        # Get streams and sampling rate
        rot = self.get_stream("rotation", raw=True).copy()
        acc = self.get_stream("translation", raw=True).copy()
        df = self.sampling_rate
        n_samples = len(rot[0].data)
        
        # Generate backazimuth values
        baz_values = np.arange(0, 360, baz_step)
        
        # Initialize results dictionary for each frequency band
        results_by_freq = []
        
        # Loop through frequency bands first
        for freq_idx, (fl, fu, fc) in enumerate(zip(flower, fupper, fcenter)):
            # Calculate window size based on frequency
            f_bandwidht = fu - fl
            win_time_s_freq = max(twin_min, bandwidth_factor/f_bandwidht)
            win_samples = int(win_time_s_freq * df)
            step = int(win_samples * (1 - overlap))
            n_windows = (n_samples - win_samples) // step + 1
            
            if n_windows < 1:
                continue
                
            # Filter data for this frequency band
            rot_filt = rot.copy()
            acc_filt = acc.copy()
            
            rot_filt = filter_stream(rot_filt, fl, fu)
            acc_filt = filter_stream(acc_filt, fl, fu)
            
            # Initialize arrays for this frequency
            times = np.zeros(n_windows)
            baz_optimal = np.zeros(n_windows)
            cc_optimal = np.zeros(n_windows)
            cc_matrix = np.zeros((n_windows, len(baz_values)))
            
            # Loop through windows
            for win_idx in range(n_windows):
                i1 = win_idx * step
                i2 = i1 + win_samples
                times[win_idx] = rot[0].times()[i1]
                
                # Get vertical components
                rot_z = rot_filt.select(channel='*Z')[0].data[i1:i2]
                acc_z = acc_filt.select(channel='*Z')[0].data[i1:i2]
                
                max_cc = -1
                
                # Loop through backazimuths
                for baz_idx, baz in enumerate(baz_values):
                    # Rotate components
                    rot_r, rot_t = rotate_ne_rt(
                        rot_filt.select(channel='*N')[0].data[i1:i2],
                        rot_filt.select(channel='*E')[0].data[i1:i2],
                        baz
                    )
                    acc_r, acc_t = rotate_ne_rt(
                        acc_filt.select(channel='*N')[0].data[i1:i2],
                        acc_filt.select(channel='*E')[0].data[i1:i2],
                        baz
                    )
                    
                    # Compute correlation based on wave type
                    if wave_type.lower() == 'love':
                        cc = xcorr_max(correlate(rot_z, acc_t, 0))[1]
                    elif wave_type.lower() == 'rayleigh':
                        cc = xcorr_max(correlate(rot_t, acc_z, 0))[1]
                    else:
                        raise ValueError(f"Invalid wave type: {wave_type}")
                    
                    cc_matrix[win_idx, baz_idx] = cc
                    
                    # Update optimal parameters if better correlation found
                    if cc > max_cc:
                        max_cc = cc
                        baz_optimal[win_idx] = baz
                        cc_optimal[win_idx] = cc
            
            # Store results for this frequency band
            results_by_freq.append({
                'frequency': {
                    'min': fl,
                    'max': fu,
                    'center': fc
                },
                'times': times,
                'backazimuth': baz_optimal,
                'cc_matrix': cc_matrix,
                'cc_optimal': cc_optimal,
                'window_samples': win_samples,
                'step': step
            })
        
        # Find optimal frequency band for each time window by comparing cc values
        n_total_windows = max(len(r['times']) for r in results_by_freq)
        final_times = np.zeros(n_total_windows)
        final_baz = np.zeros(n_total_windows)
        final_fmin = np.zeros(n_total_windows)
        final_fmax = np.zeros(n_total_windows)
        final_fcenter = np.zeros(n_total_windows)
        final_cc = np.zeros(n_total_windows)
        final_velocities = np.zeros(n_total_windows)
        
        # For each window, find the frequency band with highest cc
        for win_idx in range(n_total_windows):
            max_cc = -1
            for freq_result in results_by_freq:
                if win_idx < len(freq_result['times']):
                    cc = freq_result['cc_optimal'][win_idx]
                    if cc > max_cc:
                        max_cc = cc
                        final_times[win_idx] = freq_result['times'][win_idx]
                        final_baz[win_idx] = freq_result['backazimuth'][win_idx]
                        final_fmin[win_idx] = freq_result['frequency']['min']
                        final_fmax[win_idx] = freq_result['frequency']['max']
                        final_fcenter[win_idx] = freq_result['frequency']['center']
                        final_cc[win_idx] = cc
        
        # Compute velocities using optimal parameters
        for i in range(n_total_windows):
            # Filter and rotate with optimal parameters
            rot_opt = rot.copy()
            acc_opt = acc.copy()
            
            rot_opt = filter_stream(rot_opt, final_fmin[i], final_fmax[i])
            acc_opt = filter_stream(acc_opt, final_fmin[i], final_fmax[i])

            # Calculate window indices based on optimal frequency
            win_samples = int(max(twin_min, 1/final_fcenter[i]) * df)
            step = int(win_samples * (1 - overlap))
            i1 = i * step
            i2 = i1 + win_samples
            
            if i2 > n_samples:
                final_velocities[i] = np.nan
                continue
            
            # Get components and rotate
            rot_z = rot_opt.select(channel='*Z')[0].data[i1:i2]
            acc_z = acc_opt.select(channel='*Z')[0].data[i1:i2]
            
            rot_r, rot_t = rotate_ne_rt(
                rot_opt.select(channel='*N')[0].data[i1:i2],
                rot_opt.select(channel='*E')[0].data[i1:i2],
                final_baz[i]
            )
            acc_r, acc_t = rotate_ne_rt(
                acc_opt.select(channel='*N')[0].data[i1:i2],
                acc_opt.select(channel='*E')[0].data[i1:i2],
                final_baz[i]
            )
            
            # Compute velocity
            if wave_type.lower() == 'love':
                vel_result = self.compute_odr(
                    x_array=rot_z,
                    y_array=0.5*acc_t,
                    zero_intercept=True
                )
            elif wave_type.lower() == 'rayleigh':
                vel_result = self.compute_odr(
                    x_array=rot_t,
                    y_array=acc_z,
                    zero_intercept=True
                )
            
            final_velocities[i] = vel_result['slope']
        
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

    def compute_backazimuth(self, wave_type: str="love", baz_step: int=1, baz_win_sec: float=30.0, 
                           baz_win_sec_overlap: float=0.5, out: bool=False) -> Dict:
        """
        Estimate backazimuth for Love or Rayleigh waves
        
        Parameters:
        -----------
        wave_type : str
            Type of wave to analyze ('love' or 'rayleigh')
        baz_step : int
            Step size in degrees for backazimuth search (default: 1)
        baz_win_sec : float
            Length of backazimuth estimation windows in seconds (default: 30.0)
        baz_win_sec_overlap : float
            Overlap between windows as fraction (0-1) (default: 0.5)
        out : bool
            Return detailed output dictionary if True
            
        Returns:
        --------
        Dict : Backazimuth estimation results
        """
        from obspy.signal.rotate import rotate_ne_rt
        from obspy.signal.cross_correlation import correlate, xcorr_max
        from numpy import linspace, ones, array, nan, meshgrid, arange

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
        self.baz_win_sec_overlap = baz_win_sec_overlap

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
        overlap = baz_win_sec_overlap / baz_win_sec

        # Prepare time windows for loop
        n_windows = n_samples // (int(self.sampling_rate * baz_win_sec))

        # Prepare backazimuths for loop using integer step size
        backazimuths = linspace(0, 360 - self.baz_step, int(360 / self.baz_step))

        # Prepare data array
        corrbaz = ones([backazimuths.size, n_windows])*nan

        degrees = []
        windows = []

        bazs = ones(n_windows)*nan

        # _______________________________
        # backazimuth estimation with Love or Rayleigh waves
        # loop over backazimuth degrees
        for i_deg in range(0, len(backazimuths)):

            degrees.append(i_deg)

            # loop over time windows
            for i_win in range(0, n_windows):

                if i_deg == 0:
                    windows.append(i_win)

                # infer indices
                idx1 = int(self.sampling_rate * baz_win_sec * i_win)
                idx2 = int(self.sampling_rate * baz_win_sec * (i_win + 1))

                # add overlap
                if i_win > 0 and i_win < n_windows:
                    idx1 = int(idx1 - overlap * baz_win_sec * self.sampling_rate)
                    idx2 = int(idx2 + overlap * baz_win_sec * self.sampling_rate)

                # prepare traces according to selected wave type
                if wave_type.lower() == "love":

                    if self.verbose and i_deg == 0 and i_win == 0:
                        print(f"\nusing {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    R, T = rotate_ne_rt(ACC.select(channel='*N')[0].data,
                                        ACC.select(channel='*E')[0].data,
                                        backazimuths[i_deg]
                                       )

                    # compute correlation for backazimuth
                    ccorr = correlate(ROT.select(channel="*Z")[0][idx1:idx2],
                                      T[idx1:idx2],
                                      0, demean=True, normalize='naive', method='fft'
                                     )

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)

                    if xshift != 0:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "rayleigh":

                    if self.verbose and i_deg == 0 and i_win == 0:
                        print(f"\nusing {wave_type} waves for backazimuth estimation ...")

                    # rotate NE to RT
                    R, T = rotate_ne_rt(ROT.select(channel='*N')[0].data,
                                        ROT.select(channel='*E')[0].data,
                                        backazimuths[i_deg]
                                       )

                    # compute correlation for backazimuth
                    ccorr = correlate(ACC.select(channel="*Z")[0][idx1:idx2],
                                      T[idx1:idx2],
                                      0, demean=True, normalize='naive', method='fft'
                                     )

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)

                    if xshift != 0:
                        print(f" -> maximal cc not a shift=0: shift={xshift} | cc={cc_max}")

                elif wave_type.lower() == "tangent":

                    if self.verbose and i_deg == 0 and i_win == 0:
                        print(f"\nusing {wave_type} for backazimuth estimation ...")

                    # no grid search, no degrees loop required
                    if i_deg > 0:
                        continue

                    N = len(ROT[0].data[idx1:idx2])

                    # prepare data
                    dat = (zeros((N, 2)))
                    dat[:, 0] = ROT.select(channel='*E')[0].data[idx1:idx2]
                    dat[:, 1] = ROT.select(channel='*N')[0].data[idx1:idx2]

                    # compute covariance
                    covar = cov(dat, rowvar=False)

                    # get dominant eigenvector
                    Cprime, Q = eigh(covar, UPLO='U')

                    # sorting
                    loc = argsort(abs(Cprime))[::-1]

                    # formating
                    Q = Q[:, loc]

                    # get backazimuth using tangent of eigenvectors
                    baz0 = -arctan((Q[1, 0]/Q[0, 0]))*180/pi

                    # if negative due to tangent, then add 180 degrees
                    if baz0 <= 0:
                        baz0 += 180

                    # remove 180° ambiguity
                    R, T = rotate_ne_rt(ROT.select(channel='*N')[0].data,
                                        ROT.select(channel='*E')[0].data,
                                        baz0
                                       )

                    # correlatet with acceleration
                    ccorr = correlate(ACC.select(channel="*Z")[0][idx1:idx2],
                                      T[idx1:idx2],
                                      0, demean=True, normalize='naive', method='fft'
                                     )

                    # get maximum correlation
                    xshift, cc_max = xcorr_max(ccorr)

                    # if correlation positive add 180 degrees
                    if (cc_max > 0):
                        baz0 += 180

                    cc_max = abs(cc_max)

                    # ## add new values to array
                    # if abs(cc_max) > cc_thres:
                    #     baz[j] = baz0
                    #     ccor[j] = abs(corr_baz)

                else:
                    print(f" -> unknown wave type: {wave_type}!")

                corrbaz[i_deg, i_win] = cc_max

                if wave_type.lower() == "tangent":
                    bazs[i_win] = baz0

        # extract maxima
        if wave_type.lower() == "tangent":
            maxbaz = bazs
            maxcorr = corrbaz[0, :]
        else:
            maxbaz = array([backazimuths[corrbaz[:, l1].argmax()] for l1 in range(0, n_windows)])
            maxcorr = array([max(corrbaz[:, l1]) for l1 in range(0, n_windows)])

        # create mesh grid
        t_win = arange(0, baz_win_sec*n_windows+baz_win_sec, baz_win_sec)
        t_win_center = t_win[:-1]+baz_win_sec/2
        grid = meshgrid(t_win, backazimuths)

        # add one element for axes
        windows.append(windows[-1]+1)
        degrees.append(degrees[-1]+self.baz_step)

        # add results to object
        if wave_type.lower() == "love":
            self.baz_grid_love = corrbaz
            self.baz_degrees_love = degrees
            self.baz_windows_love = windows
            self.baz_corr_love = maxcorr
            self.baz_max_love = maxbaz
            self.baz_times_love = t_win_center

        elif wave_type.lower() == "rayleigh":
            self.baz_grid_rayleigh = corrbaz
            self.baz_degrees_rayleigh = degrees
            self.baz_windows_ralyeigh = windows
            self.baz_corr_rayleigh = maxcorr
            self.baz_max_rayleigh = maxbaz
            self.baz_times_rayleigh = t_win_center

        elif wave_type.lower() == "tangent":
            self.baz_grid_tangent = corrbaz
            self.baz_degrees_tangent = degrees
            self.baz_windows_tangent = windows
            self.baz_corr_tangent = maxcorr
            self.baz_max_tangent = maxbaz
            self.baz_times_tangent = t_win_center

        if out:
            # _______________________________
            # prepare output
            output = {}

            output['baz_mesh'] = grid
            output['baz_corr'] = corrbaz
            output['acc'] = ACC
            output['rot'] = ROT
            output['cc_max_t'] = t_win_center
            output['cc_max_y'] = maxbaz
            output['cc_max'] = maxcorr

            return output

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
            if abs(cc) >= cc_threshold:
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
                baz_win_sec_overlap=overlap,
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
            if abs(cc) >= cc_threshold:
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

    def compare_backazimuth_methods(self, Twin: float, Toverlap: float, baz_theo: float=None, 
                                  baz_theo_margin: float=10, baz_step: int=1, minors: bool=True,
                                  cc_threshold: float=0, plot: bool=True, output: bool=False,
                                  invert_rot_z: bool=False, invert_acc_z: bool=False) -> Tuple[plt.Figure, Dict]:
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
            wave_results = self.compute_backazimuth(
                wave_type=wave_type,
                baz_step=baz_step,
                baz_win_sec=Twin,
                baz_win_sec_overlap=Toverlap,
                out=True
            )
            
            # Filter out low correlation coefficients
            mask = wave_results['cc_max'] >= cc_threshold
            times_filtered = wave_results['cc_max_t'][mask]
            baz_filtered = wave_results['cc_max_y'][mask]
            cc_filtered = wave_results['cc_max'][mask]
            
            # Store filtered results
            results_dict[wave_type] = {
                'time': times_filtered,
                'backazimuth': baz_filtered,
                'correlation': cc_filtered
            }
            
            if plot:
                # Plot results for each wave type
                if wave_type == 'love':
                    scatter = ax1.scatter(times_filtered, baz_filtered,
                                        c=cc_filtered, cmap=cmap,
                                        s=70, alpha=0.7, vmin=0, vmax=1,
                                        edgecolors="k", lw=1, zorder=3)
                elif wave_type == 'rayleigh':
                    ax2.scatter(times_filtered, baz_filtered,
                              c=cc_filtered, cmap=cmap,
                              s=70, alpha=0.7, vmin=0, vmax=1,
                              edgecolors="k", lw=1, zorder=3)
                else:  # tangent
                    ax3.scatter(times_filtered, baz_filtered,
                              c=cc_filtered, cmap=cmap,
                              s=70, alpha=0.7, vmin=0, vmax=1,
                              edgecolors="k", lw=1, zorder=3)
                
                # Compute and plot histogram
                hist = histogram(baz_filtered,
                               bins=len(angles1)-1,
                               range=[min(angles1), max(angles1)],
                               weights=cc_filtered,
                               density=True)
                
                # Compute KDE
                if len(baz_filtered) > 1:  # Need at least 2 points for KDE
                    results_dict[wave_type]['kde'] = sts.gaussian_kde(baz_filtered, weights=cc_filtered)
                    baz_estimated[wave_type] = angles2[argmax(results_dict[wave_type]['kde'].pdf(angles2))]
                else:
                    baz_estimated[wave_type] = nan
                
                print(f"\nEstimated BAZ {label} = {baz_estimated[wave_type]:.0f}° (CC ≥ {cc_threshold})")
        
        if plot:

            # add histograms and KDEs to subplots
            for ax, ax_hist, label in [(ax1, ax11, "love"), (ax2, ax22, "rayleigh"), (ax3, ax33, "tangent")]:
                ax_hist.hist(results_dict[label]['backazimuth'], 
                            bins=len(angles1)-1,
                            range=[min(angles1), max(angles1)],
                            weights=results_dict[label]['correlation'],
                            orientation="horizontal", density=True, color="grey")
                ax_hist.plot(results_dict[label]['kde'].pdf(angles2), angles2, color='k', lw=3)
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
            ax1.set_title(f"Love Wave BAZ (estimated = {baz_estimated['love']:.0f}°)", fontsize=font)
            ax2.set_title(f"Rayleigh Wave BAZ (estimated = {baz_estimated['rayleigh']:.0f}°)", fontsize=font)
            ax3.set_title(f"Tangent BAZ (estimated = {baz_estimated['tangent']:.0f}°)", fontsize=font)
            ax3.set_xlabel("Time (s)", fontsize=font)
            
            for ax in [ax1, ax2, ax3]:
                ax.set_ylabel("BAZ (°)", fontsize=font)
            
            # Add colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            plt.colorbar(scatter, cax=cbar_ax, label='CC coefficient')
            
            # Add title
            title = f"{rot[0].stats.starttime.date} {str(rot[0].stats.starttime.time).split('.')[0]} UTC"
            title += f" | {self.fmin}-{self.fmax} Hz | T = {Twin} s | {Toverlap*100}% overlap"
            if baz_theo is not None:
                title += f" | expected BAz = {baz_theo:.0f}°"
            if cc_threshold > 0:
                title += f" | CC ≥ {cc_threshold}"
            fig.suptitle(title, fontsize=font+2, y=0.99)
            
            # plt.tight_layout()
            plt.show()
        
        # Store estimated BAZ
        self.baz_estimated = baz_estimated
        
        # Prepare output
        if output:
            if plot:
                return fig, results_dict
            else:
                return results_dict

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

        >>> frequencies, spectrum, phase = __fft(signal_in, dt ,window=None,normalize=None)
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
    def plot_waveform_cc(rot0: Stream, acc0: Stream, baz: float, fmin: float, fmax: float, wave_type: str="both",
                         pol_dict: Union[None, Dict]=None, distance: Union[None, float]=None, 
                         twin_sec: int=5, twin_overlap: float=0.5) -> plt.Figure:

        from obspy.signal.cross_correlation import correlate
        from obspy.signal.rotate import rotate_ne_rt
        from numpy import linspace, ones
        import matplotlib.pyplot as plt

        def __cross_correlation_windows(arr1: array, arr2: array, dt: float, Twin: float, overlap: float=0, lag: int=0, demean: bool=True, plot: bool=False) -> Tuple[array, array]:

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

        rot = rot0.copy()
        acc = acc0.copy()

        # get sampling rate
        dt = rot[0].stats.delta

        # define polarity
        pol = {"HZ":1,"HN":1,"HE":1,"HR":1,"HT":1,
               "JZ":1,"JN":1,"JE":1,"JR":1,"JT":1,
              }
        # update polarity dictionary
        if pol_dict is not None:
            for k in pol_dict.keys():
                pol[k] = pol_dict[k]

        # Change number of rows based on wave type
        if wave_type == "both":
            Nrow, Ncol = 2, 1
            fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5*Nrow), sharex=True)
            ax = axes  # axes is already an array for multiple subplots
        else:
            Nrow, Ncol = 1, 1
            fig, axes = plt.subplots(Nrow, Ncol, figsize=(15, 5), sharex=True)
            ax = [axes]  # wrap single axes in list for consistent indexing
        
        # define scaling factors
        acc_scaling, acc_unit = 1e3, f"mm/s$^2$"
        rot_scaling, rot_unit = 1e6, f"$\mu$rad/s"

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
            rot0, acc0, rot0_lbl, acc0_lbl = pol['JZ']*rot_z, pol['HT']*acc_t, f"{pol['JZ']}x ROT-Z", f"{pol['HT']}x ACC-T"
            # calculate cross-correlation
            tt0, cc0 = __cross_correlation_windows(rot0, acc0, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
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
            rot1, acc1, rot1_lbl, acc1_lbl = pol['JT']*rot_t, pol['HZ']*acc_z, f"{pol['JT']}x ROT-T", f"{pol['HZ']}x ACC-Z"
            # calculate cross-correlation
            tt1, cc1 = __cross_correlation_windows(rot1, acc1, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)
            cc.append(cc1)
            cc_all.append(max(correlate(rot1, acc1, 0, demean=True, normalize='naive', method='fft')))

        # rot2, acc2, rot2_lbl, acc2_lbl = pol['JZ']*rot_z, pol['HR']*acc_r, f"{pol['JZ']}x ROT-Z", f"{pol['HR']}x ACC-R"
        # tt2, cc2 = __cross_correlation_windows(rot2, acc2, dt, twin_sec, overlap=twin_overlap, lag=0, demean=True)

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
        for i in range(Nrow):
            ax[i].legend(loc=1, ncols=4)
            ax[i].grid(which="both", alpha=0.5)
            ax[i].set_ylabel(f"$\Omega$ ({rot_unit})", fontsize=font)
            ax[i].text(0.05, 0.9, f"CC={cc_all[i]:.2f}", ha='left', va='top', transform=ax[i].transAxes, fontsize=font-1)

        for _ax in twinaxs:
            _ax.legend(loc=4)
            _ax.set_ylabel(f"$a$ ({acc_unit})", fontsize=font)

        # Add colorbar
        cax = ax[Nrow-1].inset_axes([0.8, -0.35, 0.2, 0.1], transform=ax[Nrow-1].transAxes)
        
        # Create a ScalarMappable for the colorbar
        norm = plt.Normalize(-1, 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax, location="bottom", orientation="horizontal")
        cbar.set_label("Cross-correlation", fontsize=font-1, loc="left", labelpad=-43, color="k")
        
        # Set limits for scatter plots
        for cm in cms:
            cm.set_clim(-1, 1)

        # Add xlabel to bottom subplot
        ax[Nrow-1].set_xlabel("Time (s)", fontsize=font)

        # Set title
        tbeg = acc[0].stats.starttime
        title = f"{tbeg.date} {str(tbeg.time).split('.')[0]} UTC  |  f = {fmin}-{fmax} Hz"
        if baz is not None:
            title += f"  |  BAz = {round(baz, 1)}°"
        if distance is not None:
            title += f"  |  ED = {round(distance/1000,1)} km"
        title += f"  |  T = {twin_sec}s ({int(100*twin_overlap)}%)"
        ax[0].set_title(title)

        plt.show()
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
        
        def __multitaper_psd(arr: array, dt: float, n_win: int=5, time_bandwidth: float=4.0) -> Tuple[array, array]:
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
            f1, psd1 = __multitaper_psd(
                rot.select(channel=comp_pattern)[0].data, 
                rot[0].stats.delta,
                n_win=Tsec
            )
            f2, psd2 = __multitaper_psd(
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
            
            axes[i].grid(which="both", alpha=0.5)
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
                ax2.set_ylabel(r"PSD (rad$^2$/s$^2$/Hz)", fontsize=font, color=rot_color)
            
            # Add component label
            axes[i].set_title(f"Component {comp_name}", fontsize=font)

        # Adjust layout to accommodate supertitle
        plt.subplots_adjust(top=0.90)
        
        return fig

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
        
        def __mask_cone(arr2d: array, ff: array, thresholds: array, fill: float=nan) -> array:
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
        
        cone_mask = __mask_cone(power, freqs, 1/coi)
        power_masked = power * cone_mask
        
        # Calculate global statistics
        global_mean = nanmean(power_masked, axis=1)
        global_sum = nansum(power_masked, axis=1)
        
        # Generate diagnostic plot if requested
        if plot:
            # ... (plotting code from original function) ...
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
        n_components = len(rot) + len(acc)
        
        # Create figure with GridSpec
        # Each component needs 2 rows - one for waveform and one for CWT
        fig = plt.figure(figsize=(15, 4*n_components))
        gs = GridSpec(2*n_components, 1, figure=fig, height_ratios=[1, 3]*n_components, hspace=0.3)

        # Component mapping
        components = []
        for tr in rot:
            components.append((tr.stats.channel[-1], 'Rotation'))
        for tr in acc:
            components.append((tr.stats.channel[-1], 'Translation'))
        
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
                label = f"$\Omega_{comp}$"
            else:
                tr = acc.select(channel=f"*{comp}")[0]
                data = tr.data * acc_scale
                unit = r"mm/s$^2$"
                label = f"$a_{comp}$"
            
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
            key = f"{comp}_{data_type}"
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
        cbar_ax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.set_label("Normalized CWT Power", fontsize=font)
        
        plt.subplots_adjust(right=0.9)
        return fig
    
    def plot_backazimuth_results(self, baz_results: Dict, wave_type: str='love', 
                                baz_theo: float=None, baz_theo_margin: float=10, 
                                cc_threshold: float=None, minors: bool=True) -> plt.Figure:
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
        rot_scale, rot_unit = 1e6, r"$\mu$rad/s"
        trans_scale, trans_unit = 1e3, r"mm/s$^2$"
        
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

        # apply cc threshold if provided
        if cc_threshold is not None:
            mask = baz_results['cc_max'] >= cc_threshold
            time = baz_results['cc_max_t'][mask]
            baz = baz_results['cc_max_y'][mask]
            cc = baz_results['cc_max'][mask]
        else:
            time = baz_results['cc_max_t']
            baz = baz_results['cc_max_y']
            cc = baz_results['cc_max']

        # Plot transverse components
        times = acc.select(channel="*HZ")[0].times()

        if wave_type == "love":

            # Plot translational data
            ax_wave.plot(times, ht*trans_scale, 'black', label=f"{self.tra_seed[0].split('.')[1]}.T", lw=lw)
            ax_wave.set_ylim(-max(abs(ht*trans_scale)), max(abs(ht*trans_scale)))

            # Add rotational data on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, jz*rot_scale, 'darkred', label=f"{self.rot_seed[0].split('.')[1]}.Z", lw=lw)
            ax_wave2.set_ylim(-max(abs(jz*rot_scale)), max(abs(jz*rot_scale)))
    
        elif wave_type == "rayleigh":
            ax_wave.plot(times, hz*trans_scale, 'black', label=f"{self.tra_seed[0].split('.')[1]}.Z", lw=lw)
            ax_wave.set_ylim(-max(abs(hz*trans_scale)), max(abs(hz*trans_scale)))

            # Add rotational data on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, jt*rot_scale, 'darkred', label=f"{self.rot_seed[0].split('.')[1]}.T", lw=lw)
            ax_wave2.set_ylim(-max(abs(jt*rot_scale)), max(abs(jt*rot_scale)))
            
        # Configure waveform axes
        # ax_wave.grid(which="both", ls=":", alpha=0.7, color="grey", zorder=0)
        ax_wave.legend(loc=1)
        ax_wave.set_ylabel(f"acceleration ({trans_unit})", fontsize=font)
        ax_wave2.tick_params(axis='y', colors="darkred")
        ax_wave2.set_ylabel(f"rotation rate ({rot_unit})", color="darkred", fontsize=font)
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
        ax_baz.set_ylabel("Love Wave BAz (°)", fontsize=font)
        
        # Add theoretical backazimuth
        ax_baz.plot([min(times), max(times)], [baz_theo, baz_theo],
                    color='k', ls='--', label='Theoretical BAz', zorder=1)
        ax_baz.fill_between([baz_theo-baz_theo_margin, baz_theo+baz_theo_margin],
                           [min(times), min(times)],
                           color='grey', alpha=0.5, zorder=1)

        # Compute statistics
        deltaa = 10
        angles1 = arange(0, 365, deltaa)
        angles2 = arange(0, 365, 1)
    
        # Compute histogram
        hist = histogram(baz,
                         bins=len(angles1)-1,
                         range=[min(angles1), max(angles1)], 
                         weights=cc, 
                         density=True)

        # Compute KDE
        kde1 = sts.gaussian_kde(baz, weights=cc)

        # Show statistics
        baz_max = angles2[np.argmax(kde1.pdf(angles2))]
        baz_mean = round(average(baz, weights=cc), 0)
        baz_std = np.sqrt(cov(baz, aweights=cc))
        print(f"max = {baz_max}, mean = {baz_mean}, std = {baz_std}")

        # Add histogram
        # ax_hist2.plot(kernel_density(np.linspace(0, 360, 100)), np.linspace(0, 360, 100), 'k-', lw=2)
        ax_hist.hist(baz, bins=len(angles1)-1, range=[min(angles1), max(angles1)],
                     weights=cc, orientation="horizontal", density=True, color="grey")
        ax_hist.plot(kde1.pdf(angles1), angles1, c="k", lw=2, label='KDE')
        ax_hist.set_ylim(-5, 365)
        ax_hist.invert_xaxis()
        ax_hist.set_axis_off()
        
        # Add colorbar
        cbar_ax = ax_baz.inset_axes([1.02, 0., 0.02, 1])
        cb = plt.colorbar(scatter, cax=cbar_ax)
        cb.set_label("cross-correlation coefficient", fontsize=font)
        
        # Add title and labels
        title = f"{self.tbeg.date} {str(self.tbeg.time).split('.')[0]} UTC"
        if self.fmin is not None and self.fmax is not None:
            title += f" | {self.fmin}-{self.fmax} Hz"
        if cc_threshold is not None:
            title += f" | cc >= {cc_threshold}"

        fig.suptitle(title, fontsize=font+2, y=0.93)
        
        ax_baz.set_xlabel("time (s)", fontsize=font)

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
        rot_scale, rot_unit = 1e9, r"nrad/s"
        tra_scale, tra_unit = 1e6, r"$\mu$m/s$^2$"
        
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
            mask = velocity_results['ccoef'] >= cc_threshold
        else:
            mask = velocity_results['ccoef'] >= 0

        # Plot waveforms based on wave type
        if  wave_type == 'love':

            # Plot transverse acceleration
            ax_wave.plot(times, acc_t, 'black', 
                        label=f"{self.tra_seed[0].split('.')[1]}.T", lw=lw)
            
            # Plot vertical rotation on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, rot_z, 'darkred',
                         label=f"2x {self.rot_seed[0].split('.')[1]}.Z", lw=lw)
            
        elif wave_type == 'rayleigh':

            # Plot vertical acceleration
            ax_wave.plot(times, acc_z, 'black',
                        label=f"{self.tra_seed[0].split('.')[1]}.Z", lw=lw)
            
            # Plot transverse rotation on twin axis
            ax_wave2 = ax_wave.twinx()
            ax_wave2.plot(times, rot_t, 'darkred',
                         label=f"{self.rot_seed[0].split('.')[1]}.T", lw=lw)

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
            title += f" | cc >= {cc_threshold}"
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
        from numpy import array, std, ones, sum, mean, ones_like
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

            st_new.select(component='N')[0].data = romy_n
            st_new.select(component='E')[0].data = romy_e
        except Exception as e:
            print(f"Warning: Error updating rotated data: {e}")
            return st

        return st_new

