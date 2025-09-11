"""
SeismicArray class for handling seismic array data.

This module provides functionality to set up and process seismic array data
by providing station codes and a reference station through YAML configuration.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from obspy import UTCDateTime, Stream, read_inventory
from obspy.clients.fdsn import Client
from obspy.signal.util import util_geo_km
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal import array_analysis as AA
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from .utils.print_dict_tree import print_dict_tree
from .plots.plot_azimuth_distance_range import plot_azimuth_distance_range
from .plots.plot_frequency_patterns import plot_frequency_patterns, plot_frequency_patterns_simple
from .plots.plot_array_geometry import plot_array_geometry


class seismicarray:
    """
    A class to handle seismic array data and operations for Array derived rotation.

    This class allows setting up an array by providing a list of seed codes
    and a reference station. It handles data fetching, preprocessing, and
    array-specific computations.
    """

    def __init__(self, config_file: str):
        """
        Initialize SeismicArray with configuration from YAML file.

        Args:
            config_file (str): Path to YAML configuration file containing array setup
        """
        self.config = self._load_config(config_file)
        self.client = Client(self.config.get('fdsn_client', 'IRIS'))
        self.stations = self.config['stations']
        self.reference_station = self.config['reference_station']
        self.channel_prefix = self.config.get('channel_prefix', 'B')  # Default to broadband
        self.response_output = self.config.get('response_output', 'VEL')  # Default to velocity
        self.output_format = self.config.get('output_format', 'file')
        self.combined_stream = None
        self.inventories = {}
        self.stream = Stream()
        self.rot_stream = Stream()  # Store rotation stream
        self.station_coordinates = {}
        self.station_distances = {}
        self.failed_stations = []  # Track failed stations
        self.adr_parameters = {
            'vp': float(self.config.get('vp', 6200.)),  # P-wave velocity in m/s
            'vs': float(self.config.get('vs', 3700.)),  # S-wave velocity in m/s
            'sigmau': float(self.config.get('sigmau', 1e-7))  # Uncertainty in displacement
        }
        
        # Store azimuthal distance results
        self.azimuthal_distances = {
            'azimuth_angles': None,
            'min_projections': None,
            'max_projections': None,
            'azimuth_step': None
        }
        
        # Validate configuration
        self._validate_config()

    def _load_config(self, config_file: str) -> Dict:
        """
        Load and parse YAML configuration file.

        Args:
            config_file (str): Path to YAML configuration file

        Returns:
            Dict: Parsed configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise ValueError(f"Failed to load configuration file: {str(e)}")

    def _validate_config(self):
        """Validate the loaded configuration."""
        required_fields = ['stations', 'reference_station']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in configuration")
        
        if self.reference_station not in self.stations:
            raise ValueError("Reference station must be included in stations list")

    def _validate_array_status(self, verbose: bool = False) -> None:
        """
        Validate array status by checking reference station and minimum station count.
        Removes stations that don't have all three components from both stream and class attributes.
        
        Args:
            verbose (bool): Whether to print verbose output
            
        Raises:
            ValueError: If validation fails after removing incomplete stations
        """
        # Check if we have any data
        if len(self.stream) == 0:
            raise ValueError("No waveform data available in stream")
            
        ref_station = self.reference_station.split('.')[1]
        stations_to_remove = set()
        valid_stations = set()

        # Get all station names (without network code)
        stations = {sta.split('.')[1] for sta in self.stations}
        
        # First remove any previously failed stations
        for station in stations:
            full_station = next(s for s in self.stations if s.split('.')[1] == station)
            if full_station in self.failed_stations:
                if verbose:
                    print(f" -> skipping previously failed station {full_station}")
                stations_to_remove.add(station)
                continue
                
            # Check components
            stream_selection = self.stream.select(station=station)
            if not stream_selection:
                if verbose:
                    print(f" -> no data found for station {full_station}")
                stations_to_remove.add(station)
                continue
                
            components = set(tr.stats.channel[-1] for tr in stream_selection)
            if not components.issuperset({'Z', 'N', 'E'}):
                if verbose:
                    print(f" -> station {full_station} missing components. Found only {components}")
                stations_to_remove.add(station)
            else:
                valid_stations.add(station)
                if verbose:
                    print(f" -> station {full_station} has all required components")
        
        # drop stations
        for station in stations_to_remove:
            full_station = next(s for s in self.stations if s.split('.')[1] == station)
            self._remove_station(full_station, f"Dropped station: {full_station}", verbose)

        # Remove traces from invalid stations
        if stations_to_remove:
            for sta in stations_to_remove:
                selection = self.stream.select(station=sta)
                if selection:
                    for tr in selection:
                        self.stream.remove(tr)
                
        # Check reference station
        if ref_station not in valid_stations:
            raise ValueError(f"Reference station {ref_station} not found in data or missing components")
        elif verbose:
            print(f" -> reference station {ref_station} found with all components")
            
        # Check minimum station count
        station_count = len(valid_stations)
        if station_count < 3:
            raise ValueError(f"Not enough complete stations for ADR computation. Found {station_count}, need at least 3")
        elif verbose:
            print(f"\nValidation status:")
            print(f" -> Successfully validated: {station_count} stations")
            print(f" -> Removed/Failed: {len(stations_to_remove)} stations")
            print("\nRemaining stations:")
            for station in valid_stations:
                full_station = next(s for s in self.stations if s.split('.')[1] == station)
                print(f" - {full_station}")

    def _trim_to_same_samples(self, stream: Stream, verbose: bool = False) -> Stream:
        """
        Trim all traces in stream to have the same number of samples.
        Uses the shortest trace as reference to avoid data gaps.
        
        Args:
            stream (Stream): Stream to trim
            verbose (bool): Whether to print verbose output
            
        Returns:
            Stream: Trimmed stream
        """
        if not stream:
            return stream
            
        # Find shortest trace length
        min_npts = min(tr.stats.npts for tr in stream)
        max_npts = max(tr.stats.npts for tr in stream)

        if verbose:
            print("\nTrimming traces to same length:")
            print(f" -> shortest trace has {min_npts} samples")
            print(f" -> longest trace has {max_npts} samples")
            print(" -> samples per trace before trimming:")
            # for tr in stream:
            #     print(f"    {tr.id}: {tr.stats.npts}")
        
        # Trim all traces to shortest length
        for tr in stream:
            if tr.stats.npts > min_npts:
                tr.data = tr.data[:min_npts]
                tr.stats.npts = min_npts
        
        if verbose:
            print(f" -> all traces now have {min_npts} samples")
            
        return stream

    def _remove_station(self, station: str, reason: str, verbose: bool = False) -> None:
        """
        Remove a station from all class attributes.
        
        Args:
            station (str): Station code to remove
            reason (str): Reason for removal
            verbose (bool): Whether to print verbose output
        """
        if verbose:
            print(f" -> removing station {station}: {reason}")
            
        # Remove from all class attributes
        if station in self.stations:
            self.stations.remove(station)
        if station in self.inventories:
            del self.inventories[station]
        if station in self.station_coordinates:
            del self.station_coordinates[station]
        if station in self.station_distances:
            del self.station_distances[station]

    def request_inventories(self, starttime: UTCDateTime, endtime: UTCDateTime, verbose: bool = False) -> None:
        """
        Fetch station inventories for all stations in the array.
        Stations without valid inventories are removed from the array.

        Args:
            starttime (UTCDateTime): Start time for inventory request
            endtime (UTCDateTime): End time for inventory request
            verbose (bool): Whether to print verbose output
            
        Raises:
            ValueError: If reference station inventory cannot be obtained or too few stations have inventories
        """
        successful_stations = []
        failed_stations = []
        
        for station in self.stations[:]:  # Create copy to allow modification during iteration
            net, sta = station.split(".")
            try:
                if verbose:
                    print(f" -> requesting inventory for station {station}")
                    
                inventory = self.client.get_stations(
                    network=net,
                    station=sta,
                    starttime=starttime,
                    endtime=endtime,
                    level="response"
                )
                
                # Verify we can get coordinates with the specified channel
                try:
                    coords = inventory.get_coordinates(f"{net}.{sta}..{self.channel_prefix}HZ")
                    self.inventories[station] = inventory
                    self.station_coordinates[station] = {
                        'latitude': float(coords['latitude']),
                        'longitude': float(coords['longitude']),
                        'elevation': float(coords['elevation'])
                    }
                    successful_stations.append(station)
                    
                    if verbose:
                        print(f" -> successfully obtained inventory for {station}")
                except Exception as e:
                    self._remove_station(station, f"Invalid channel configuration: {str(e)}", verbose)
                    failed_stations.append(station)
                    
            except Exception as e:
                self._remove_station(station, f"Failed to get inventory: {str(e)}", verbose)
                failed_stations.append(station)
        
        # Check if we have the reference station
        if self.reference_station not in successful_stations:
            raise ValueError(f"Could not obtain inventory for reference station {self.reference_station}")
            
        # Check if we have enough stations
        if len(successful_stations) < 3:
            raise ValueError(f"Not enough station inventories obtained. Found {len(successful_stations)}, need at least 3")
            
        if verbose:
            print(f"\nInventory status:")
            print(f" -> Successfully validated: {len(successful_stations)} stations")
            print(f" -> Removed: {len(failed_stations)} stations")
            print("\nRemaining stations:")
            for station in successful_stations:
                print(f" - {station}")

    def request_waveforms(self, starttime: UTCDateTime, endtime: UTCDateTime,
                     remove_response: bool = True, 
                     detrend: bool = True,
                     taper: bool = True,
                     filter_params: Optional[Dict] = None,
                     trim_samples: bool = True,
                     output: bool = False,
                     verbose: bool = False) -> Stream:
        """
        Fetch and preprocess waveforms for all stations.

        Args:
            starttime (UTCDateTime): Start time for data request
            endtime (UTCDateTime): End time for data request
            remove_response (bool): Whether to remove instrument response
            detrend (bool): Whether to detrend the data
            taper (bool): Whether to taper the data
            filter_params (Dict): Optional dictionary with filter parameters
                                (e.g., {'type': 'bandpass', 'freqmin': 0.1, 'freqmax': 1.0})
            output (bool): Whether to return the stream
            verbose (bool): Whether to print verbose output
        Returns:    
            Stream: Processed ObsPy Stream object
            
        Raises:
            ValueError: If array validation fails
        """
        self.stream = Stream()
        output = output or self.response_output
        self.failed_stations = []

        # print info on processing
        if verbose:
            print(f"processing to be applied:  \n response: {remove_response} \n rotate: to ZNE \n detrend: {detrend} \n taper: {taper} \n filter: {True if filter_params else False}")

        for station in self.stations:
            net, sta = station.split(".")

            if verbose:
                print(20*"-")
                print(f"requesting waveforms for station {station}")
            try:
                # get waveforms
                st = self.client.get_waveforms(
                    network=net,
                    station=sta,
                    location="*",
                    channel=f"{self.channel_prefix}H*",
                    starttime=starttime,
                    endtime=endtime
                )

                # remove response
                if remove_response and station in self.inventories:
                    st.remove_response(
                        inventory=self.inventories[station],
                        output=self.response_output,
                        water_level=60
                    )

                # rotate to ZNE
                st.rotate(method="->ZNE", inventory=self.inventories[station])

                # detrend waveforms
                if detrend:
                    st.detrend('demean')
                    st.detrend('linear')
                    st.detrend('simple')

                # taper waveforms
                if taper:
                    st.taper(0.02)

                # filter waveforms
                if filter_params:
                    st.filter(**filter_params)
                    st.detrend('demean')

                self.stream += st

            except Exception as e:
                # add station to failed stations
                self.failed_stations.append(station)
                print(f"WARRNING: Failed to get waveforms for station {station}: {str(e)}")

        # Sort stream to ensure reference station is first
        if len(self.stream) > 0:
            # Split stream into reference and other stations
            ref_sta = self.reference_station.split('.')[1]
            ref_traces = self.stream.select(station=ref_sta)
            other_traces = Stream([tr for tr in self.stream if tr.stats.station != ref_sta])
            
            # Sort reference station traces by component
            ref_traces.sort(keys=['channel'])
            
            # Sort other stations by station name and component
            other_traces.sort(keys=['station', 'channel'])
            
            # Combine back ensuring reference station is first
            self.stream = Stream()
            self.stream += ref_traces
            self.stream += other_traces

        # Validate array status after getting all waveforms
        self._validate_array_status(verbose)
        
        # Trim all traces to same number of samples if requested
        if len(self.stream) > 0:
            self.stream = self._trim_to_same_samples(self.stream, verbose)
        
        if output:
            return self.stream

    def compute_station_distances(self) -> None:
        """
        Compute distances between all stations and the reference station.
        """
        if not self.station_coordinates:
            raise ValueError("Station coordinates not available. Run get_station_inventories first.")

        ref_coords = self.station_coordinates[self.reference_station]
        ref_lat = ref_coords['latitude']
        ref_lon = ref_coords['longitude']
        ref_elev = ref_coords['elevation']

        for station, coords in self.station_coordinates.items():
            if station == self.reference_station:
                self.station_distances[station] = 0.0
                continue

            # Convert to local coordinate system (in meters)
            lon, lat = util_geo_km(
                ref_lon,
                ref_lat,
                coords['longitude'],
                coords['latitude']
            )
            
            # Convert from km to meters and include elevation difference
            x = lon * 1000  # E-W distance in meters
            y = lat * 1000  # N-S distance in meters
            z = coords['elevation'] - ref_elev  # Vertical distance in meters
            
            # Compute 3D distance
            distance = np.sqrt(x**2 + y**2 + z**2)
            self.station_distances[station] = round(distance, 3)

    def _prepare_coordinates_for_adr(self, verbose: bool = False) -> np.ndarray:
        """
        Prepare station coordinates in the format required for ADR computation.
        Converts to local cartesian coordinates relative to reference station.
        
        Returns:
            np.ndarray: Array of shape (n_stations, 3) with [x, y, z] coordinates in meters
        """
        if not self.station_coordinates:
            raise ValueError("Station coordinates not available")
            
        ref_coords = self.station_coordinates[self.reference_station]
        ref_lat = ref_coords['latitude']
        ref_lon = ref_coords['longitude']
        ref_elev = ref_coords['elevation']
        
        coordinates = []
        
        # Process stations in the same order as data arrays
        for station in self.stations:
            if station in self.failed_stations:
                continue
                
            coords = self.station_coordinates[station]
            
            # Convert to local coordinate system
            lon, lat = util_geo_km(
                ref_lon, 
                ref_lat,
                coords['longitude'],
                coords['latitude']
            )
            
            # Convert to meters and get relative elevation
            x = lon * 1000  # E-W distance in meters
            y = lat * 1000  # N-S distance in meters
            z = coords['elevation'] - ref_elev  # Vertical distance in meters
            
            coordinates.append([x, y, z])
            
            if verbose:
                print(f" -> {station} coordinates [m]: E={x:.1f}, N={y:.1f}, Z={z:.1f}")
                
        return np.array(coordinates, dtype=np.float64)

    def compute_adr(self, stream: Optional[Stream] = None, output: bool = False, verbose: bool = False) -> Stream:
        """
        Compute array-derived rotation from the current stream.

        Args:
            stream (Stream, optional): Stream to use for ADR computation.
                                     If None, uses self.stream
            output (bool): Whether to return the stream
            verbose (bool): Whether to print verbose output
        Returns:
            Stream: Stream containing the computed rotation rates
        """
        if stream is None:
            stream = self.stream

        if len(stream) == 0:
            raise ValueError("No data available for ADR computation")

        if verbose:
            print("\nPreparing data for ADR computation:")

        # Prepare data arrays for each component
        tsz, tsn, tse = [], [], []
        ref_stream = None
        valid_stations = []

        for station in self.stations:
            if station in self.failed_stations:
                continue
                
            sta_stream = stream.select(station=station.split('.')[1])
            if len(sta_stream) != 3:
                if verbose:
                    print(f" -> skipping {station}: missing components")
                continue

            # Store reference station stream
            if station == self.reference_station:
                ref_stream = sta_stream.copy()
                if verbose:
                    print(f" -> using {station} as reference station")

            # Sort components into arrays
            z_comp = sta_stream.select(component='Z')[0]
            n_comp = sta_stream.select(component='N')[0]
            e_comp = sta_stream.select(component='E')[0]
            
            tsz.append(z_comp.data)
            tsn.append(n_comp.data)
            tse.append(e_comp.data)
            valid_stations.append(station)
            
            if verbose:
                print(f" -> added {station} components to arrays")

        if ref_stream is None:
            raise ValueError("Reference station data not found in stream")

        # Convert lists to numpy arrays with explicit float dtype
        tse = np.array(tse, dtype=np.float64)
        tsn = np.array(tsn, dtype=np.float64)
        tsz = np.array(tsz, dtype=np.float64)
        
        # Get coordinates in correct format
        if verbose:
            print("\nPreparing station coordinates:")
        coordinates = self._prepare_coordinates_for_adr(verbose)

        # Compute ADR
        try:
            if verbose:
                print("\nComputing array-derived rotation:")
                print(f" -> using {len(valid_stations)} stations")
            
            # estimate ADR with ObsPy module
            result = AA.array_rotation_strain(
                np.arange(len(valid_stations)),
                np.transpose(tse),
                np.transpose(tsn),
                np.transpose(tsz),
                self.adr_parameters['vp'],
                self.adr_parameters['vs'],
                coordinates,
                self.adr_parameters['sigmau']
            )

            if verbose:
                print(" -> ADR computation completed")
        except Exception as e:
            raise RuntimeError(f"ADR computation failed: {str(e)}")

        # Create output stream with rotation rates
        rot_stream = ref_stream.copy()
        rot_stream.clear()  # Clear data but keep metadata

        # Create traces for each rotation component
        for comp, data, channel in zip(
            ['Z', 'N', 'E'],
            [result['ts_w3'], result['ts_w2'], result['ts_w1']],
            ['BJZ', 'BJN', 'BJE']
        ):
            tr = ref_stream.select(component=comp)[0].copy()
            tr.data = data
            tr.stats.channel = channel
            rot_stream += tr
            
            if verbose:
                print(f" -> created rotation trace: {tr.id}")

        self.rot_stream = rot_stream

        if output:
            return rot_stream

    def save_6dof_data(self, output_format: str = 'file', output_path: Optional[str] = None, output_file: Optional[str] = None):
        """
        Save 6 degrees of freedom data (3 translations + 3 rotations).

        Args:
            output_format (str): Format to save data in ('file' or 'sds')
            output_path (str, optional): Path to save the data
                                       If None, uses current directory
            output_file (str, optional): Filename to save the data
                                           If None, uses current date and time
        """
        # get reference station stream
        ref_stream = self.stream.select(station=self.reference_station.split('.')[1]).copy()

        # deriviate translation data
        ref_stream = ref_stream.differentiate()

        # get rotation stream
        rot_stream = self.rot_stream.copy()

        if len(ref_stream) != 3 or len(rot_stream) != 3:
            raise ValueError("Both reference and rotation streams must have 3 components")

        # Combine streams
        combined = Stream()
        combined += ref_stream
        combined += rot_stream

        # add as attribute
        self.combined_stream = combined
        self.output_format = output_format
        
        # Save data
        if output_format.lower() == 'file':
            output_path = output_path or '.'
            filename = output_file or f"6dof_{ref_stream[0].stats.starttime.datetime.strftime('%Y%m%d_%H%M%S')}.mseed"
            if not filename.endswith('.mseed'):
                filename += ".mseed"
            output = os.path.join(output_path, filename)
            combined.write(output, format='MSEED')
            self.path_to_mseed_file = output

        elif output_format.lower() == 'sds':
            if output_path is None:
                raise ValueError("output_path must be provided for SDS format")
            
            # Save in SDS structure: Year/Network/Station/Channel.Type/Network.Station.Location.Channel.Type.Year.Day
            for tr in combined:
                year = str(tr.stats.starttime.year)
                day = str(tr.stats.starttime.julday).zfill(3)
                
                sds_path = Path(output_path) / year / tr.stats.network / tr.stats.station / f"{tr.stats.channel}.D"
                sds_path.mkdir(parents=True, exist_ok=True)
                
                filename = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}.D.{year}.{day}"
                tr.write(str(sds_path / filename), format='MSEED')
        else:
            raise ValueError("Invalid output format. Use 'file' or 'sds'.")

    def save_config_for_sixdegrees(self, output_path: str, output_file: Optional[str] = None) -> None:
        """
        Write array configuration to a YAML file in sixdegrees format.
        
        Args:
            output_path (str): Path to save the configuration file
            output_file (str, optional): Filename to save the configuration file
        """

        def get_seed(tr):
            return f"{tr.stats.network}.{tr.stats.station}.{tr.stats.location}.{tr.stats.channel}"

        # check if combined stream is available
        if self.combined_stream is None:
            raise ValueError("Combined stream not available. Run save_6dof_data first.")

        # Build configuration dictionary
        config = {}
        
        if self.output_format == 'file':
            config['data_source'] = 'mseed_file'
            config['path_to_mseed_file'] = self.path_to_mseed_file
        elif self.output_format == 'sds':
            config['data_source'] = 'sds'

        config['tbeg'] = self.combined_stream[0].stats.starttime.strftime('%Y-%m-%d %H:%M:%S')
        config['tend'] = self.combined_stream[0].stats.endtime.strftime('%Y-%m-%d %H:%M:%S')

        config['station_lon'] = float(self.station_coordinates[self.reference_station]['longitude'])
        config['station_lat'] = float(self.station_coordinates[self.reference_station]['latitude'])

        config['seed'] = str(self.reference_station)+'..'

        config['rot_seed'] = [get_seed(tr) for tr in self.combined_stream.select(channel='*J*')]
        config['tra_seed'] = [get_seed(tr) for tr in self.combined_stream.select(channel='*H*')]

        # if output_file is not provided, use default filename
        if output_file is None:
            output_file = "config_adr.yml"
        else:
            # check if name ends with .yml
            if not output_file.endswith('.yml'):
                output_file += ".yml"

        # if output_path is not provided, use current directory
        if output_path is None:
            output_path = "."

        # create output path if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = Path(output_path) / output_file

        # Write configuration with header comment
        with open(output_path, 'w') as f:
            f.write("# " + output_file + "\n")
            f.write("# Configuration generated by SeismicArray\n")
            f.write("# Date: " + UTCDateTime().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
        print(f"Configuration saved to {output_path}")

    def show_array_info(self) -> None:
        """
        Display the array object attributes in a tree format.
        """
        # Create a simple object view of the attributes
        info = {
            'stations': self.stations,
            'reference_station': self.reference_station,
            'channel_prefix': self.channel_prefix,
            'response_output': self.response_output,
            'adr_parameters': self.adr_parameters,
            'station_coordinates': self.station_coordinates,
            'station_distances': self.station_distances
        }
        print_dict_tree(info)

    def plot_array_geometry(self, show_distances: bool = True, show_dropped: bool = True, 
                          save_path: Optional[str] = None) -> None:
        """
        Plot the array geometry showing station positions relative to the reference station.
        
        Args:
            show_distances (bool): Whether to show distances to reference station
            show_dropped (bool): Whether to show dropped/failed stations
            save_path (str, optional): Path to save the plot. If None, displays the plot
        """
        if not self.station_coordinates:
            raise ValueError("No station coordinates available. Run get_station_inventories first.")
        
        # Call the plotting function
        plot_array_geometry(self.station_coordinates, self.reference_station, 
                           self.failed_stations, show_distances, show_dropped, save_path)



    def compute_azimuth_distance_range(self, azimuth_step: float = 1.0, plot: bool = True, 
                                     save_path: Optional[str] = None, show_station_labels: bool = True) -> Dict:
        """
        Compute the minimal and maximal distance with respect to the reference station 
        for each azimuth angle using both radial distance and projection methods.
        
        Args:
            azimuth_step (float): Step size for azimuth angles in degrees (default: 1.0)
            plot (bool): Whether to create a plot showing the results (default: True)
            save_path (str, optional): Path to save the plot. If None, displays the plot
            show_station_labels (bool): Whether to show station labels on the plot (default: True)
            
        Returns:
            Dict: Dictionary containing azimuth angles, min/max distances, and projections
            
        Raises:
            ValueError: If station coordinates are not available
        """

        def project_station_onto_azimuth(station_x: float, station_y: float, azimuth_degrees: float) -> float:
            """
            Project a station position onto a specific azimuth direction.
            
            Args:
                station_x: East-West coordinate of station (meters)
                station_y: North-South coordinate of station (meters)
                azimuth_degrees: Azimuth angle in degrees (0-360)
            
            Returns:
                Projection distance along the azimuth direction (meters)
            """
            # Convert azimuth to unit vector
            azimuth_rad = np.radians(azimuth_degrees)
            azimuth_vector = np.array([np.sin(azimuth_rad), np.cos(azimuth_rad)])
            
            # Station position vector
            station_vector = np.array([station_x, station_y])
            
            # Project station onto azimuth direction
            projection = np.dot(station_vector, azimuth_vector)
            
            return projection
    
        if not self.station_coordinates:
            raise ValueError("Station coordinates not available. Run request_inventories first.")
            
        # Get reference station coordinates
        ref_coords = self.station_coordinates[self.reference_station]
        ref_lat = ref_coords['latitude']
        ref_lon = ref_coords['longitude']
        
        # Prepare data for all stations (excluding reference)
        station_data = []
        for station, coords in self.station_coordinates.items():
            if station == self.reference_station:
                continue
                
            # Convert to local coordinate system (in meters)
            lon, lat = util_geo_km(
                ref_lon,
                ref_lat,
                coords['longitude'],
                coords['latitude']
            )
            
            x = lon * 1000  # E-W distance in meters
            y = lat * 1000  # N-S distance in meters
            
            # Calculate distance and azimuth
            distance = np.sqrt(x**2 + y**2)
            azimuth = np.degrees(np.arctan2(x, y))  # Azimuth from North (0-360)
            if azimuth < 0:
                azimuth += 360
                
            station_data.append({
                'station': station,
                'x': x,
                'y': y,
                'distance': distance,
                'azimuth': azimuth
            })
        
        if not station_data:
            raise ValueError("No stations available for distance calculation")
        
        # Create azimuth bins
        azimuth_bins = np.arange(0, 360, azimuth_step)
        min_projections = []
        max_projections = []
        
        for az in azimuth_bins:

            # Project all stations onto this azimuth direction
            projections = []
            for station in station_data:
                projection = project_station_onto_azimuth(station['x'], station['y'], az)
                projections.append(abs(projection))
            
            if projections:
                min_projections.append(min(projections))
                max_projections.append(max(projections))
            else:
                min_projections.append(np.nan)
                max_projections.append(np.nan)
        print(min_projections)
        print(max_projections)
        # Convert to numpy arrays
        azimuth_bins = np.array(azimuth_bins)
        min_projections = np.array(min_projections)
        max_projections = np.array(max_projections)
        
        # Create results dictionary
        results = {
            'azimuth_angles': azimuth_bins,
            'min_projections': min_projections,
            'max_projections': max_projections,
            'azimuth_step': azimuth_step,
            'station_data': station_data
        }
        
        # Store results in the object
        self.azimuthal_distances = {
            'azimuth_angles': azimuth_bins,
            'min_projections': min_projections,
            'max_projections': max_projections,
            'azimuth_step': azimuth_step
        }
        
        # Create plot if requested
        if plot:
            plot_azimuth_distance_range(results, save_path, show_station_labels)
        
        return results
    
    def convert_distances_to_frequencies(self, apparent_velocity: float, 
                                       optional_amplitude_uncertainty: float = 1e-7) -> Dict:
        """
        Convert azimuthal distances to frequency bounds using the formulas:
        fmin = optional_amplitude_uncertainty * apparent_velocity / distance_max
        fmax = 0.25 * apparent_velocity / distance_min
        
        Args:
            apparent_velocity (float): Apparent velocity in m/s
            optional_amplitude_uncertainty (float): Amplitude uncertainty (default: 1e-7)
            
        Returns:
            Dict: Dictionary containing azimuth angles, min/max frequencies, and parameters
            
        Raises:
            ValueError: If azimuthal distances are not available
        """
        if self.azimuthal_distances['azimuth_angles'] is None:
            raise ValueError("Azimuthal distances not available. Run compute_azimuth_distance_range first.")
        
        # Get stored data
        azimuth_angles = self.azimuthal_distances['azimuth_angles']
        min_projections = self.azimuthal_distances['min_projections']
        max_projections = self.azimuthal_distances['max_projections']
        
        # Print formulas as LaTeX
        # print("\\textbf{Frequency Calculation Formulas:}")
        # print("\\begin{align}")
        # print(f"f_{{min}} &= \\sigma_u \\cdot \\frac{{v_{{app}}}}{{d_{{max}}}} = {optional_amplitude_uncertainty:.2e} \\cdot \\frac{{{apparent_velocity:.0f}}}{{d_{{max}}}} \\\\")
        # print(f"f_{{max}} &= 0.25 \\cdot \\frac{{v_{{app}}}}{{d_{{min}}}} = 0.25 \\cdot \\frac{{{apparent_velocity:.0f}}}{{d_{{min}}}}")
        # print("\\end{align}")
        # print("\\textbf{Where:}")
        # print("\\begin{itemize}")
        # print(f"\\item $\\sigma_u$ = amplitude uncertainty = {optional_amplitude_uncertainty:.2e}")
        # print(f"\\item $v_{{app}}$ = apparent velocity = {apparent_velocity:.0f} m/s")
        # print("\\item $d_{{max}}$ = maximum projection distance per azimuth")
        # print("\\item $d_{{min}}$ = minimum projection distance per azimuth")
        # print("\\end{itemize}")
        # print()
        
        # filter min projections with threshold 10 m 
        min_projections = np.where(min_projections > 10, min_projections, np.nan)

        # Calculate frequencies
        # fmin = optional_amplitude_uncertainty * apparent_velocity / distance_max
        fmin = optional_amplitude_uncertainty * apparent_velocity / max_projections
        
        # fmax = 0.25 * apparent_velocity / distance_min
        fmax = 0.25 * apparent_velocity / min_projections
        
        # Handle NaN and inf values (where no stations were found or division by zero)
        fmin = np.where(np.isnan(fmin) | np.isinf(fmin), np.nan, fmin)
        fmax = np.where(np.isnan(fmax) | np.isinf(fmax), np.nan, fmax)
        
        # Round to two decimal places (only for finite values)
        fmin = np.where(np.isfinite(fmin), np.round(fmin, 2), fmin)
        fmax = np.where(np.isfinite(fmax), np.round(fmax, 2), fmax)
        
        # Calculate optimistic and conservative bounds
        # Get finite values only for bounds calculation
        finite_fmin = fmin[np.isfinite(fmin)]
        finite_fmax = fmax[np.isfinite(fmax)]
        
        if len(finite_fmin) > 0 and len(finite_fmax) > 0:
            # Optimistic: maximum range for all azimuths (best case scenario)
            fmin_optimistic = np.round(np.min(finite_fmin), 2)  # Lowest minimum frequency across all azimuths
            fmax_optimistic = np.round(np.max(finite_fmax), 2)  # Highest maximum frequency across all azimuths
            
            # Conservative: minimum range for all azimuths (worst case scenario)
            fmin_conservative = np.round(np.max(finite_fmin), 2)  # Highest minimum frequency across all azimuths
            fmax_conservative = np.round(np.min(finite_fmax), 2)  # Lowest maximum frequency across all azimuths
        else:
            # If no finite values, set to NaN
            fmin_optimistic = np.nan
            fmax_optimistic = np.nan
            fmin_conservative = np.nan
            fmax_conservative = np.nan
        
        # Create results dictionary
        frequency_results = {
            'azimuth_angles': azimuth_angles,
            'fmin': fmin,
            'fmax': fmax,
            'fmin_optimistic': fmin_optimistic,
            'fmax_optimistic': fmax_optimistic,
            'fmin_conservative': fmin_conservative,
            'fmax_conservative': fmax_conservative,
            'apparent_velocity': apparent_velocity,
            'amplitude_uncertainty': optional_amplitude_uncertainty,
            'min_projections': min_projections,
            'max_projections': max_projections
        }
        
        return frequency_results
    
    def plot_frequency_patterns(self, velocity_range: List[float], 
                              optional_amplitude_uncertainty: float = 1e-7,
                              log_scale: bool = False,
                              save_path: Optional[str] = None) -> None:
        """
        Plot frequency patterns for different apparent velocities on polar plots.
        Creates two subplots side by side: minimum and maximum frequencies.
        Each velocity is shown as a different color.
        
        Args:
            velocity_range (List[float]): List of apparent velocities in m/s
            optional_amplitude_uncertainty (float): Amplitude uncertainty (default: 1e-7)
            log_scale (bool): Whether to use logarithmic scale for frequency axis (default: False)
            save_path (str, optional): Path to save the plot. If None, displays the plot
            
        Raises:
            ValueError: If azimuthal distances are not available
        """
        if self.azimuthal_distances['azimuth_angles'] is None:
            raise ValueError("Azimuthal distances not available. Run compute_azimuth_distance_range first.")
        
        # Get stored data
        azimuth_angles = self.azimuthal_distances['azimuth_angles']
        min_projections = self.azimuthal_distances['min_projections']
        max_projections = self.azimuthal_distances['max_projections']
        
        # Call the plotting function
        plot_frequency_patterns(azimuth_angles, min_projections, max_projections, 
                               velocity_range, optional_amplitude_uncertainty, 
                               log_scale, save_path)
    
    def plot_frequency_patterns_simple(self, velocity_range: List[float], 
                                     optional_amplitude_uncertainty: float = 1e-7,
                                     log_scale: bool = False,
                                     save_path: Optional[str] = None) -> None:
        """
        Simple version: Convert azimuthal distances to frequencies and plot on polar maps.
        Creates two subplots side by side: minimum and maximum frequencies.
        
        Args:
            velocity_range (List[float]): List of apparent velocities in m/s
            optional_amplitude_uncertainty (float): Amplitude uncertainty (default: 1e-7)
            log_scale (bool): Whether to use logarithmic scale for frequency axis (default: False)
            save_path (str, optional): Path to save the plot. If None, displays the plot
        """
        # First compute azimuthal distances if not available
        if self.azimuthal_distances['azimuth_angles'] is None:
            print("Computing azimuthal distances first...")
            self.compute_azimuth_distance_range(azimuth_step=5.0, plot=False)
        
        # Get data
        azimuth_angles = self.azimuthal_distances['azimuth_angles']
        min_projections = self.azimuthal_distances['min_projections']
        max_projections = self.azimuthal_distances['max_projections']
        
        # Call the plotting function
        plot_frequency_patterns_simple(azimuth_angles, min_projections, max_projections, 
                                     velocity_range, optional_amplitude_uncertainty, 
                                     log_scale, save_path)
    
