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
            'vp': self.config.get('vp', 6200),  # P-wave velocity in m/s
            'vs': self.config.get('vs', 3700),  # S-wave velocity in m/s
            'sigmau': self.config.get('sigmau', 1e-7)  # Uncertainty in displacement
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
                        'latitude': coords['latitude'],
                        'longitude': coords['longitude'],
                        'elevation': coords['elevation']
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
                
        return np.array(coordinates)

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

        # Convert lists to numpy arrays
        tse = np.array(tse)
        tsn = np.array(tsn)
        tsz = np.array(tsz)
        
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
            
        # Get reference station coordinates
        ref_coords = self.station_coordinates[self.reference_station]
        ref_lat = ref_coords['latitude']
        ref_lon = ref_coords['longitude']
        
        # Create figure with white background
        plt.figure(figsize=(10, 10), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')
        
        # Prepare station categories
        active_stations = []
        failed_stations = []
        ref_coords = None
        
        # Process all stations
        for station, coords in self.station_coordinates.items():
            # Convert to local coordinate system (in km)
            x, y = util_geo_km(
                ref_lon,
                ref_lat,
                coords['longitude'],
                coords['latitude']
            )
            
            # Convert to meters
            x *= 1000
            y *= 1000
            
            station_info = {
                'x': x,
                'y': y,
                'label': station.split('.')[1],
                'full_name': station
            }
            
            if station in self.failed_stations:
                failed_stations.append(station_info)
            elif station == self.reference_station:
                ref_coords = station_info
            else:
                active_stations.append(station_info)
        
        # Plot grid (behind everything)
        plt.grid(True, linestyle='--', alpha=0.3, zorder=0)
        
        # Plot stations by status
        legend_elements = []
        
        # Plot active stations
        if active_stations:
            active_x = [s['x'] for s in active_stations]
            active_y = [s['y'] for s in active_stations]
            active_scatter = plt.scatter(
                active_x, active_y, 
                c='dodgerblue',
                s=100,
                label='Active Stations', 
                zorder=2
            )
            legend_elements.append(active_scatter)
            
            # Add labels and distances for active stations
            for station in active_stations:
                # Station label
                plt.annotate(
                    station['label'], 
                    (station['x'], station['y']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    color='black',
                    zorder=4
                )
                
                # Distance label if requested
                if show_distances:
                    distance = self.station_distances[station['full_name']]
                    plt.annotate(
                        f'{distance:.1f}m',
                        (station['x'], station['y']),
                        xytext=(5, -15),
                        textcoords='offset points',
                        fontsize=8,
                        color='gray',
                        zorder=4
                    )
        
        # Plot reference station
        if ref_coords:
            ref_scatter = plt.scatter(
                ref_coords['x'], ref_coords['y'], 
                c='green', 
                s=200,
                marker='*', 
                label='Reference Station', 
                zorder=3
            )
            legend_elements.append(ref_scatter)
            
            # Add reference station label
            plt.annotate(
                ref_coords['label'], 
                (ref_coords['x'], ref_coords['y']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontweight='bold',
                color='red',
                zorder=4
            )
        
        # Plot failed/dropped stations
        if show_dropped and failed_stations:
            # Plot markers
            failed_x = [s['x'] for s in failed_stations]
            failed_y = [s['y'] for s in failed_stations]
            
            # Plot dropped station markers
            dropped_scatter = plt.scatter(
                failed_x, failed_y, 
                c='lightgray',
                s=80,
                marker='d',  # square marker
                label='Dropped Stations', 
                alpha=0.9,
                zorder=1
            )
            legend_elements.append(dropped_scatter)
            
            # Add 'x' overlay on dropped stations
            plt.scatter(
                failed_x,
                failed_y,
                c='red',
                s=50,
                marker='x',
                alpha=0.9,
                zorder=1
            )
            
            # Add labels for dropped stations
            for station in failed_stations:
                plt.annotate(
                    station['label'],
                    (station['x'], station['y']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    color='gray',
                    alpha=0.7,
                    style='italic',
                    zorder=4
                )

        # Calculate plot limits
        all_x = [s['x'] for s in active_stations + failed_stations + ([ref_coords] if ref_coords else [])]
        all_y = [s['y'] for s in active_stations + failed_stations + ([ref_coords] if ref_coords else [])]
        
        max_range = max(
            abs(max(all_x, default=0)), abs(min(all_x, default=0)),
            abs(max(all_y, default=0)), abs(min(all_y, default=0))
        )
        
        # Set equal aspect ratio and limits
        plt.axis('equal')
        margin = max_range * 0.1
        plt.xlim(-max_range - margin, max_range + margin)
        plt.ylim(-max_range - margin, max_range + margin)
        
        # Add labels and title
        plt.xlabel('East-West Distance (m)')
        plt.ylabel('North-South Distance (m)')
        plt.title('Array Geometry', pad=20)
        
        # Adjust legend with collected elements
        if legend_elements:
            plt.legend(
                handles=legend_elements, 
                loc='upper right',
            )
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
