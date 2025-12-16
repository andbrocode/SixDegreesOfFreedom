"""
SeismicArray class for handling seismic array data.

This module provides functionality to set up and process seismic array data
by providing station codes and a reference station through YAML configuration.
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from obspy import UTCDateTime, Stream, Inventory, read_inventory
from obspy.clients.fdsn import Client
from obspy.signal.util import util_geo_km
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal import array_analysis as AA
from typing import List, Dict, Optional, Tuple, Union
from .utils.print_dict_tree import print_dict_tree
from .utils.decimator import Decimator
from .plots.plot_azimuth_distance_range import plot_azimuth_distance_range
from .plots.plot_frequency_patterns import plot_frequency_patterns, plot_frequency_patterns_simple
from .plots.plot_frequency_limits import plot_frequency_limits
from .plots.plot_array_geometry import plot_array_geometry


class seismicarray:
    """
    A class to handle seismic array data and operations for Array derived rotation.

    This class allows setting up an array by providing a list of seed codes
    and a reference station. It handles data fetching, preprocessing, and
    array-specific computations.
    """

    def __init__(self, config_file: str, fdsn_client: Optional[Union[str, List[str]]] = None):
        """
        Initialize SeismicArray with configuration from YAML file.

        Args:
            config_file (str): Path to YAML configuration file containing array setup
            fdsn_client (str or List[str], optional): FDSN client name(s) to use.
                                                      Can be a single client name (str) or a list of client names.
                                                      If provided, overrides the 'fdsn_client' value in the config file.
                                                      If None, uses the value from the config file (default: 'IRIS').
        """
        self.config = self._load_config(config_file)
        
        # Handle client configuration - can be string or list
        # Use provided fdsn_client parameter if given, otherwise use config file value
        if fdsn_client is not None:
            fdsn_client_config = fdsn_client
        else:
            fdsn_client_config = self.config.get('fdsn_client', 'IRIS')
        
        self.clients = {}
        self.client_mapping = {}
        
        if isinstance(fdsn_client_config, str):
            # Single client
            try:
                client = Client(fdsn_client_config)
                self.clients[fdsn_client_config] = client
                self.client = client  # Keep backward compatibility
            except Exception as e:
                raise ValueError(f"Failed to initialize FDSN client '{fdsn_client_config}': {str(e)}")
        elif isinstance(fdsn_client_config, list):
            # Multiple clients - initialize all available ones
            if len(fdsn_client_config) == 0:
                raise ValueError("fdsn_client list cannot be empty")
            
            available_clients = []
            failed_clients = []
            
            for client_name in fdsn_client_config:
                if not isinstance(client_name, str):
                    raise ValueError(f"All FDSN client names must be strings. Got: {type(client_name).__name__}")
                
                try:
                    client = Client(client_name)
                    self.clients[client_name] = client
                    available_clients.append(client_name)
                except Exception as e:
                    failed_clients.append((client_name, str(e)))
                    print(f"Warning: Failed to initialize client '{client_name}': {str(e)}")
            
            if not available_clients:
                error_msg = f"All FDSN clients failed to initialize:\n"
                for client_name, error in failed_clients:
                    error_msg += f"  - {client_name}: {error}\n"
                raise ValueError(error_msg.strip())
            
            if failed_clients and len(available_clients) < len(fdsn_client_config):
                print(f"Note: {len(available_clients)} out of {len(fdsn_client_config)} clients initialized successfully.")
            
            self.client = self.clients[available_clients[0]]  # Use first available as default
        else:
            raise ValueError(f"fdsn_client must be a string or list of strings. Got: {type(fdsn_client_config).__name__}")
        
        self.stations = self.config['stations']
        self.reference_station = self.config['reference_station']
        
        # Auto-map clients to stations if multiple clients available
        if len(self.clients) > 1:
            self._auto_map_clients(verbose=False)  # Silent during initialization
        
        # If auto-mapping failed or only one client, map all stations to default client
        if not self.client_mapping:
            for station in self.stations:
                self.client_mapping[station] = list(self.clients.keys())[0]
        else:
            # Ensure all stations are mapped (add failed ones to default client)
            default_client = list(self.clients.keys())[0]
            for station in self.stations:
                if station not in self.client_mapping:
                    self.client_mapping[station] = default_client
        
        # Handle channel_prefix configuration - can be string or list
        channel_prefix_config = self.config.get('channel_prefix', 'B')
        if isinstance(channel_prefix_config, str):
            self.channel_prefixes = [channel_prefix_config]
            self.channel_prefix = channel_prefix_config  # Keep backward compatibility
        elif isinstance(channel_prefix_config, list):
            if len(channel_prefix_config) == 0:
                raise ValueError("channel_prefix list cannot be empty")
            self.channel_prefixes = channel_prefix_config
            self.channel_prefix = channel_prefix_config[0]  # Default to first for backward compatibility
        else:
            raise ValueError(f"channel_prefix must be a string or list of strings. Got: {type(channel_prefix_config).__name__}")
        
        # Track which channel prefix works for each station
        self.channel_prefix_mapping = {}
        
        # Initialize channel prefix mapping for all stations (will be updated during auto-mapping or requests)
        for station in self.stations:
            self.channel_prefix_mapping[station] = self.channel_prefix  # Default to first prefix
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

    def copy(self):
        """
        Create a deep copy of the seismicarray object.
        
        Returns:
            seismicarray: A new seismicarray instance with copied attributes
        """
        import copy
        
        # Create a new instance with the same configuration
        new_instance = seismicarray.__new__(seismicarray)
        
        # Copy all attributes that may have been modified during the object's lifecycle
        attributes_to_copy = [
            'config', 'clients', 'client_mapping', 'client', 'stations', 'reference_station',
            'channel_prefix', 'channel_prefixes', 'channel_prefix_mapping', 'response_output', 
            'output_format', 'combined_stream', 'inventories', 'stream', 'rot_stream', 
            'station_coordinates', 'station_distances', 'failed_stations', 'adr_parameters', 
            'azimuthal_distances'
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

    def _auto_map_clients(self, test_duration_hours: float = 1.0, verbose: bool = True):
        """
        Automatically map clients to stations by doing trial runs.
        Tests each station against all available clients to find which client works best.
        
        Args:
            test_duration_hours (float): Duration in hours for test data requests
            verbose (bool): Whether to print progress information
        """
        if len(self.clients) == 0:
            if verbose:
                print("Warning: No FDSN clients available for auto-mapping")
            return
        
        if len(self.clients) == 1:
            # Only one client, no need to map
            if verbose:
                print("Only one FDSN client available, skipping auto-mapping")
            return

        # Use a recent time window for testing
        from obspy import UTCDateTime
        end_time = UTCDateTime.now() - 10*86400
        start_time = end_time - (test_duration_hours * 3600)
        
        successful_mappings = {}
        failed_stations = []
        
        if verbose:
            print(f"\nAuto-mapping clients to stations (testing {len(self.clients)} clients for {len(self.stations)} stations)...")
        
        for station in self.stations:
            net, sta = station.split(".")
            station_success = False
            
            # Try each client for this station
            for client_name, client in self.clients.items():
                try:
                    # Test inventory request
                    inventory = client.get_stations(
                        network=net,
                        station=sta,
                        starttime=start_time,
                        endtime=end_time,
                        level="response"
                    )
                    
                    # Test if we can get coordinates - try each channel prefix
                    coords = None
                    working_prefix = None
                    
                    # Build list of prefixes to try: config prefixes first, then common fallbacks
                    prefixes_to_try = list(self.channel_prefixes)
                    
                    # Add common fallback prefixes if not already tried
                    fallback_prefixes = ['B', 'H', 'E', 'S', 'L']  # Common channel prefixes
                    for fallback in fallback_prefixes:
                        if fallback not in prefixes_to_try:
                            prefixes_to_try.append(fallback)
                    
                    # Try each prefix
                    for prefix in prefixes_to_try:
                        try:
                            # Handle both single letter (H) and full code (HHZ) formats
                            if len(prefix) == 1:
                                channel_code = f"{prefix}HZ"
                            elif len(prefix) == 3:
                                channel_code = prefix
                            else:
                                continue  # Skip invalid formats
                            
                            coords = inventory.get_coordinates(f"{net}.{sta}..{channel_code}")
                            working_prefix = prefix[0] if len(prefix) == 3 else prefix  # Store as single letter
                            break
                        except Exception:
                            continue
                    
                    # If still no coordinates found, try to discover available channels from inventory
                    if coords is None:
                        try:
                            # Try to find any available Z channel in the inventory
                            for network in inventory:
                                for station_obj in network:
                                    if station_obj.code == sta:
                                        for channel in station_obj:
                                            # Look for any vertical (Z) component channel
                                            if len(channel.code) >= 3 and channel.code[2] == 'Z':
                                                try:
                                                    channel_code = channel.code
                                                    coords = inventory.get_coordinates(f"{net}.{sta}..{channel_code}")
                                                    working_prefix = channel_code[0]  # Store first letter as prefix
                                                    break
                                                except Exception:
                                                    continue
                                        if coords is not None:
                                            break
                                    if coords is not None:
                                        break
                                if coords is not None:
                                    break
                        except Exception:
                            pass  # If discovery fails, continue to next client
                    
                    if coords is not None:
                        # If we get here, this client works for this station
                        successful_mappings[station] = client_name
                        if working_prefix:
                            self.channel_prefix_mapping[station] = working_prefix
                        station_success = True
                        if verbose:
                            prefix_info = f" (prefix: {working_prefix})" if working_prefix else ""
                            print(f"  {station} -> {client_name}{prefix_info}")
                        break
                    
                except Exception:
                    continue
            
            if not station_success:
                failed_stations.append(station)
                if verbose:
                    print(f"  {station} -> failed (will use default client)")

        # Store the successful mappings
        self.client_mapping = successful_mappings
        
        if verbose:
            print(f"\nAuto-mapping completed:")
            print(f"  Successfully mapped: {len(successful_mappings)} stations")
            if failed_stations:
                print(f"  Failed to map: {len(failed_stations)} stations (will use default client)")
                if len(failed_stations) <= 5:  # Only show if not too many
                    print(f"  Failed stations: {failed_stations}")
            
            if successful_mappings:
                print(f"\nClient usage summary:")
                client_usage = {}
                for station, client_name in successful_mappings.items():
                    client_usage[client_name] = client_usage.get(client_name, 0) + 1
                for client_name, count in sorted(client_usage.items()):
                    print(f"  {client_name}: {count} stations")

    def get_client_for_station(self, station: str):
        """
        Get the appropriate client for a given station.
        
        Args:
            station (str): Station code (e.g., 'PY.PFOIX')
            
        Returns:
            Client: The appropriate FDSN client for this station
        """
        if station in self.client_mapping:
            client_name = self.client_mapping[station]
            return self.clients[client_name]
        else:
            # Fallback to default client
            return self.client

    def show_client_mapping(self):
        """Display the current client-to-station mapping."""
        print("Client-to-Station Mapping:")
        print("=" * 40)
        
        if not self.client_mapping:
            print("No client mapping available. Using default client for all stations.")
            return
        
        for station, client_name in self.client_mapping.items():
            status = "✓" if client_name in self.clients else "✗"
            print(f"  {station} -> {client_name} {status}")
        
        # Show usage summary
        client_usage = {}
        for station, client_name in self.client_mapping.items():
            client_usage[client_name] = client_usage.get(client_name, 0) + 1
        
        print(f"\nClient usage summary:")
        for client_name, count in client_usage.items():
            print(f"  {client_name}: {count} stations")

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
            self._drop_station(full_station, f"Missing components or no data", verbose)

        # Note: Traces are automatically removed by _drop_station
 
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

        
        # Trim all traces to shortest length
        for tr in stream:
            if tr.stats.npts > min_npts:
                tr.data = tr.data[:min_npts]
                tr.stats.npts = min_npts
        
        if verbose:
            print(f" -> all traces now have {min_npts} samples")
            
        return stream

    def _adjust_channel_prefix_by_sampling_rate(self, stream: Stream) -> Stream:
        """
        Adjust channel prefix according to SEED naming convention based on sampling rate.
        
        SEED channel naming convention:
          E: Extremely short period (>= 1000 Hz)
          S: Short period (1-10 Hz, less common)
          H: High frequency (>= 80 Hz, typically 100-250 Hz)
          B: Broadband (10-80 Hz, commonly 20-40 Hz)
          M: Mid period (0.1-1 Hz)
          L: Long period (< 1 Hz, commonly 0.1-1 Hz)
        
        This method uses a simplified version matching common practice:
          H: >= 100 Hz (high frequency)
          B: 10-100 Hz (broadband)
          L: < 10 Hz (long period)
        
        Args:
            stream (Stream): ObsPy Stream object with traces to adjust
            
        Returns:
            Stream: The same stream object with adjusted channel prefixes (modified in-place)
        """
        for tr in stream:
            sps = tr.stats.sampling_rate
            
            # Preserve the rest of the channel code (component and optional subcode)
            channel_suffix = tr.stats.channel[1:] if len(tr.stats.channel) > 1 else ""
            
            # Determine prefix based on sampling rate (check from highest to lowest)
            if sps >= 100:
                # High frequency: >= 100 Hz
                tr.stats.channel = "H" + channel_suffix
            elif sps >= 10:
                # Broadband: 10-100 Hz
                tr.stats.channel = "B" + channel_suffix
            else:
                # Long period: < 10 Hz
                tr.stats.channel = "L" + channel_suffix
        
        return stream

    def decimation(self, target_freq: float = 1.0, verbose: bool = False) -> Stream:
        """
        Internal function to apply decimation to a stream.
        
        Args:
            target_freq (float): Target sampling frequency in Hz
            verbose (bool): Whether to print verbose output
            
        Returns:
            Stream: Decimated stream
        """

        stream = self.stream

        if len(stream) == 0:
            return stream
            
        if verbose:
            print(f"\nApplying decimation to stream with {len(stream)} traces")
            print(f"Original sampling rate: {stream[0].stats.sampling_rate:.2f} Hz")
            print(f"Target sampling rate: {target_freq:.2f} Hz")
        
        try:
            # Create decimator instance
            decimator = Decimator(target_freq=target_freq)
            
            # Apply decimation
            decimated_stream = decimator.apply_decimation_stream(stream)
            
            # Adjust channel prefixes based on new sampling rate after decimation
            # decimated_stream = self._adjust_channel_prefix_by_sampling_rate(decimated_stream)
            
            if verbose and len(decimated_stream) > 0:
                print(f"Decimated sampling rate: {decimated_stream[0].stats.sampling_rate:.2f} Hz")
                
            self.stream = decimated_stream
            
        except Exception as e:
            if verbose:
                print(f"Warning: Decimation failed: {str(e)}")

    def _drop_station(self, station: str, reason: str, verbose: bool = False) -> None:
        """
        Drop a station from all class attributes and data structures.
        This is the comprehensive function to remove a station completely.
        
        Args:
            station (str): Station code to remove
            reason (str): Reason for removal
            verbose (bool): Whether to print verbose output
        """
        if verbose:
            print(f" -> dropping station {station}: {reason}")
            
        # Remove from stations list
        if station in self.stations:
            self.stations.remove(station)
            
        # Remove from inventories
        if station in self.inventories:
            del self.inventories[station]
            
        # Remove from station coordinates
        if station in self.station_coordinates:
            del self.station_coordinates[station]
            
        # Remove from station distances
        if station in self.station_distances:
            del self.station_distances[station]
            
        # Remove from client mapping
        if station in self.client_mapping:
            del self.client_mapping[station]
            
        # Remove from channel prefix mapping
        if station in self.channel_prefix_mapping:
            del self.channel_prefix_mapping[station]
            
        # Add to failed stations list
        if station not in self.failed_stations:
            self.failed_stations.append(station)
            
        # Remove traces from stream if it exists
        if len(self.stream) > 0:
            sta_code = station.split('.')[1]  # Get station code without network
            selection = self.stream.select(station=sta_code)
            if selection:
                for tr in selection:
                    self.stream.remove(tr)
                if verbose:
                    print(f"    -> removed {len(selection)} traces from stream")
                    
        # Remove traces from rotation stream if it exists
        if len(self.rot_stream) > 0:
            sta_code = station.split('.')[1]  # Get station code without network
            selection = self.rot_stream.select(station=sta_code)
            if selection:
                for tr in selection:
                    self.rot_stream.remove(tr)
                if verbose:
                    print(f"    -> removed {len(selection)} traces from rotation stream")
                    
        # Remove traces from combined stream if it exists
        if isinstance(self.combined_stream, Stream) and len(self.combined_stream) > 0:
            sta_code = station.split('.')[1]  # Get station code without network
            selection = self.combined_stream.select(station=sta_code)
            if selection:
                for tr in selection:
                    self.combined_stream.remove(tr)
                if verbose:
                    print(f"    -> removed {len(selection)} traces from combined stream")

    def _remove_station(self, station: str, reason: str, verbose: bool = False) -> None:
        """
        Legacy function - now calls _drop_station for backward compatibility.
        
        Args:
            station (str): Station code to remove
            reason (str): Reason for removal
            verbose (bool): Whether to print verbose output
        """
        self._drop_station(station, reason, verbose)

    def request_inventories(self, starttime: UTCDateTime, endtime: UTCDateTime, verbose: bool = False) -> None:
        """
        Fetch station inventories for all stations in the array.
        Stations without valid inventories are removed from the array.
        When multiple FDSN clients are available, tries all clients as fallback if the mapped client fails.

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
            station_success = False
            last_error = None
            
            # Get list of clients to try - start with mapped client, then try others if available
            clients_to_try = []
            mapped_client_name = self.client_mapping.get(station)
            if mapped_client_name and mapped_client_name in self.clients:
                clients_to_try.append((mapped_client_name, self.clients[mapped_client_name]))
            
            # Add other clients as fallback if we have multiple clients
            if len(self.clients) > 1:
                for client_name, client in self.clients.items():
                    if client_name != mapped_client_name:
                        clients_to_try.append((client_name, client))
            
            # If no mapping exists, try all clients
            if not clients_to_try:
                clients_to_try = [(name, client) for name, client in self.clients.items()]
            
            # Try each client until one succeeds
            for client_name, client in clients_to_try:
                try:
                    if verbose and not station_success:
                        print(f" -> requesting inventory for station {station} using {client_name}")
                    
                    inventory = client.get_stations(
                        network=net,
                        station=sta,
                        starttime=starttime,
                        endtime=endtime,
                        level="response"
                    )

                    # Verify we can get coordinates - try each channel prefix
                    coords = None
                    working_prefix = None
                    
                    # Build list of prefixes to try: mapped prefix first, then config prefixes, then common fallbacks
                    prefixes_to_try = []
                    
                    # First try the mapped prefix if available
                    mapped_prefix = self.channel_prefix_mapping.get(station)
                    if mapped_prefix and mapped_prefix in self.channel_prefixes:
                        prefixes_to_try.append(mapped_prefix)
                    
                    # Add all config prefixes (avoid duplicates)
                    for prefix in self.channel_prefixes:
                        if prefix not in prefixes_to_try:
                            prefixes_to_try.append(prefix)
                    
                    # Add common fallback prefixes if not already tried
                    fallback_prefixes = ['B', 'H', 'E', 'L']
                    for fallback in fallback_prefixes:
                        if fallback not in prefixes_to_try:
                            prefixes_to_try.append(fallback)
                    
                    # Try each prefix
                    for prefix in prefixes_to_try:
                        try:
                            # Handle both single letter (H) and full code (HHZ) formats
                            if len(prefix) == 1:
                                channel_code = f"{prefix}HZ"
                            elif len(prefix) == 3:
                                channel_code = prefix
                            else:
                                continue  # Skip invalid formats
                            
                            coords = inventory.get_coordinates(f"{net}.{sta}..{channel_code}")
                            working_prefix = prefix[0] if len(prefix) == 3 else prefix  # Store as single letter
                            break
                        except Exception:
                            continue
                    
                    # If still no coordinates found, try to discover available channels from inventory
                    if coords is None:
                        try:
                            # Try to find any available Z channel in the inventory
                            for network in inventory:
                                for station_obj in network:
                                    if station_obj.code == sta:
                                        for channel in station_obj:
                                            # Look for any vertical (Z) component channel
                                            if len(channel.code) >= 3 and channel.code[2] == 'Z':
                                                try:
                                                    channel_code = channel.code
                                                    coords = inventory.get_coordinates(f"{net}.{sta}..{channel_code}")
                                                    working_prefix = channel_code[0]  # Store first letter as prefix
                                                    if verbose:
                                                        print(f" -> found alternative channel {channel_code} for {station}")
                                                    break
                                                except Exception:
                                                    continue
                                        if coords is not None:
                                            break
                                    if coords is not None:
                                        break
                                if coords is not None:
                                    break
                        except Exception as e:
                            if verbose:
                                print(f" -> could not discover channels from inventory: {str(e)}")
                    
                    if coords is None:
                        raise ValueError(f"Could not get coordinates for {station} with any channel prefix")
                    
                    # Success - store inventory and coordinates
                    self.inventories[station] = inventory
                    self.station_coordinates[station] = {
                        'latitude': float(coords['latitude']),
                        'longitude': float(coords['longitude']),
                        'elevation': float(coords['elevation'])
                    }
                    
                    # Update client mapping if we used a different client
                    if client_name != mapped_client_name:
                        self.client_mapping[station] = client_name
                        if verbose:
                            print(f" -> updated client mapping for {station} to {client_name}")
                    
                    # Update channel prefix mapping if we used a different prefix
                    if working_prefix and working_prefix != mapped_prefix:
                        self.channel_prefix_mapping[station] = working_prefix
                        if verbose:
                            print(f" -> updated channel prefix for {station} to {working_prefix}")
                    
                    successful_stations.append(station)
                    station_success = True
                    
                    if verbose:
                        print(f" -> successfully obtained inventory for {station} using {client_name}")
                    break  # Success, no need to try other clients
                    
                except Exception as e:
                    last_error = str(e)
                    if verbose and client_name == mapped_client_name:
                        print(f" -> failed with mapped client {client_name}: {str(e)}")
                    continue  # Try next client
            
            # If all clients failed, drop the station
            if not station_success:
                error_msg = f"Failed to get inventory from all clients"
                if last_error:
                    error_msg += f": {last_error}"
                self._drop_station(station, error_msg, verbose)
                failed_stations.append(station)
        
        # Check if we have the reference station
        if self.reference_station not in successful_stations:
            raise ValueError(f"Could not obtain inventory for reference station {self.reference_station}")
            
        # Check if we have enough stations
        if len(successful_stations) < 1:
            raise ValueError(f"Not enough station inventories obtained. Found {len(successful_stations)}, need at least 1")
        
        # Warn if very few stations
        if len(successful_stations) < 3:
            print(f"Warning: Only {len(successful_stations)} stations available. Array analysis may be limited.")
            
        if verbose:
            print(f"\nInventory status:")
            print(f" -> Successfully validated: {len(successful_stations)} stations")
            print(f" -> Removed: {len(failed_stations)} stations")
            print("\nRemaining stations:")
            for station in successful_stations:
                print(f" - {station}")

    def request_waveforms(self, begtime: UTCDateTime, endtime: UTCDateTime,
                     remove_response: bool = True, 
                     detrend: bool = True,
                     taper: bool = True,
                     filter_params: Optional[Dict] = None,
                     trim_samples: bool = True,
                     output: bool = False,
                     verbose: bool = False,
                     plot: bool = False) -> Stream:
        """
        Fetch and preprocess waveforms for all stations.
        When multiple FDSN clients are available, tries all clients as fallback if the mapped client fails.

        Args:
            begtime (UTCDateTime): Start time for data request
            endtime (UTCDateTime): End time for data request
            remove_response (bool): Whether to remove instrument response
            detrend (bool): Whether to detrend the data
            taper (bool): Whether to taper the data
            filter_params (Dict): Optional dictionary with filter parameters
                                (e.g., {'type': 'bandpass', 'freqmin': 0.1, 'freqmax': 1.0})
            output (bool): Whether to return the stream
            verbose (bool): Whether to print verbose output
            plot (bool): Whether to plot the waveforms for each requested station
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
            station_success = False
            last_error = None
            
            # Get list of clients to try - start with mapped client, then try others if available
            clients_to_try = []
            mapped_client_name = self.client_mapping.get(station)
            if mapped_client_name and mapped_client_name in self.clients:
                clients_to_try.append((mapped_client_name, self.clients[mapped_client_name]))
            
            # Add other clients as fallback if we have multiple clients
            if len(self.clients) > 1:
                for client_name, client in self.clients.items():
                    if client_name != mapped_client_name:
                        clients_to_try.append((client_name, client))
            
            # If no mapping exists, try all clients
            if not clients_to_try:
                clients_to_try = [(name, client) for name, client in self.clients.items()]
            
            # Try each client until one succeeds
            for client_name, client in clients_to_try:
                # Get list of prefixes to try - start with mapped prefix, then try others
                prefixes_to_try = []
                mapped_prefix = self.channel_prefix_mapping.get(station, self.channel_prefix)
                if mapped_prefix in self.channel_prefixes:
                    prefixes_to_try.append(mapped_prefix)
                
                # Add other prefixes as fallback if we have multiple prefixes
                if len(self.channel_prefixes) > 1:
                    for prefix in self.channel_prefixes:
                        if prefix != mapped_prefix:
                            prefixes_to_try.append(prefix)
                
                # If no mapping exists, try all prefixes
                if not prefixes_to_try:
                    prefixes_to_try = self.channel_prefixes.copy()
                
                # Try each prefix until one succeeds
                st = None
                working_prefix = None
                for station_prefix in prefixes_to_try:
                    try:
                        if verbose and not station_success:
                            print(20*"-")
                            print(f"requesting waveforms for station {station} using {client_name} (prefix: {station_prefix})")
                        
                        # get waveforms
                        st = client.get_waveforms(
                            network=net,
                            station=sta,
                            location="*",
                            channel=f"{station_prefix}H*",
                            starttime=begtime-300,
                            endtime=endtime+300
                        )
                        
                        # If we get here, this prefix works
                        working_prefix = station_prefix
                        break
                        
                    except Exception as prefix_error:
                        if verbose and station_prefix == mapped_prefix:
                            print(f" -> failed with mapped prefix {station_prefix}: {str(prefix_error)}")
                        # Try next prefix
                        continue
                
                # If no prefix worked for this client, try next client
                if st is None:
                    continue
                
                # If we get here, we successfully got waveforms with this client and prefix
                try:

                    # check if length requires merging
                    if len(st) > 3:
                        if verbose:
                            print(f" -> merging {station} waveforms")
                        st = st.merge(fill_value='latest')

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
                        st.taper(0.05, type='cosine')

                    # filter waveforms
                    if filter_params:
                        st.filter(**filter_params)
                        st.detrend('demean')

                    # plot waveforms (if requested)
                    if plot:
                        print(st.select(station=station))
                        st.plot(equal_scale=False)

                    self.stream += st
                    station_success = True

                    # Update client mapping if we used a different client
                    if client_name != mapped_client_name:
                        self.client_mapping[station] = client_name
                        if verbose:
                            print(f" -> updated client mapping for {station} to {client_name}")
                    
                    # Update channel prefix mapping if we used a different prefix
                    if working_prefix and working_prefix != mapped_prefix:
                        self.channel_prefix_mapping[station] = working_prefix
                        if verbose:
                            print(f" -> updated channel prefix for {station} to {working_prefix}")

                    if verbose:
                        print(f" -> successfully obtained waveforms for {station} using {client_name} (prefix: {working_prefix})")
                    break  # Success, no need to try other clients

                except Exception as e:
                    last_error = str(e)
                    if verbose:
                        print(f" -> failed with mapped client {client_name}: {str(e)}")
                    continue  # Try next client

            # If all clients failed, drop the station
            if not station_success:
                error_msg = f"Failed to get waveforms from all clients"
                if last_error:
                    error_msg += f": {last_error}"
                self._drop_station(station, error_msg, verbose)
                print(f"WARNING: Failed to get waveforms for station {station} from all clients: {last_error}")

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

        rot_stream = self._adjust_channel_prefix_by_sampling_rate(rot_stream)

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

        # Adjust channel prefixes based on new sampling rate after decimation
        combined = self._adjust_channel_prefix_by_sampling_rate(combined)

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
            'channel_prefixes': self.channel_prefixes,
            'channel_prefix_mapping': self.channel_prefix_mapping,
            'response_output': self.response_output,
            'fdsn_clients': list(self.clients.keys()),
            'client_mapping': self.client_mapping,
            'adr_parameters': self.adr_parameters,
            'station_coordinates': self.station_coordinates,
            'station_distances': self.station_distances,
            'failed_stations': self.failed_stations,
        }
        print_dict_tree(info)

    def plot_array_geometry(self, show_distances: bool = True, show_dropped: bool = True, 
                          save_path: Optional[str] = None, unit: str = 'm') -> None:
        """
        Plot the array geometry showing station positions relative to the reference station.
        
        Args:
            show_distances (bool): Whether to show distances to reference station
            show_dropped (bool): Whether to show dropped/failed stations
            save_path (str, optional): Path to save the plot. If None, displays the plot
            unit (str, optional): Unit of distance. Can be 'm' or 'km' (default: 'm')
        """
        if not self.station_coordinates:
            raise ValueError("No station coordinates available. Run get_station_inventories first.")
        
        # Call the plotting function
        plot_array_geometry(self.station_coordinates, self.reference_station, 
                           self.failed_stations, show_distances, show_dropped, save_path, unit)

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
        azimuth_bins = np.arange(0, 360+azimuth_step, azimuth_step)
        min_projections = []
        max_projections = []
        
        for az in azimuth_bins:

            # Project all stations onto this azimuth direction
            projections = []
            projections_abs = []
            for station in station_data:
                projection = project_station_onto_azimuth(station['x'], station['y'], az)
                projections.append(projection)
                projections_abs.append(abs(projection))
            
            if projections:
                # min_projections.append(min(projections))
                min_projections.append(min(projections_abs))
                max_projections.append(max(projections))
            else:
                min_projections.append(np.nan)
                max_projections.append(np.nan)
        
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
        
        # filter projections with threshold to avoid division issues
        min_projections = np.where(min_projections > 10, min_projections, np.nan)


        # Calculate frequencies
        # fmin = optional_amplitude_uncertainty * apparent_velocity / distance_max
        fmin = optional_amplitude_uncertainty * apparent_velocity / max_projections
        
        # fmax = 0.25 * apparent_velocity / distance_min
        fmax = 0.25 * apparent_velocity / max_projections
        
        # Handle NaN and inf values (where no stations were found or division by zero)
        fmin = np.where(np.isnan(fmin) | np.isinf(fmin), np.nan, fmin)
        fmax = np.where(np.isnan(fmax) | np.isinf(fmax), np.nan, fmax)
    
        
        if len(fmin) > 0 and len(fmax) > 0:
            # Optimistic: maximum range for all azimuths (best case scenario)
            fmin_optimistic = np.round(np.min(fmin), 5)  # Lowest minimum frequency across all azimuths
            fmax_optimistic = np.round(np.max(fmax), 5)  # Highest maximum frequency across all azimuths
            
            # Conservative: minimum range for all azimuths (worst case scenario)
            fmin_conservative = np.round(np.max(fmin), 5)  # Highest minimum frequency across all azimuths
            fmax_conservative = np.round(np.min(fmax), 5)  # Lowest maximum frequency across all azimuths
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
    
    def plot_frequency_limits(self, velocity_range: Optional[List[float]] = None,
                            amplitude_uncertainty: float = 0.02,
                            log_scale: bool = True,
                            figsize: tuple = (10, 8),
                            save_path: Optional[str] = None) -> None:
        """
        Plot frequency limits (fmin and fmax) for different velocities on polar plots.
        
        Args:
            velocity_range (List[float], optional): List of velocities in m/s. 
                                                  If None, uses [1000, 2000, 3000, 4000]
            amplitude_uncertainty (float): Amplitude uncertainty factor (default: 0.02)
            log_scale (bool): Whether to use logarithmic scale (default: True)
            figsize (tuple): Figure size (width, height) (default: (10, 8))
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
        
        # Create freq_results dictionary
        freq_results = {
            'azimuth_angles': azimuth_angles,
            'min_projections': min_projections,
            'max_projections': max_projections
        }
        
        # Call the plotting function
        plot_frequency_limits(freq_results, velocity_range, amplitude_uncertainty, 
                            log_scale, figsize, save_path)

    def drop_stations_by_distance(self, min_distance: Optional[float] = None, 
                                max_distance: Optional[float] = None, 
                                verbose: bool = False):
        """
        Drop stations based on distance criteria from the reference station.
        
        Args:
            min_distance (float, optional): Minimum distance in meters. Stations closer than this will be dropped.
            max_distance (float, optional): Maximum distance in meters. Stations farther than this will be dropped.
            verbose (bool): Whether to print verbose output about dropped stations
            
        Raises:
            ValueError: If station distances are not available or no criteria provided
        """
        if not self.station_distances:
            raise ValueError("Station distances not available. Run compute_station_distances first.")
        
        if min_distance is None and max_distance is None:
            raise ValueError("At least one distance criterion (min_distance or max_distance) must be provided.")
        
        dropped_stations = []
        stations_to_remove = []
        
        if verbose:
            print(f"\nDropping stations based on distance criteria:")
            if min_distance is not None:
                print(f"  Minimum distance: {min_distance} m")
            if max_distance is not None:
                print(f"  Maximum distance: {max_distance} m")
            print(f"  Reference station: {self.reference_station}")
        
        for station, distance in self.station_distances.items():
            # Skip reference station - never drop it
            if station == self.reference_station:
                continue
            
            should_drop = False
            reason = []
            
            # Check minimum distance criterion
            if min_distance is not None and distance < min_distance:
                should_drop = True
                reason.append(f"too close ({distance:.1f} m < {min_distance} m)")
            
            # Check maximum distance criterion
            if max_distance is not None and distance > max_distance:
                should_drop = True
                reason.append(f"too far ({distance:.1f} m > {max_distance} m)")
            
            if should_drop:
                stations_to_remove.append(station)
                dropped_stations.append(station)
                if verbose:
                    print(f"  -> dropping {station}: {', '.join(reason)}")

        # Remove stations from all class attributes
        for station in stations_to_remove:
            self._drop_station(station, f"Distance criteria: {', '.join(reason)}", verbose)
        
        if verbose:
            print(f"\nSummary:")
            print(f"  -> Dropped: {len(dropped_stations)} stations")
            print(f"  -> Remaining: {len(self.stations)} stations")
            if dropped_stations:
                print(f"  -> Dropped stations: {dropped_stations}")

    def get_dropped_stations_info(self) -> Dict:
        """
        Get information about dropped/failed stations.
        
        Returns:
            Dict: Dictionary containing information about dropped stations
        """
        return {
            'total_dropped': len(self.failed_stations),
            'dropped_stations': self.failed_stations.copy(),
            'remaining_stations': len(self.stations),
            'remaining_station_list': self.stations.copy()
        }


