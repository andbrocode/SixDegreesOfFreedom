# Processing Workflows

This directory contains ready-to-use processing workflows for common 6-DoF seismic data analysis tasks. Each workflow is designed to be easily configurable and can be run independently or as part of larger processing chains.

## Available Workflows

### 1. Array-Derived Rotation (ADR) Computation
**Directory**: `compute_adr/`

Computes array-derived rotation data from seismic array observations and stores the results in SDS (SeisComP Data Structure) format for continuous processing.

**Key Features**:
- Daily continuous 6-DoF data creation
- SDS format storage and management
- Processing statistics and monitoring
- Configurable array parameters

### 2. Backazimuth Analysis
**Directory**: `compute_baz/`

Automated backazimuth estimation and analysis workflow for 6-DoF seismic data, with comprehensive data processing and visualization capabilities.

**Key Features**:
- Automated backazimuth estimation
- Data merging and aggregation tools
- Visualization and plotting utilities
- Parallel processing support
- Comprehensive logging and monitoring
