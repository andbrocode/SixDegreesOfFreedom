This folder with scripts/notebooks contains demonstrations for the sixdegrees class.

The environment.yml file specifies the python environment with required packages: 

For setting up a conda environment run:
<code> conda env create -f environment.yml </code> 

1) fetch_data_Gring:
example of how to obtain data of G ring laser (together with co-located seismometer WET)

2) fetch_data_ROMY:
example of how to obtain data of ROMY ring laser (together with co-located seismometer FUR)

3) XBSPF_Analysis_demo1: 
demonstration for the six degrees-of-freedom (6DoF) station at Pinon Flat observatory in Southern California.
- data is loaded from local files using the discontinued PY.PFO seismometer
- analysis is performed on the 6DoF data (e.g. backazimuth or phase velocity estimation)

5) XBSPF_Analysis_demo2: 
demonstration for the six degrees-of-freedom (6DoF) station at Pinon Flat observatory in Southern California. 
- data is requested from FDSN services for a local M5.2 event using the new STS-2 seismometer (PY.PFOIX)
- analysis is performed on the 6DoF data (e.g. backazimuth or phase velocity estimation)

5) XG_Analysis:
demonstration of the usage of vertical rotation rate G ring data as 4 DoF analysis (thus mainly Love waves)

6) XROMY_Analysis_demo1:
demonstration of how to use ROMY ring laser data (this is not yet public data, therefore cannot be obtained using FSDN web services)
- loads data from local .mseed files

7) XROMY_Analysis_demo2:
demonstration of how to use ROMY ring laser data (this is not yet public data, therefore cannot be obtained using FSDN web services)
- loads data from internal server using FDSN clients

