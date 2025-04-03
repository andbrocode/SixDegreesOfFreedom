This folder with scripts/notebooks contains demonstrations for the sixdegrees class.

The environment.yml file specifies the python environment with required packages: 

For setting up a conda environment run:
<code> conda env create -f environment.yml </code> 

1) fetch_data_Gring:
example of how to obtain data of G ring laser (together with co-located seismometer WET)

2) fetch_data_ROMY:
example of how to obtain data of ROMY ring laser (together with co-located seismometer FUR)

3) XBSPF_Analysis: 
demonstration for the six degrees-of-freedom (6DoF) station at Pinon Flat observatory in Southern California. It includes data retrieval and peforms analysis on the 6DoF data (e.g. backazimuth or phase velocity estimation)

4) XG_Analysis:
demonstration of the usage of vertical rotation rate G ring data as 4 DoF analysis (thus mainly Love waves)

5) XROMY_Analysis_demo:
demonstration of how to use ROMY ring laser data (this is not yet public data, therefore cannot be obtained using FSDN web services)
