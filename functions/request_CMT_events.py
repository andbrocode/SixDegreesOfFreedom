#!/bin/python3

def __request_CMT_events(starttime="2000-01-01", endtime="2023-01-01", outtype="csv", outfile="./gcmt_solutions.csv", unit="Nm"):
    
    '''
    Obtains event catalog from CGMT database and writes it to an output file.
    
    ARGUMGENTS:
    - starttime (str) = start date of period to filter catalog [ "2023-01-01" ]
    - endtime (str) = end date of period to filter catalog [ "2000-01-01" ] 
    - outtype (str) = defines type of output file ( ["csv"] or "QUAKEML" )
    - outfile (str) = specifies name of output file [ "./gcmt_solutions.csv" ]
    - unit (str) = specifies unit of moment tensor content ( ["Nm"] or "dyn cm" )
    
    EXAMPLE: 
    
    >>> __request_CMT_events(starttime="2000-01-01", endtime="2023-01-01", outtype="csv", outfile="./gcmt_solutions.csv", unit="Nm")
    
    '''
    
    from os.path import exists
    from obspy import read_events, UTCDateTime
    from numpy import asarray, where
    
    ## url path to the OLD gcmt catalog
    path_to_CMT_cat = 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk' #1976-2020

    ## url path to the NEW gcmt catalog    
    path_to_NEW_quick_CMT_cat = 'https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_QUICK/qcmt.ndk' #2021-


    if exists(outfile):
        print(f" -> file: {outfile} already exists!")
        return
    
    else:
        
        ## read the new ndk file (since 2021) using obspy
        try:
            CMT_cat = read_events( path_to_NEW_quick_CMT_cat )
        except:
            print(f" -> failed to read new CMT data: {path_to_NEW_quick_CMT_cat}!")
            return            
        
        ## read the old data base for events prior 2020 and join with CMT catalog
        if UTCDateTime(starttime).year <= 2020:
            try:
                CMT_cat_old = read_events( path_to_CMT_cat )
                CMT_cat += CMT_cat_old
            except:
                print(f" -> failed to read new CMT data: {path_to_NEW_quick_CMT_cat}!")
                return  
        
        ## filter for specified time period
        CMT_cat = CMT_cat.filter(f"time > {starttime}", f"time < {endtime}")
        
        if not outtype == "csv":
            
            ## write data as quakeml file
            CMT_cat.write(outfile, format="QUAKEML")
            print(f" -> store catalog as: {outfile}")
            
        else:
            ## extract information
            cmt_codes = asarray([ ev.get('resource_id').id.split('/')[2] for ev in CMT_cat ])
            cmt_dates = asarray([ ev.origins[0].time for ev in CMT_cat ])
            CMT_events_lat = asarray([ev.preferred_origin().latitude for ev in CMT_cat])
            CMT_events_lon = asarray([ev.preferred_origin().longitude for ev in CMT_cat])
            CMT_events_depth = asarray([ev.preferred_origin().depth/1000. for ev in CMT_cat])
            CMT_events_mag = asarray([ev.preferred_magnitude().mag for ev in CMT_cat])


            ## write information into csv file
            with open(outfile, 'w') as gcmtsol:

                ## write header info
                gcmtsol.write("evt,year,month,day,hour,minute,second,evlat,evlon,evdep,evmag,time_shift,half_duration,m_rr,m_tt,m_pp,m_rt,m_rp,m_tp\n")

                for evt2 in cmt_codes:
                    idx = where( cmt_codes == evt2 )[0]
                    origin_time = UTCDateTime( int( cmt_dates[idx][0].year ),
                                            int( cmt_dates[idx][0].month ),
                                            int( cmt_dates[idx][0].day ),
                                            int( cmt_dates[idx][0].hour ),
                                            int( cmt_dates[idx][0].minute ), 00 )

                    ## get focal mech info
                    m_rr = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_rr)*1e7 #from Nm to dyn cm
                    m_tt = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_tt)*1e7 #from Nm to dyn cm
                    m_pp = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_pp)*1e7 #from Nm to dyn cm
                    m_rt = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_rt)*1e7 #from Nm to dyn cm
                    m_rp = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_rp)*1e7 #from Nm to dyn cm
                    m_tp = (CMT_cat[idx[0]].preferred_focal_mechanism().moment_tensor.tensor.m_tp)*1e7 #from Nm to dyn cm

                    if unit is "Nm":
                        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp = m_rr*1e-7, m_tt*1e-7, m_pp*1e-7, m_rt*1e-7, m_rp*1e-7, m_tp*1e-7

                    ## get source location info
                    assert CMT_cat[idx[0]].origins[1].origin_type == 'centroid'
                    evlon = CMT_cat[idx[0]].origins[1].longitude
                    evlat = CMT_cat[idx[0]].origins[1].latitude
                    evdep = (CMT_cat[idx[0]].origins[1].depth/1000)
                    evmag = (CMT_cat[idx[0]].magnitudes[0].mag)

                    time_shift=0  
                    half_duration = (CMT_cat[idx[0]].focal_mechanisms[0].moment_tensor.source_time_function.duration)/2

                    ## write info to csv file
                    gcmtsol.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    evt2,cmt_dates[idx][0].year,cmt_dates[idx][0].month,cmt_dates[idx][0].day,cmt_dates[idx][0].hour,cmt_dates[idx][0].minute,
                    cmt_dates[idx][0].second,evlat,evlon,evdep,evmag,time_shift,half_duration,m_rr,m_tt,m_pp,m_rt,m_rp,m_tp))
            
            print(f" -> store catalog as: {outfile}")

## End of File