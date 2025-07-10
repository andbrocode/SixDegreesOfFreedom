#!/bin/python3

def __add_radial_and_transverse_channel(st_in, sta, baz):
    
    from obspy import Trace
    from obspy.signal.rotate import rotate_ne_rt

    st_acc = st_in.select(station=sta).copy()
    
    r_acc, t_acc = rotate_ne_rt(st_acc.select(channel='*N')[0].data, 
                                st_acc.select(channel='*E')[0].data,
                                baz
                                )
    
    
    tr_r = st_acc[0].copy()
    tr_r.data = r_acc
    tr_r.stats.channel = tr_r.stats.channel[:-1]+"R"
    
    tr_t = st_acc[0].copy()
    tr_t.data = t_acc
    tr_t.stats.channel = tr_t.stats.channel[:-1]+"T"
    
    st_in += tr_r
    st_in += tr_t
    
    return st_in
    
## End of File