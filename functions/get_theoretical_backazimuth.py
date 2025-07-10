#!/bin/python3

def __get_theoretical_backazimuth(station_lat, station_lon, event_time=None, time_offset=20, event_obj=None, fdsn_client="USGS"):

    from obspy.clients.fdsn import Client
    from obspy.geodetics.base import gps2dist_azimuth

    ## get event if not provided
    if event_obj is None and event_time is None:
        print(" -> provide event_time or event_obj!!!")
        return 0,0,0

    elif event_obj is None and event_time is not None:
        events = Client(fdsn_client).get_events(starttime=event_time-time_offset, endtime=event_time+time_offset)
        if len(events) > 1:
            print(f" -> {len(events)} events found!!!")
            print(events)

        event = events[0]
    elif event_obj is not None and event_time is None:
        event = event_obj

    ## event location from event info
    source_latitude = event.origins[0].latitude
    source_longitude = event.origins[0].longitude


    dist, az, baz = gps2dist_azimuth(
                                    source_latitude, source_longitude,
                                    station_lat, station_lon,
                                    )


    return baz, az, dist

## End of File