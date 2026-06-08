"""
Functions for requesting seismic data from FDSN web services.
"""
from obspy.clients.fdsn import Client
from obspy import Stream

def request_data(seed, tbeg, tend, bulk_download=True, translation_type="ACC"):
    """
    Request seismic data from FDSN web services.
    
    Args:
        seed (str): Seed ID in format "NET.STA.LOC.CHA"
        tbeg (UTCDateTime): Start time for data request
        tend (UTCDateTime): End time for data request
        bulk_download (bool): Whether to use bulk download method
        translation_type (str): Output type for response removal ("ACC", "VEL", "DISP")
        
    Returns:
        tuple: (waveform, inventory) where:
            - waveform (Stream): ObsPy Stream containing the requested data
            - inventory (Inventory): Station metadata
    """
    client = Client("IRIS")

    net, sta, loc, cha = seed.split(".")

    # querry inventory data
    try:
        inventory = client.get_stations(network=net,
                                    station=sta,
                                    location=loc,
                                    starttime=tbeg-60,
                                    endtime=tend+60,
                                    level="response",
                                    )
    except:
        print(" -> Failed to load inventory!")
        inventory = None

    # querry waveform data
    try:
        if bulk_download:
            bulk = [(net, sta, loc, cha, tbeg-60, tend+60)]
            waveform = client.get_waveforms_bulk(bulk, attach_response=True)
        else:
            waveform = client.get_waveforms(network=net,
                                       station=sta,
                                       location=loc,
                                       channel=cha, 
                                       starttime=tbeg-60,
                                       endtime=tend+60,
                                       attach_response=False
                                       )
    except Exception as E:
        print(E)
        print(f" -> getting waveforms failed for {net}.{sta}.{loc}.{cha} ...")
        waveform = None

    # adjust channel names
    if cha[1] == "J" and waveform is not None:
        waveform.remove_sensitivity(inventory=inventory)
        print(" -> sensitivity removed!")

    # adjust channel names
    elif cha[1] == "H" and waveform is not None:
        waveform.remove_response(inventory=inventory, output=translation_type, plot=False)
        print(" -> response removed!")

    try:
        if waveform is not None:
            waveform.rotate(method="->ZNE", inventory=inventory)
            print(" -> rotated to ZNE")
    except:
        print(" -> failed to rotate to ZNE")

    if waveform is not None:
        waveform = waveform.trim(tbeg, tend)

    return waveform, inventory
