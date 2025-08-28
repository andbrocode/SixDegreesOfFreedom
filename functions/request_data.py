#!/bin/python3

def __request_data(seed, tbeg, tend, bulk_download=True, translation_type="ACC"):

    from obspy.clients.fdsn import Client

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
    except:
        print(" -> Failed to load waveforms!")
        waveform = None

    # adjust channel names
    if cha[1] == "J" and waveform is not None:

        waveform.remove_sensitivity(inventory=inventory)
        print(" -> sensitivity removed!")

        # for tr in waveform:
        #     if tr.stats.channel[-1] == "1":
        #         tr.stats.channel = str(tr.stats.channel).replace("1","E")
        #     elif tr.stats.channel[-1] == "2":
        #         tr.stats.channel = str(tr.stats.channel).replace("2","N")
        #     elif tr.stats.channel[-1] == "3":
        #         tr.stats.channel = str(tr.stats.channel).replace("3","Z")

    # adjust channel names
    elif cha[1] == "H" and waveform is not None:
        waveform.remove_response(inventory=inventory, output=translation_type, plot=False)
        print(" -> response removed!")

        # waveform.remove_sensitivity(inventory=inventory)
        # print(" -> sensitivity removed!")

        # if translation_type == "ACC":
        #     print(" -> differentiate")
        #     waveform.differentiate()

        # for tr in waveform:
        #     if tr.stats.channel[-1] == "1":
        #         tr.stats.channel = str(tr.stats.channel).replace("1","N")
        #     if tr.stats.channel[-1] == "2":
        #         tr.stats.channel = str(tr.stats.channel).replace("2","E")

    try:
        waveform.rotate(method="->ZNE", inventory=inventory)
    except:
        print(" -> failed to rotate to ZNE")

    waveform = waveform.trim(tbeg, tend)

    return waveform, inventory

# End of File