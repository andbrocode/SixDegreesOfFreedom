def __get_stream(tbeg, tend, status=False):

    from functions.request_data import __request_data
    from functions.compute_adr_pfo import __compute_adr_pfo
    from obspy import UTCDateTime

    tbeb, tend = UTCDateTime(tbeg), UTCDateTime(tend)

    try:
        ##load rotation
        bspf0, bspf_inv = __request_data("PY.BSPF..HJ*", tbeg-100, tend+100)
        bspf0 = bspf0.resample(40, no_filter=False);

        # bspf0 = bspf0.detrend("linear").taper(0.01).filter("lowpass", freq=18.0, corners=4, zerophase=True)
        # bspf0 = bspf0.decimate(5, no_filter=True) ## 200 -> 40 Hz
        # bspf0 = bspf0.decimate(2, no_filter=True) ## 40 -> 20 Hz
        # bspf0 = bspf0.trim(tbeg, tend)

        ## load translation
        if tbeg > UTCDateTime("2023-04-02"):
            pfo0, pfo_inv = __request_data("PY.PFOIX..HH*", tbeg-100, tend+100, translation_type="ACC")
            pfo0 = pfo0.resample(40, no_filter=False);

            # pfo0 = pfo0.detrend("linear").taper(0.01).filter("lowpass", freq=18.0, corners=4, zerophase=True)
            # pfo0 = pfo0.decimate(5, no_filter=True) ## 200 -> 40 Hz
            # pfo0 = pfo0.decimate(2, no_filter=True) ## 40 -> 20 Hz
            # pfo0 = pfo0.trim(tbeg, tend)

        else:
            pfo0, pfo_inv = __request_data("II.PFO.10.BH*", tbeg-100, tend+100, translation_type="ACC")
            pfo0 = pfo0.resample(40, no_filter=False);

            # pfo0 = pfo0.detrend("linear").taper(0.01).filter("lowpass", freq=18.0, corners=4, zerophase=True)
            # pfo0 = pfo0.decimate(2, no_filter=True) ## 40 -> 20 Hz
            # pfo0 = pfo0.trim(tbeg, tend])
    except:
        pass

    # merge to one stream
    st0 = bspf0.copy();
    st0 += pfo0.copy();


    ## ADR
    submask = "inner"
    try:
        if status:
            adr0, status = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=True)
        else:
            adr0 = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=False)

        for tr in adr0:
            tr.stats.location = "in"
        st0 += adr0.copy();
    except Exception as e:
        print(e)
        pass


    submask = "mid"
    try:
        if status:
            adr0, status = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=True)
        else:
            adr0 = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=False)

        for tr in adr0:
            tr.stats.location = "mi"
        st0 += adr0.copy();
    except Exception as e:
        print(e)
        pass 


    submask = "all"
    try:
        if status:
            adr0, status = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=True)
        else:
            adr0 = __compute_adr_pfo(tbeg-100, tend+100, submask=submask, status=False)

        for tr in adr0:
            tr.stats.location = "al"
        st0 += adr0.copy();
    except Exception as e:
        print(e)
        pass

    st0 = st0.resample(40, no_filter=True);

    st0 = st0.detrend("demean")
    st0 = st0.taper(0.01, type="cosine")

    st0 = st0.sort();

    st0 = st0.trim(tbeg, tend);

    print(st0)

    return st0