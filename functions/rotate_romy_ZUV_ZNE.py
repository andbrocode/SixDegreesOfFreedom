def __rotate_romy_ZUV_ZNE(st, inv, keep_z=False):

    from obspy.signal.rotate import rotate2zne

    ori_z = inv.get_orientation("BW.ROMY.10.BJZ")
    ori_u = inv.get_orientation("BW.ROMY..BJU")
    ori_v = inv.get_orientation("BW.ROMY..BJV")

    romy_z = st.select(channel="*Z")[0].data
    romy_u = st.select(channel="*U")[0].data
    romy_v = st.select(channel="*V")[0].data


    romy_z, romy_n, romy_e =rotate2zne(
                                       romy_z, ori_z['azimuth'], ori_z['dip'],
                                       romy_u, ori_u['azimuth'], ori_u['dip'],
                                       romy_v, ori_v['azimuth'], ori_v['dip'],
                                       inverse=False
                                      )

    st_new = st.copy()

    if keep_z:
        st_new.select(channel="*Z")[0].data = st.select(channel="*Z")[0].data
    else:
        st_new.select(channel="*Z")[0].data = romy_z

    st_new.select(channel="*U")[0].data = romy_n
    st_new.select(channel="*V")[0].data = romy_e

    ch = st_new.select(channel="*U")[0].stats.channel[:2]

    st_new.select(channel="*U")[0].stats.channel = f"{ch}N"
    st_new.select(channel="*V")[0].stats.channel = f"{ch}E"

    return st_new