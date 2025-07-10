#!/bin/python3

def __compute_cc_for_fbands(tr1, tr2, fbands, plot=False):
    
    import matplotlib.pyplot as plt
    from numpy import array, corrcoef, correlate
    from obspy.signal.cross_correlation import correlate
    
    def max_number(l):
        abs_maxval = max(l,key=abs)
        maxval = max(l)
        minval = min(l)
        if maxval == abs_maxval:
            return maxval
        else:
            return minval
    
    f_centers = [(fu - fl) / 2 + fl for (fl, fu) in fbands]
    
    ccorrs, ccorrs_max = [], []
    for (fll, fuu) in fbands:

        tr01 = tr1.copy();
        tr02 = tr2.copy();

        tr01 = tr01.detrend("linear")
        tr02 = tr02.detrend("linear")

#         tr01 = tr01.normalize();
#         tr02 = tr02.normalize();
        
        tr01 = tr01.taper(0.1);
        tr02 = tr02.taper(0.1);

        tr01 = tr01.filter("bandpass", freqmin=fll, freqmax=fuu, corners=8, zerophase=True);
        tr02 = tr02.filter("bandpass", freqmin=fll, freqmax=fuu, corners=8, zerophase=True);
        
        tr01 = tr01.normalize();
        tr02 = tr02.normalize();

        cc1 = correlate(tr01.data, tr02.data, 0, demean=True, normalize='naive', method='fft')
        ccorrs_max.append(max_number(cc1))
        
     
        cc2 = corrcoef(tr01.data, tr02.data)[0][1]
        ccorrs.append(cc2)  
        
  
        if plot:
            tr01 = tr01.normalize();
            tr02 = tr02.normalize();            
            
            plt.figure()
            plt.plot(tr01.data)
            plt.plot(tr02.data)
            plt.title(f"{round(fll,3)}-{round(fuu,3)} Hz  | CCmax = {round(cc2,2)}")
            plt.show()
            
    return array(f_centers), array(ccorrs), array(ccorrs_max)


## End of File