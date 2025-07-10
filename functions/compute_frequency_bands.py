#!/bin/python3

def __compute_frequency_bands(fmin=1, fmax=20, fband_type='octave'):

    from numpy import sqrt, array
    
    f_lower, f_upper, f_centers = [], [], []
    fcenter = f_max

    if fband_type == "octave":
        while fcenter > f_min:
            f_lower.append(fcenter/(sqrt(sqrt(2.))))
            f_upper.append(fcenter*(sqrt(sqrt(2.))))
            f_centers.append(fcenter)

            fcenter = fcenter/(sqrt(2.))
    else:
        print(f"{fband_type} not known!")
        
    return array(f_lower), array(f_upper), array(f_centers)

## End of File