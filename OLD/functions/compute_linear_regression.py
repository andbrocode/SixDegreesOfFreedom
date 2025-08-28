#!/bin/python3

def __compute_linear_regression(x_array, y_array, intercept_is_zero=False, method="sklearn"):
    
    """
    
    Performs a linear regression on the data. Optionally force intercept at the origin.
    
    ARGS:
        - x_array:   array of x values
        - y_array:   array of y values
        - intercept_is_zero:   fordes intercept to be zero
        - method:   method to use: [sklearn], lstsq, curve_fit
        
    OUTPUT:
        - slope:   (float)
        - intercept:   (float)
    
    
    >>> __compute_linear_regression(x_array, y_array, intercept_is_zero=False)
    
    """
    
    from numpy import sqrt, diag, mean, vstack, ones, array, newaxis, reshape
    from sklearn.linear_model import LinearRegression
    from scipy.stats import linregress
    from numpy.linalg import lstsq

    ## _________________________
    ## Method 1
    
    if method == "curve_fit":
        
        from scipy.optimize import curve_fit

        if intercept_is_zero:
            p0 = [mean(y_array)/mean(x_array)]
            def f(x, a):
                return a * x
        else:
            p0 = [mean(y_array)/mean(x_array), 0]
            def f(x, a, b):
                return a * x + b

        popt, pcov = curve_fit(f, y_array, x_array, p0=p0)

        slope = popt[0]
        slope_std = sqrt(diag(pcov))

        if intercept_is_zero:
            intercept = 0
        else:
            intercept = popt[1]
    
    ## _________________________
    ## Method 2
    
    elif method == "lstsq":
        
        if intercept_is_zero:
            A = array(x_array)[:, newaxis]
            a, res, rkn, _ = lstsq(A, y_array, rcond=None)
            slope, intercept = float(a), 0

        else:
            linreg = linregress(x_array, y_array)
            slope = linreg.slope
            intercept = linreg.intercept
            slope_std = linreg.stderr
        
    ## _________________________
    ## Method 3
    
    elif method == "sklearn":
        
        x, y = array(x_array).reshape(-1,1), y_array
        model = LinearRegression(fit_intercept=(not intercept_is_zero)).fit(x, y)
        r_sq = model.score(x, y)
        slope, intercept = float(model.coef_), float(model.intercept_)

    
#     print(slope, intercept)
    return float(slope), float(intercept)
    
## End of File