"""
Utility functions for regression analysis.
Extracted from sixdegrees.py for independent use.
"""
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from scipy import odr
from scipy.stats import pearsonr


def mad(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculate Median Absolute Deviation (MAD) using nan-aware functions.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    axis : int, optional
        Axis along which to compute MAD. If None, compute over flattened array.
    
    Returns:
    --------
    np.ndarray
        MAD value(s)
    """
    median = np.nanmedian(data, axis=axis, keepdims=True) if axis is not None else np.nanmedian(data)
    return np.nanmedian(np.abs(data - median), axis=axis)


def regression(x_data: np.ndarray, y_data: np.ndarray, method: str = "odr", 
               zero_intercept: bool = True, verbose: bool = False,
               bootstrap: Optional[Dict] = None) -> Dict:
    """
    Perform regression analysis using various methods.
    
    Parameters:
    -----------
    x_data : np.ndarray
        Input data (e.g., rotation data)
    y_data : np.ndarray
        Target data (e.g., translation data)
    method : str, optional
        Regression method ('odr', 'ransac', 'theilsen', 'ols'), by default 'odr'
    zero_intercept : bool, optional
        Force intercept through zero if True, by default True
    verbose : bool, optional
        Print regression results if True, by default False
    bootstrap : dict, optional
        Bootstrap options dictionary. If None, bootstrap is disabled.
        Valid keys:
        - 'n_iterations': int, number of bootstrap iterations (default: 1000)
        - 'stat': str, statistic to use ('mean' or 'median', default: 'mean')
        - 'random_seed': int, random seed for reproducibility (default: 42)
        Example: bootstrap={'n_iterations': 2000, 'stat': 'median', 'random_seed': 123}
    
    Returns:
    --------
    Dict
        Dictionary containing:
        - slope: Regression slope (or mean/median if bootstrap enabled)
        - intercept: Y-intercept (0 if zero_intercept=True, or mean/median if bootstrap enabled)
        - r_squared: R-squared value
        - method: Method used
        - slope_std or slope_mad: Standard deviation or MAD of slope (if bootstrap enabled)
        - intercept_std or intercept_mad: Standard deviation or MAD of intercept (if bootstrap enabled)
    """
    # Validate inputs
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    if len(x_data) < 2:
        raise ValueError("Need at least 2 data points for regression")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_clean = x_data[valid_mask]
    y_clean = y_data[valid_mask]
    
    if len(x_clean) < 2:
        raise ValueError("Not enough valid data points after removing NaN values")
    
    # Parse bootstrap options
    bootstrap_enabled = bootstrap is not None
    if bootstrap_enabled:
        if not isinstance(bootstrap, dict):
            raise TypeError("bootstrap must be a dictionary or None")
        
        n_bootstrap = bootstrap.get('n_iterations', 100)
        bootstrap_stat = bootstrap.get('stat', 'mean')
        random_seed = bootstrap.get('random_seed', 42)
        
        # Validate bootstrap parameters
        if bootstrap_stat not in ['mean', 'median']:
            raise ValueError(f"bootstrap['stat'] must be 'mean' or 'median', got '{bootstrap_stat}'")
        if n_bootstrap < 1:
            raise ValueError("bootstrap['n_iterations'] must be at least 1")
    
    # Reshape for sklearn compatibility
    X = x_clean.reshape(-1, 1)
    y = y_clean
    
    # Bootstrap resampling if enabled
    if bootstrap_enabled:
        slopes = []
        intercepts = []
        r_squareds = []
        
        np.random.seed(random_seed)  # For reproducibility
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(x_clean), size=len(x_clean), replace=True)
            x_boot = x_clean[indices]
            y_boot = y_clean[indices]
            
            # Perform regression on bootstrap sample
            try:
                boot_result = regression(
                    x_boot, y_boot, method=method, 
                    zero_intercept=zero_intercept, verbose=False,
                    bootstrap=None  # Prevent recursive bootstrapping
                )
                
                # Collect all results, including NaN values (will be handled by nanmean/nanmedian)
                slopes.append(boot_result['slope'])
                intercepts.append(boot_result['intercept'])
                r_squareds.append(boot_result['r_squared'])
            
            except Exception:
                # Skip failed bootstrap iterations by appending NaN
                slopes.append(np.nan)
                intercepts.append(np.nan)
                r_squareds.append(np.nan)
        
        slopes = np.array(slopes)
        intercepts = np.array(intercepts)
        r_squareds = np.array(r_squareds)
        
        # Calculate statistics based on bootstrap_stat using nan-aware functions
        if bootstrap_stat == 'mean':
            result = {
                'slope': np.nanmean(slopes),
                'intercept': np.nanmean(intercepts) if not zero_intercept else 0.0,
                'r_squared': np.nanmean(r_squareds),
                'method': method.lower(),
                'slope_dev': np.nanstd(slopes, ddof=1),
                'intercept_dev': np.nanstd(intercepts, ddof=1) if not zero_intercept else 0.0,
            }
        else:  # median
            result = {
                'slope': np.nanmedian(slopes),
                'intercept': np.nanmedian(intercepts) if not zero_intercept else 0.0,
                'r_squared': np.nanmedian(r_squareds),
                'method': method.lower(),
                'slope_dev': mad(slopes),
                'intercept_dev': mad(intercepts) if not zero_intercept else 0.0,
            }
        
        if verbose:
            stat_name = 'mean±std' if bootstrap_stat == 'mean' else 'median±MAD'
            uncertainty = result.get('slope_std', result.get('slope_mad', 0))
            print(f"Bootstrap Results ({stat_name}): "
                  f"slope={result['slope']:.6f}±{uncertainty:.6f}, "
                  f"R²={result['r_squared']:.4f}")
        
        return result
    
    # Initialize results
    result = {
        'slope': np.nan,
        'intercept': 0.0 if zero_intercept else np.nan,
        'r_squared': np.nan,
        'method': method.lower()
    }
    
    try:
        if method.lower() == "odr":
            # Orthogonal Distance Regression
            def linear_func(B, x):
                if zero_intercept:
                    return B[0] * x
                else:
                    return B[0] * x + B[1]
            
            # Create ODR model
            model = odr.Model(linear_func)
            
            # Estimate uncertainties
            sx = np.std(x_clean) * np.ones_like(x_clean)
            sy = np.std(y_clean) * np.ones_like(y_clean)
            
            # Create ODR data object
            data = odr.RealData(x_clean, y_clean, sx=sx, sy=sy)
            
            # Set initial parameters
            if zero_intercept:
                beta0 = [np.mean(y_clean) / np.mean(x_clean)]
            else:
                beta0 = [np.mean(y_clean) / np.mean(x_clean), 0.0]
            
            # Fit model
            odr_obj = odr.ODR(data, model, beta0=beta0)
            output = odr_obj.run()
            
            result['slope'] = output.beta[0]
            if not zero_intercept:
                result['intercept'] = output.beta[1]
            
            # Calculate R-squared
            if zero_intercept:
                y_pred = output.beta[0] * x_clean
            else:
                y_pred = output.beta[0] * x_clean + output.beta[1]
            
            # Calculate R-squared
            r, _ = pearsonr(x_clean, y_clean)
            result['r_squared'] = r**2

            if verbose:
                print(f"ODR Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
        
        elif method.lower() == "ransac":
            # RANSAC regression
            try:
                model = RANSACRegressor(
                    estimator=LinearRegression(fit_intercept=not zero_intercept),
                    random_state=42,
                    max_trials=1000
                ).fit(X, y)
            except TypeError:
                # Fallback for older sklearn versions
                model = RANSACRegressor(
                    base_estimator=LinearRegression(fit_intercept=not zero_intercept),
                    random_state=42,
                    max_trials=1000
                ).fit(X, y)
            
            result['slope'] = model.estimator_.coef_[0]
            if not zero_intercept:
                result['intercept'] = model.estimator_.intercept_
            result['r_squared'] = model.score(X, y)
            
            if verbose:
                print(f"RANSAC Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
        
        elif method.lower() == "theilsen":
            # Theil-Sen regression
            model = TheilSenRegressor(fit_intercept=not zero_intercept, random_state=42).fit(X, y)
            
            result['slope'] = model.coef_[0]
            if not zero_intercept:
                result['intercept'] = model.intercept_
            result['r_squared'] = model.score(X, y)
            
            if verbose:
                print(f"Theil-Sen Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
        
        elif method.lower() == "ols":
            # Ordinary Least Squares
            model = LinearRegression(fit_intercept=not zero_intercept).fit(X, y)
            
            result['slope'] = model.coef_[0]
            if not zero_intercept:
                result['intercept'] = model.intercept_
            result['r_squared'] = model.score(X, y)
            
            if verbose:
                print(f"OLS Results: slope={result['slope']:.6f}, R²={result['r_squared']:.4f}")
        
        else:
            raise ValueError(f"Invalid method: {method}. Use 'odr', 'ransac', 'theilsen', or 'ols'")
    
    except Exception as e:
        if verbose:
            print(f"Regression failed: {str(e)}")
        # Return NaN values if regression fails
        result['slope'] = np.nan
        result['intercept'] = np.nan if not zero_intercept else 0.0
        result['r_squared'] = np.nan
    
    return result
