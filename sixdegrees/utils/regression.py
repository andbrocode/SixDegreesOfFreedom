"""
Utility functions for regression analysis.
Extracted from sixdegrees.py for independent use.
"""
import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from scipy import odr
from scipy.stats import pearsonr


def regression(x_data: np.ndarray, y_data: np.ndarray, method: str = "odr", 
               zero_intercept: bool = True, verbose: bool = False) -> Dict:
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
    
    Returns:
    --------
    Dict
        Dictionary containing:
        - slope: Regression slope
        - intercept: Y-intercept (0 if zero_intercept=True)
        - r_squared: R-squared value
        - method: Method used
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
    
    # Reshape for sklearn compatibility
    X = x_clean.reshape(-1, 1)
    y = y_clean
    
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
