"""
Functions for printing deviation analysis summaries.
"""
import numpy as np

def print_deviation_summary(analysis_results):
    """
    Print detailed summary of deviation analysis results.
    
    Parameters:
    -----------
    analysis_results : dict
        Results from plot_backazimuth_deviation_analysis containing:
            - deviations: Dictionary of deviation data per wave type
            - theoretical_baz: Theoretical backazimuth value
            - center_frequencies: Array of center frequencies
            - bin_info: Information about histogram binning
    """
    deviations = analysis_results['deviations']
    theoretical_baz = analysis_results['theoretical_baz']
    center_freqs = analysis_results['center_frequencies']
    
    print("="*60)
    print("BACKAZIMUTH DEVIATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Theoretical Backazimuth: {theoretical_baz:.1f}°")
    print(f"Total Frequency Bands: {len(center_freqs)}")
    if 'bin_info' in analysis_results:
        print(f"Histogram binning: {analysis_results['bin_info']}")
    print()
    
    for wave_type, data in deviations.items():
        print(f"{wave_type.upper()} WAVES:")
        print(f"  Valid estimates: {len(data['deviation'])}/{len(center_freqs)}")
        print(f"  Frequency range: {data['frequencies'].min():.3f} - {data['frequencies'].max():.3f} Hz")
        print(f"  Mean deviation: {data['mean_deviation']:.2f}°")
        print(f"  Std deviation: {data['std_deviation']:.2f}°")
        print(f"  RMS deviation: {data['rms_deviation']:.2f}°")
        print(f"  Median deviation: {np.median(data['deviation']):.2f}°")
        print(f"  Max absolute deviation: {np.max(np.abs(data['deviation'])):.2f}°")
        print(f"  95th percentile: {np.percentile(np.abs(data['deviation']), 95):.2f}°")
        print()
