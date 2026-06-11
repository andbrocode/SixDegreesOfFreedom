"""
Tests for compute_dispersion_curve and dispersion plot functions.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.rotate import rotate_ne_rt

from sixdegrees.plots.plot_dispersion_curves import plot_dispersion_curves
from sixdegrees.plots.plot_dispersion_traces import plot_dispersion_traces
from sixdegrees.sixdegrees import sixdegrees


def _expected_window_count(n_samples, sampling_rate, time_window_sec, overlap):
    win_samples = int(time_window_sec * sampling_rate)
    overlap_samples = int(win_samples * overlap)
    step = win_samples - overlap_samples
    return int((n_samples - win_samples) / step) + 1


def _expected_window_times(n_samples, sampling_rate, time_window_sec, overlap):
    win_samples = int(time_window_sec * sampling_rate)
    overlap_samples = int(win_samples * overlap)
    step = win_samples - overlap_samples
    n_windows = _expected_window_count(n_samples, sampling_rate, time_window_sec, overlap)
    return np.array([(i * step + win_samples / 2) / sampling_rate for i in range(n_windows)])


@pytest.fixture
def dispersion_sd():
    """sixdegrees instance with synthetic Love-wave-like 6-component data."""
    sr = 20.0
    duration = 300.0
    npts = int(duration * sr)
    t = np.arange(npts) / sr
    freq = 0.2
    baz = 45.0
    velocity = 3000.0

    acc_n = np.sin(2 * np.pi * freq * t)
    acc_e = np.sin(2 * np.pi * freq * t + 0.3)
    _, acc_t = rotate_ne_rt(acc_n, acc_e, baz)
    rot_z = acc_t / (2 * velocity)

    st_rot = Stream()
    for ch, data in [
        ("BJZ", rot_z),
        ("BJN", np.sin(2 * np.pi * freq * t + 0.1)),
        ("BJE", np.sin(2 * np.pi * freq * t + 0.2)),
    ]:
        tr = Trace(data=data.astype(np.float64))
        tr.stats.sampling_rate = sr
        tr.stats.starttime = UTCDateTime(2024, 1, 1)
        tr.stats.network = "XX"
        tr.stats.station = "TEST"
        tr.stats.location = ""
        tr.stats.channel = ch
        st_rot += tr

    st_acc = Stream()
    for ch, data in [("BHZ", acc_n), ("BHN", acc_n), ("BHE", acc_e)]:
        tr = Trace(data=data.astype(np.float64))
        tr.stats.sampling_rate = sr
        tr.stats.starttime = UTCDateTime(2024, 1, 1)
        tr.stats.network = "XX"
        tr.stats.station = "TEST"
        tr.stats.location = ""
        tr.stats.channel = ch
        st_acc += tr

    sd = sixdegrees(
        {
            "tbeg": "2024-01-01",
            "tend": "2024-01-01T00:05:00",
            "fmin": 0.1,
            "fmax": 0.5,
            "fdsn_client_rot": "IRIS",
            "fdsn_client_tra": "IRIS",
            "rot_seed": "XX.TEST..BJZ",
            "tra_seed": "XX.TEST..BHZ",
        }
    )
    sd.st0 = st_rot + st_acc
    sd.st = sd.st0.copy()
    sd.sampling_rate = sr
    sd.baz_theo = baz
    return sd


@pytest.fixture
def love_dispersion_results(dispersion_sd):
    return dispersion_sd.compute_dispersion_curve(
        wave_type="love",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        window_factor=1.0,
        time_window_overlap=0.5,
        use_theoretical_baz=False,
        cc_threshold=0.0,
        cc_method="max",
        baz_step=15,
        verbose=False,
        n_jobs=1,
    )


def test_compute_dispersion_curve_structure(love_dispersion_results):
    results = love_dispersion_results

    assert results["wave_type"] == "love"
    assert len(results["frequency_bands"]) > 0
    assert len(results["frequencies"]) == len(results["frequency_bands"])
    assert len(results["velocities"]) == len(results["frequency_bands"])
    assert len(results["uncertainties"]) == len(results["frequency_bands"])

    band = results["frequency_bands"][0]
    for key in (
        "f_lower",
        "f_upper",
        "f_center",
        "time_window",
        "filtered_rot",
        "filtered_acc",
        "backazimuths",
        "velocities",
        "ccoefs",
        "times",
        "kde_peak_velocity",
        "kde_deviation",
    ):
        assert key in band


def test_compute_dispersion_curve_time_windows_per_band(dispersion_sd, love_dispersion_results):
    window_factor = 1.0

    for band in love_dispersion_results["frequency_bands"]:
        fc = band["f_center"]
        expected_tw = max(window_factor / fc, 1.0)
        assert band["time_window"] == pytest.approx(expected_tw, rel=1e-6)


def test_compute_dispersion_curve_window_arrays_aligned(dispersion_sd, love_dispersion_results):
    sr = dispersion_sd.sampling_rate
    overlap = 0.5

    for band in love_dispersion_results["frequency_bands"]:
        n_samples = len(band["filtered_rot"][0].data)
        tw = band["time_window"]
        expected_n = _expected_window_count(n_samples, sr, tw, overlap)
        expected_times = _expected_window_times(n_samples, sr, tw, overlap)

        assert len(band["backazimuths"]) == expected_n
        assert len(band["velocities"]) == expected_n
        assert len(band["times"]) == expected_n
        assert len(band["ccoefs"]) == expected_n
        assert np.allclose(band["times"], expected_times, rtol=0, atol=1e-9)

        if expected_n > 1:
            step = expected_times[1] - expected_times[0]
            assert np.allclose(np.diff(band["times"]), step, rtol=0, atol=1e-9)


def test_compute_dispersion_curve_higher_frequency_shorter_windows(dispersion_sd):
    results = dispersion_sd.compute_dispersion_curve(
        wave_type="love",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        window_factor=1.0,
        use_theoretical_baz=False,
        cc_threshold=0.0,
        baz_step=15,
        verbose=False,
        n_jobs=1,
    )

    bands = results["frequency_bands"]
    assert bands[0]["f_center"] < bands[-1]["f_center"]
    assert bands[0]["time_window"] > bands[-1]["time_window"]


def test_compute_dispersion_curve_rayleigh(dispersion_sd):
    results = dispersion_sd.compute_dispersion_curve(
        wave_type="rayleigh",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        window_factor=1.0,
        use_theoretical_baz=False,
        cc_threshold=0.0,
        baz_step=15,
        verbose=False,
        n_jobs=1,
    )

    assert results["wave_type"] == "rayleigh"
    assert len(results["frequency_bands"]) > 0
    band = results["frequency_bands"][0]
    assert len(band["backazimuths"]) == len(band["velocities"]) == len(band["times"])


def test_compute_dispersion_curve_rejects_tangent(dispersion_sd):
    with pytest.raises(ValueError, match="tangent"):
        dispersion_sd.compute_dispersion_curve(wave_type="tangent", fmin=0.1, fmax=0.5)


def test_compute_dispersion_curve_velocity_threshold(dispersion_sd):
    kwargs = dict(
        wave_type="love",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        window_factor=1.0,
        time_window_overlap=0.5,
        use_theoretical_baz=False,
        cc_threshold=0.0,
        cc_method="mid",
        baz_step=15,
        verbose=False,
        n_jobs=1,
    )

    results_no_limit = dispersion_sd.compute_dispersion_curve(**kwargs, velocity_threshold=None)
    results_with_limit = dispersion_sd.compute_dispersion_curve(**kwargs, velocity_threshold=3000.0)

    assert results_no_limit["parameters"]["velocity_threshold"] is None
    assert results_with_limit["parameters"]["velocity_threshold"] == 3000.0

    for band_no, band_lim in zip(
        results_no_limit["frequency_bands"],
        results_with_limit["frequency_bands"],
    ):
        np.testing.assert_array_equal(band_no["velocities"], band_lim["velocities"])
        kde_lim = band_lim["kde_peak_velocity"]
        if not np.isnan(kde_lim):
            assert kde_lim <= 3000.0

def test_plot_dispersion_curves(love_dispersion_results):
    fig = plot_dispersion_curves(
        dispersion_results=love_dispersion_results,
        show_errors=False,
        xlog=True,
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_dispersion_curves_velocity_stat(love_dispersion_results):
    fig_kde = plot_dispersion_curves(
        dispersion_results=love_dispersion_results,
        velocity_stat="kde_peak",
        show_errors=False,
    )
    fig_med = plot_dispersion_curves(
        dispersion_results=love_dispersion_results,
        velocity_stat="median",
        show_errors=False,
    )
    assert isinstance(fig_kde, Figure)
    assert isinstance(fig_med, Figure)
    with pytest.raises(ValueError, match="velocity_stat"):
        plot_dispersion_curves(
            dispersion_results=love_dispersion_results,
            velocity_stat="mean",
        )


def test_plot_dispersion_curves_love_and_rayleigh(dispersion_sd, love_dispersion_results):
    rayleigh = dispersion_sd.compute_dispersion_curve(
        wave_type="rayleigh",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        use_theoretical_baz=False,
        cc_threshold=0.0,
        baz_step=15,
        verbose=False,
        n_jobs=1,
    )
    fig = plot_dispersion_curves(
        love_results=love_dispersion_results,
        rayleigh_results=rayleigh,
        show_errors=False,
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_dispersion_traces(love_dispersion_results):
    fig = plot_dispersion_traces(
        love_dispersion_results,
        data_type="acceleration",
        unitscale="nano",
    )
    assert isinstance(fig, Figure)
    n_bands = len(love_dispersion_results["frequency_bands"])
    assert len(fig.axes) == n_bands


def test_plot_dispersion_traces_velocity_stat(love_dispersion_results):
    fig_med = plot_dispersion_traces(
        love_dispersion_results,
        velocity_stat="median",
        data_type="acceleration",
        unitscale="nano",
    )
    fig_kde = plot_dispersion_traces(
        love_dispersion_results,
        velocity_stat="kde_peak",
        data_type="acceleration",
        unitscale="nano",
    )
    assert isinstance(fig_med, Figure)
    assert isinstance(fig_kde, Figure)


def test_plot_dispersion_traces_requires_frequency_bands():
    with pytest.raises(ValueError, match="frequency_bands"):
        plot_dispersion_traces({"wave_type": "love"})


@pytest.mark.xfail(strict=True, reason="use_theoretical_baz yields no frequency bands (backazimuth/window count mismatch)")
def test_compute_dispersion_curve_theoretical_baz_produces_results(dispersion_sd):
    results = dispersion_sd.compute_dispersion_curve(
        wave_type="love",
        fmin=0.1,
        fmax=0.5,
        octave_fraction=3,
        window_factor=1.0,
        use_theoretical_baz=True,
        cc_threshold=0.0,
        verbose=False,
        n_jobs=1,
    )
    assert len(results["frequency_bands"]) > 0
