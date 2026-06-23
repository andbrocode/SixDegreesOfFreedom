"""
Functions for plotting filtered traces from compute_dispersion_curve output.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from obspy.signal.rotate import rotate_ne_rt


def _format_freq_label(f_lower: float, f_upper: float) -> str:
    if f_lower < 0.1:
        fl_str = f"{f_lower:.3f}"
    elif f_lower < 1:
        fl_str = f"{f_lower:.2f}"
    else:
        fl_str = f"{f_lower:.1f}"
    if f_upper < 0.1:
        fu_str = f"{f_upper:.3f}"
    elif f_upper < 1:
        fu_str = f"{f_upper:.2f}"
    else:
        fu_str = f"{f_upper:.1f}"
    return f"{fl_str}-{fu_str} Hz"


def _translation_peak(band: Dict, wave_type: str) -> float:
    """Max |translation| trace amplitude for one band."""
    acc = band["filtered_acc"]
    if wave_type.lower() == "rayleigh":
        return float(np.max(np.abs(acc.select(channel="*Z")[0].data)))
    peaks = [
        float(np.max(np.abs(acc.select(channel=ch)[0].data)))
        for ch in ("*N", "*E")
        if len(acc.select(channel=ch))
    ]
    return max(peaks) if peaks else 1.0


def _global_translation_peak(reference_band: Dict, wave_type: str) -> float:
    """Max |translation| on reference band (e.g. broadband panel)."""
    return _translation_peak(reference_band, wave_type)


def _valid_per_window_velocities(band: Dict) -> np.ndarray:
    velocities = band.get("velocities", [])
    if velocities is None or len(velocities) == 0:
        return np.array([], dtype=float)
    valid = np.asarray(velocities, dtype=float)
    return valid[~np.isnan(valid)]


def _band_velocity_value(band: Dict, velocity_stat: str = "median") -> Tuple[float, bool]:
    """Return band velocity (m/s) and whether it is valid."""
    if velocity_stat not in ("kde_peak", "median"):
        raise ValueError("velocity_stat must be 'kde_peak' or 'median'")

    if velocity_stat == "median":
        valid = _valid_per_window_velocities(band)
        if len(valid) > 0:
            return float(np.nanmedian(valid)), True
        return np.nan, False

    peak = band.get("kde_peak_velocity", np.nan)
    try:
        peak = float(peak)
    except (TypeError, ValueError):
        peak = np.nan
    return peak, not np.isnan(peak)


def _band_velocity_uncertainty(band: Dict, velocity_stat: str = "median") -> float:
    """Uncertainty for annotation/error bars: KDE deviation or per-window std."""
    if velocity_stat == "kde_peak":
        return float(band.get("kde_deviation", np.nan))

    valid = _valid_per_window_velocities(band)
    if len(valid) > 1:
        return float(np.nanstd(valid))
    return np.nan


def _band_velocity_for_scaling(band: Dict, velocity_stat: str = "median") -> Tuple[float, bool]:
    return _band_velocity_value(band, velocity_stat=velocity_stat)


def _get_unit_scales(unitscale: str, data_type: str) -> Tuple[float, float, str, str, str, str]:
    if data_type.lower() == "velocity":
        if unitscale == "nano":
            return 1e9, 1e9, "nm/s", "nrad", "v", "r"
        if unitscale == "micro":
            return 1e6, 1e6, "µm/s", "µrad", "v", "r"
        raise ValueError("unitscale must be 'nano' or 'micro'")
    if unitscale == "nano":
        return 1e9, 1e9, "nm/s²", "nrad/s", "a", r"$\dot{r}$"
    if unitscale == "micro":
        return 1e6, 1e6, "µm/s²", "µrad/s", "a", r"$\dot{r}$"
    raise ValueError("unitscale must be 'nano' or 'micro'")


def _plot_dispersion_trace_panels(
    frequency_bands: List[Dict],
    wave_type: str,
    *,
    data_type: str = "velocity",
    unitscale: str = "nano",
    title: Optional[str] = None,
    baz_theoretical: Optional[float] = None,
    show_errors: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    global_scaling: bool = False,
    global_scaling_reference_band: Optional[Dict] = None,
    broadband_label: Optional[str] = None,
    font: int = 12,
    lw: float = 0.8,
    ylim_pad: float = 1.05,
    text_y_pos: float = 0.8,
    suptitle_y: float = 0.91,
    tra_color: str = "black",
    rot_color: str = "red",
    text_bbox: Optional[Dict] = None,
    velocity_stat: str = "median",
) -> plt.Figure:
    """Shared plotting loop: translation and rotation on one axis per panel."""
    if text_bbox is None:
        text_bbox = dict(facecolor="white", alpha=0.8, edgecolor="none")

    n_bands = len(frequency_bands)
    if n_bands == 0:
        raise ValueError("No frequency bands to plot")

    acc_scaling, rot_scaling, tra_unit, rot_unit, tra_label_prefix, rot_label_prefix = (
        _get_unit_scales(unitscale, data_type)
    )
    scale_unit = "m/s" if data_type.lower() == "velocity" else "m/s²"

    if global_scaling:
        ref = global_scaling_reference_band or frequency_bands[-1]
        global_tra_peak = _global_translation_peak(ref, wave_type)
        if global_tra_peak <= 0:
            global_tra_peak = 1.0
    else:
        global_tra_peak = None

    if figsize is None:
        figsize = (15, 1.3 * n_bands)

    fig, axes = plt.subplots(n_bands, 1, figsize=figsize, sharex=True)
    if n_bands == 1:
        axes = [axes]
    plt.subplots_adjust(hspace=0.0)

    last_acc_filtered = None
    has_broadband_panel = broadband_label is not None

    for i, band in enumerate(frequency_bands):
        ax = axes[i]
        for spine in ("bottom", "top", "right", "left"):
            ax.spines[spine].set_visible(False)

        rot_filtered = band["filtered_rot"]
        acc_filtered = band["filtered_acc"]
        last_acc_filtered = acc_filtered
        times = rot_filtered[0].times()

        baz_used = band.get("baz_used")
        if baz_used is None or (isinstance(baz_used, float) and np.isnan(baz_used)):
            backazimuths = band.get("backazimuths", [])
            if len(backazimuths) > 0:
                baz_used = float(np.nanmedian(np.asarray(backazimuths, dtype=float)))
            else:
                raise ValueError(f"No valid backazimuth for panel {i + 1}")
        else:
            baz_used = float(baz_used)

        velocity_for_scaling, velocity_valid = _band_velocity_for_scaling(
            band, velocity_stat=velocity_stat
        )
        velocity_deviation = _band_velocity_uncertainty(band, velocity_stat=velocity_stat)

        if wave_type.lower() == "rayleigh":
            tra = acc_filtered.select(channel="*Z")[0].data
            _, rot = rotate_ne_rt(
                rot_filtered.select(channel="*N")[0].data,
                rot_filtered.select(channel="*E")[0].data,
                baz_used,
            )
        elif wave_type.lower() == "love":
            rot = rot_filtered.select(channel="*Z")[0].data
            _, tra = rotate_ne_rt(
                acc_filtered.select(channel="*N")[0].data,
                acc_filtered.select(channel="*E")[0].data,
                baz_used,
            )
        else:
            raise ValueError(f"Unsupported wave_type: {wave_type}")

        if global_scaling:
            tra_scaled = tra / global_tra_peak * acc_scaling
        else:
            tra_scaled = tra * acc_scaling

        rot_scale = velocity_for_scaling if velocity_valid else 1.0
        if wave_type.lower() == "love":
            rot_scaled = rot * rot_scaling * rot_scale * 2 # times two for velocity scaling (plotting only)
        elif wave_type.lower() == "rayleigh":
            rot_scaled = rot * rot_scaling * rot_scale

        ax.plot(times, tra_scaled, color=tra_color, lw=lw, label=tra_label_prefix)
        ax.plot(times, rot_scaled, color=rot_color, lw=lw, label=rot_label_prefix)

        tra_max = float(np.max(np.abs(tra_scaled))) if len(tra_scaled) else 0.0
        rot_max = float(np.max(np.abs(rot_scaled))) if len(rot_scaled) else 0.0

        if global_scaling:
            ax.set_ylim(-acc_scaling * ylim_pad, acc_scaling * ylim_pad)
        else:
            panel_max = max(tra_max, rot_max)
            if panel_max > 0:
                ax.set_ylim(-panel_max * ylim_pad, panel_max * ylim_pad)
            else:
                ax.set_ylim(-1, 1)

        ax.set_ylabel("")
        ax.set_yticks([])

        if velocity_valid:
            if show_errors and not np.isnan(velocity_deviation):
                scale_txt = (
                    f"velocity: {velocity_for_scaling:.0f}"
                    f"±{float(velocity_deviation):.0f} {scale_unit}"
                )
            else:
                if wave_type.lower() == "love":
                    scale_txt = f"velocity: {velocity_for_scaling:.0f} {scale_unit} (2x)"
                elif wave_type.lower() == "rayleigh":
                    scale_txt = f"velocity: {velocity_for_scaling:.0f} {scale_unit}"
        else:
            scale_txt = "no velocity found"

        ax.text(
            0.02, text_y_pos - 0.1, scale_txt,
            transform=ax.transAxes, fontsize=font - 2, va="center", ha="left", bbox=text_bbox,
        )
        ax.text(
            0.02, abs(1 - text_y_pos + 0.1), f"baz: {baz_used:.1f}°",
            transform=ax.transAxes, fontsize=font - 2, va="center", ha="left", bbox=text_bbox,
        )

        if i == n_bands - 1 and has_broadband_panel:
            ax.set_xlabel("Time (s)", fontsize=font)
            freq_label = broadband_label
        else:
            if i == n_bands - 1:
                ax.set_xlabel("Time (s)", fontsize=font)
            freq_label = _format_freq_label(band["f_lower"], band["f_upper"])

        ax.text(
            0.99, text_y_pos - 0.1, freq_label,
            transform=ax.transAxes, fontsize=font - 2, va="center", ha="right", bbox=text_bbox,
        )
        ax.text(
            0.99, abs(1 - text_y_pos + 0.1),
            f"peak {data_type[:3]}: {tra_max:.0f} {tra_unit}",
            transform=ax.transAxes, fontsize=font - 2, va="center", ha="right", bbox=text_bbox,
        )

    axes[-1].spines["bottom"].set_visible(True)

    if title is None and last_acc_filtered is not None:
        start_time = last_acc_filtered[0].stats.starttime
        title = f"{wave_type.capitalize()} waves"
        if baz_theoretical is not None:
            title += f" | BAz = {baz_theoretical:.1f}°"
        title += f" | {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"

    fig.suptitle(title, fontsize=font + 2, y=suptitle_y)
    return fig


def plot_dispersion_traces(
    dispersion_results: Dict,
    unitscale: str = "nano",
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    data_type: str = "acceleration",
    baz_theoretical: Optional[float] = None,
    show_errors: bool = False,
    global_scaling: bool = False,
    global_scaling_reference_band: Optional[Dict] = None,
    font: int = 12,
    lw: float = 0.8,
    ylim_pad: float = 1.05,
    text_y_pos: float = 0.8,
    suptitle_y: float = 0.91,
    tra_color: str = "black",
    rot_color: str = "red",
    text_bbox: Optional[Dict] = None,
    velocity_stat: str = "median",
) -> plt.Figure:
    """
    Plot filtered traces from compute_dispersion_curve output.

    Translation (black) and rotation (red) are drawn on a single axis per panel.
    Rotation amplitude is scaled by the band phase velocity (m/s) when available.

    Parameters
    ----------
    dispersion_results : dict
        Output from compute_dispersion_curve (must contain ``frequency_bands``).
    unitscale : str
        ``nano`` or ``micro`` display scaling.
    data_type : str
        ``velocity`` or ``acceleration``.
    velocity_stat : str
        ``median`` (per-window median) or ``kde_peak`` (KDE peak velocity).
    global_scaling : bool
        If True, normalize translation traces to the peak of
        ``global_scaling_reference_band`` (default: last frequency band).
    global_scaling_reference_band : dict, optional
        Band dict used as translation reference for ``global_scaling``.
    """
    if "frequency_bands" not in dispersion_results:
        raise ValueError("dispersion_results must contain 'frequency_bands' key")

    frequency_bands = dispersion_results["frequency_bands"]
    if len(frequency_bands) == 0:
        raise ValueError("No frequency bands found in dispersion_results")

    wave_type = dispersion_results.get("wave_type", "love")

    return _plot_dispersion_trace_panels(
        frequency_bands,
        wave_type,
        data_type=data_type,
        unitscale=unitscale,
        title=title,
        baz_theoretical=baz_theoretical,
        show_errors=show_errors,
        figsize=figsize,
        global_scaling=global_scaling,
        global_scaling_reference_band=global_scaling_reference_band,
        font=font,
        lw=lw,
        ylim_pad=ylim_pad,
        text_y_pos=text_y_pos,
        suptitle_y=suptitle_y,
        tra_color=tra_color,
        rot_color=rot_color,
        text_bbox=text_bbox,
        velocity_stat=velocity_stat,
    )


def plot_dispersion_traces_with_broadband(
    dispersion_results: Dict,
    broadband_band: Dict,
    unitscale: str = "nano",
    data_type: str = "velocity",
    title: Optional[str] = None,
    baz_theoretical: Optional[float] = None,
    show_errors: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    broadband_label: Optional[str] = None,
    broadband_velocity: Optional[float] = None,
    global_scaling: bool = False,
    font: int = 12,
    lw: float = 0.8,
    ylim_pad: float = 1.05,
    text_y_pos: float = 0.8,
    suptitle_y: float = 0.91,
    tra_color: str = "black",
    rot_color: str = "red",
    text_bbox: Optional[Dict] = None,
    velocity_stat: str = "median",
) -> plt.Figure:
    """
    Like :func:`plot_dispersion_traces`, with an extra broadband bandpass panel.

    Parameters
    ----------
    broadband_band : dict
        Band dict with ``filtered_rot``, ``filtered_acc``, ``f_lower``, ``f_upper``.
    broadband_velocity : float, optional
        Phase velocity (m/s) for rotation scaling on the broadband panel.
        Overrides ``velocity_stat`` for the broadband panel only.
    velocity_stat : str
        ``median`` or ``kde_peak`` for per-band rotation scaling and labels.
    global_scaling : bool
        If True, normalize translation to the broadband translation peak.
    """
    if "frequency_bands" not in dispersion_results:
        raise ValueError("dispersion_results must contain 'frequency_bands'")

    if broadband_velocity is not None:
        broadband_band = dict(broadband_band)
        broadband_band["kde_peak_velocity"] = float(broadband_velocity)

    frequency_bands = list(dispersion_results["frequency_bands"]) + [broadband_band]
    wave_type = dispersion_results.get("wave_type", "rayleigh")

    if broadband_label is None:
        broadband_label = (
            f"broadband: {broadband_band['f_lower']:.2f}-"
            f"{broadband_band['f_upper']:.2f} Hz"
        )

    return _plot_dispersion_trace_panels(
        frequency_bands,
        wave_type,
        data_type=data_type,
        unitscale=unitscale,
        title=title,
        baz_theoretical=baz_theoretical,
        show_errors=show_errors,
        figsize=figsize,
        global_scaling=global_scaling,
        global_scaling_reference_band=broadband_band,
        broadband_label=broadband_label,
        font=font,
        lw=lw,
        ylim_pad=ylim_pad,
        text_y_pos=text_y_pos,
        suptitle_y=suptitle_y,
        tra_color=tra_color,
        rot_color=rot_color,
        text_bbox=text_bbox,
        velocity_stat=velocity_stat,
    )
