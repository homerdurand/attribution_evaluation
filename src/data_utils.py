# src/data_utils.py
"""
Data loading, event detection, and SLP extraction utilities.

Design principle
----------------
The extraction pipeline stores the raw global SLP field as a 3-D array
(T, n_lat, n_lon) together with event barycentres (lat, lon). Local SLP
sub-domains are NOT pre-computed at extraction time — participants slice
the sub-domain of their choice at attribution time via `extract_local_slp`.

Public API
----------
    get_smoothed_gmt      : Centred rolling mean on a monthly GMT series.
    detect_extreme_events : Connected-component detection of extreme events.
    extract_event_fast    : Vectorised area-averaged event extraction + barycentre.
    extract_local_slp     : Slice a lat/lon box around an event barycentre.
    event_frequency_map   : Count timesteps where each pixel is part of an event.
"""

import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from scipy.ndimage import label


# ---------------------------------------------------------------------------
# GMT smoothing
# ---------------------------------------------------------------------------

def get_smoothed_gmt(anom_values, window_years=4):
    """
    Apply a centred rolling mean to a monthly GMT anomaly series.

    Parameters
    ----------
    anom_values  : array-like (T,)
    window_years : int   Window width in years (default 4 → 48 months).

    Returns
    -------
    ndarray (T,)
    """
    window_size = window_years * 12
    return (
        pd.Series(anom_values)
        .rolling(window=window_size, min_periods=1, center=True)
        .mean()
        .values
    )


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_extreme_events(ds, var_name, percentile, min_area=10, land_only=True):
    """
    Detect spatially connected extreme events via connected-component labelling.

    Parameters
    ----------
    ds         : xarray.Dataset
    var_name   : str
    percentile : float   Climatological percentile threshold (e.g. 99.9).
    min_area   : int     Minimum grid-cell count for a valid connected region.
    land_only  : bool    Restrict to land pixels via regionmask (default True).

    Returns
    -------
    xarray.DataArray (time, lat, lon)
        Integer-labelled mask. 0 = background; positive integers identify
        distinct connected regions at each timestep.
    """
    P = ds[var_name].squeeze().transpose("time", "lat", "lon")

    if land_only:
        land_mask = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask      = land_mask.mask(P.lon, P.lat)
        P         = P.where(mask == 0)

    threshold  = P.quantile(percentile / 100.0, dim="time")
    exceedance = (P > threshold).fillna(False)

    structure    = np.ones((3, 3), dtype=int)
    event_labels = []

    for t in range(exceedance.sizes["time"]):
        labeled, num = label(exceedance.isel(time=t).values, structure=structure)
        if num > 0:
            counts = np.bincount(labeled.ravel())
            for rid in np.where(counts < min_area)[0]:
                labeled[labeled == rid] = 0
        event_labels.append(labeled)

    return xr.DataArray(np.stack(event_labels), coords=P.coords, dims=P.dims)


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------

def extract_event_fast(ds, event_mask, var_name, start_year):
    """
    Vectorised extraction of area-averaged event time series and barycentres.

    Parameters
    ----------
    ds          : xarray.Dataset
    event_mask  : xarray.DataArray (time, lat, lon)
    var_name    : str
    start_year  : int   Events before this year are skipped.

    Returns
    -------
    event_series  : ndarray (T, n_events) or None
    event_values  : ndarray (n_events,)   or None   Value at event timestep.
    events_idx    : ndarray (n_events,)   or None   Integer time indices.
    event_coords  : ndarray (n_events, 2) or None   Barycentre [lat, lon].

    All four are None if no valid events are found.
    """
    P              = ds[var_name].squeeze().transpose("time", "lat", "lon")
    nt, nlat, nlon = P.shape

    lon_grid, lat_grid = np.meshgrid(ds.lon.values, ds.lat.values)
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    P_flat    = np.nan_to_num(P.values.reshape(nt, -1))
    mask_flat = event_mask.values.reshape(nt, -1)

    all_masks, events_idx, event_coords = [], [], []

    for t in range(nt):
        if pd.to_datetime(P.time.values[t]).year < start_year:
            continue
        for eid in np.unique(mask_flat[t]):
            if eid == 0:
                continue
            m = mask_flat[t] == eid
            all_masks.append(m)
            events_idx.append(t)
            event_coords.append((np.mean(lat_flat[m]), np.mean(lon_flat[m])))

    if not all_masks:
        return None, None, None, None

    mask_matrix  = np.stack(all_masks).astype(np.float32)
    event_series = (P_flat @ mask_matrix.T) / mask_matrix.sum(axis=1)
    event_values = event_series[events_idx, np.arange(len(events_idx))]

    return event_series, event_values, np.array(events_idx), np.array(event_coords)


# ---------------------------------------------------------------------------
# Local SLP extraction  (called at attribution time, not extraction time)
# ---------------------------------------------------------------------------

def extract_local_slp(slp_3d, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg, return_2d=True):
    """
    Slice a local SLP sub-domain centred on an event barycentre.

    Called at attribution time, not at extraction time, giving each
    participant full flexibility over the box size and centre.

    Parameters
    ----------
    slp_3d         : ndarray (T, n_lat, n_lon)   data['f_slp'] or data['c_slp']
    slp_lat        : ndarray (n_lat,)             data['slp_lat']
    slp_lon        : ndarray (n_lon,)             data['slp_lon']
    ev_lat         : float                        data['location'][e_idx, 0]
    ev_lon         : float                        data['location'][e_idx, 1]
    half_width_deg : float
        Half-width of the bounding box in degrees.
        Total box = 2*half_width_deg x 2*half_width_deg.
        Suggested values: 12.5 -> 25x25 deg,  25.0 -> 50x50 deg.

    Returns
    -------
    ndarray (T, n_pts)   Flattened local SLP over the full time axis.
    """
    lat_min = np.clip(ev_lat - half_width_deg, slp_lat.min(), slp_lat.max())
    lat_max = np.clip(ev_lat + half_width_deg, slp_lat.min(), slp_lat.max())
    lon_min = np.clip(ev_lon - half_width_deg, slp_lon.min(), slp_lon.max())
    lon_max = np.clip(ev_lon + half_width_deg, slp_lon.min(), slp_lon.max())

    lat_idx = np.where((slp_lat >= lat_min) & (slp_lat <= lat_max))[0]
    lon_idx = np.where((slp_lon >= lon_min) & (slp_lon <= lon_max))[0]

    local = slp_3d[:, lat_idx[:, None], lon_idx[None, :]]   # (T, n_lat_b, n_lon_b)
    if return_2d:
        T, nl, nlo = local.shape
        return np.nan_to_num(local.reshape(T, nl * nlo))
    else:
        return local
    


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def event_frequency_map(event_mask):
    """
    Count the number of timesteps each pixel participates in an extreme event.

    Parameters
    ----------
    event_mask : xarray.DataArray (time, lat, lon)

    Returns
    -------
    xarray.DataArray (lat, lon)
    """
    freq      = (event_mask > 0).sum(dim="time")
    freq.name = "event_count"
    return freq


def get_window(d, e_idx, is_factual, window_before=72, window_after=12):
    """
    Return (t_obs, t0, t1) for a given event index.

    Parameters
    ----------
    d            : member data dict from the .pkl file.
    e_idx        : int   event index.
    is_factual   : bool  True → use factual run indices.
    window_before: int   months before event (default 72).
    window_after : int   months after  event (default 12).

    Returns
    -------
    t_obs : int   absolute time index of the event.
    t0    : int   start of the PN window.
    t1    : int   end   of the PN window.
    """
    idx_arr = d['idx_f'] if is_factual else d['idx_c']
    t_obs   = idx_arr[e_idx]
    t0 = max(0, t_obs - window_before)
    t1 = min(d['f_tas'].shape[0], t_obs + window_after)
    return t_obs, t0, t1
