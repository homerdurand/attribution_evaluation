# src/visualization.py
"""
Evaluation plots for extreme event attribution methods.

Public API
----------
    plot_time_evolution : Rolling Type I error and power over time.
    plot_qq_analysis    : Log-log QQ plot and power curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_time_evolution(df, algo_groups, window=1, save_path='figures/evolution.png'):
    """
    Rolling Type I error and statistical power over time.

    Two sub-panels per algorithm group:
        Left  — yearly Type I error rate (counterfactual scenario)
        Right — yearly statistical power  (factual scenario)

    Solid lines = alpha 5 %,  dashed lines = alpha 1 %.

    Parameters
    ----------
    df          : pd.DataFrame
        Results with columns 'scenario', 'time' (datetime), and one PN
        column per method.
    algo_groups : dict
        { group_label: [(display_label, column_name, color), ...] }
    window      : int   Rolling window in years (default 1 = no smoothing).
    save_path   : str   Output path (parent directory must exist).
    """
    def _rate(sub_df, col, alpha, win):
        rejected    = (sub_df[col] > (1 - alpha)).astype(float)
        yearly_rate = rejected.groupby(sub_df['time'].dt.year).mean()
        return yearly_rate.rolling(window=win, center=True).mean()

    df_null = df[df['scenario'] == 'counterfactual']
    df_fact = df[df['scenario'] == 'factual']

    n_groups = len(algo_groups)
    fig, axes = plt.subplots(n_groups, 2,
                             figsize=(14, 3 * n_groups),
                             sharex=True, squeeze=False)

    for i, (algo_name, configs) in enumerate(algo_groups.items()):
        ax_err, ax_pow = axes[i, 0], axes[i, 1]

        for label, col, color in configs:
            if col not in df.columns:
                continue
            ax_err.plot(_rate(df_null, col, 0.05, window).index,
                        _rate(df_null, col, 0.05, window),
                        color=color, lw=1.8, label=label)
            ax_err.plot(_rate(df_null, col, 0.01, window).index,
                        _rate(df_null, col, 0.01, window),
                        color=color, lw=1.2, ls='--', alpha=0.7)
            ax_pow.plot(_rate(df_fact, col, 0.05, window).index,
                        _rate(df_fact, col, 0.05, window),
                        color=color, lw=1.8)
            ax_pow.plot(_rate(df_fact, col, 0.01, window).index,
                        _rate(df_fact, col, 0.01, window),
                        color=color, lw=1.2, ls='--', alpha=0.7)

        ax_err.axhline(0.05, color='black', lw=0.8, ls=':', alpha=0.5)
        ax_err.set_title(f'{algo_name} — Type I Error (5 % & 1 %)',
                         loc='left', fontsize=10, fontweight='bold')
        ax_pow.set_title(f'{algo_name} — Statistical Power',
                         loc='left', fontsize=10, fontweight='bold')
        for ax in (ax_err, ax_pow):
            ax.grid(True, alpha=0.2)
        ax_err.legend(loc='upper left', fontsize=7, frameon=True)

    axes[-1, 0].set_xlabel('Year')
    axes[-1, 1].set_xlabel('Year')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


def plot_qq_analysis(df, algo_groups, save_path='figures/qq_analysis.png'):
    """
    Log-log QQ plot of Type I error control and power curves.

    Panel A (left)  — Observed false-positive rate vs. nominal alpha (log-log).
                      Perfect calibration lies on the diagonal.
    Panel B (right) — Statistical power vs. significance level (log-x).

    Line style encodes the PN statistical estimator:
        solid  -> Empirical  label contains '(Emp)'
        dashed -> GEV        label contains '(GEV)'
        dotted -> Gaussian   label contains '(Norm)'

    Parameters
    ----------
    df          : pd.DataFrame
    algo_groups : dict   Same structure as plot_time_evolution.
    save_path   : str
    """
    df_null = df[df['scenario'] == 'counterfactual']
    df_fact = df[df['scenario'] == 'factual']
    alphas  = np.geomspace(1e-4, 1, 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    style_map = {'(Emp)': '-', '(GEV)': '--', '(Norm)': ':'}

    for algo_name, configs in algo_groups.items():
        for label, col, color in configs:
            if col not in df.columns:
                continue
            ls       = next((s for k, s in style_map.items() if k in label), '-')
            observed = [np.mean(df_null[col] > (1 - a)) for a in alphas]
            power    = [np.mean(df_fact[col] > (1 - a)) for a in alphas]
            ax1.plot(alphas, observed, label=label, color=color, ls=ls, lw=2)
            ax2.plot(alphas, power,    label=label, color=color, ls=ls, lw=2)

    # Perfect-calibration diagonal
    ax1.plot([1e-4, 1], [1e-4, 1], color='black', lw=1, ls='-', alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('A — Type I Error Control (QQ)', loc='left', fontweight='bold')
    ax1.set_xlabel(r'Nominal $\alpha$')
    ax1.set_ylabel('Observed Rate')
    ax1.grid(True, which='both', ls=':', alpha=0.3)

    ax2.set_xscale('log')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('B — Statistical Power', loc='left', fontweight='bold')
    ax2.set_xlabel(r'Significance Level ($\alpha$)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, which='both', ls=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


# ---------------------------------------------------------------------------
# Exploration / diagnostic plots  (require cartopy)
# ---------------------------------------------------------------------------

import cartopy.crs as ccrs
import cartopy.feature as cfeature

_PROJ    = ccrs.PlateCarree()
_LAND    = cfeature.NaturalEarthFeature('physical', 'land',       '50m', facecolor='#f0ebe3')
_OCEAN   = cfeature.NaturalEarthFeature('physical', 'ocean',      '50m', facecolor='#cfe2f3')
_COAST   = cfeature.NaturalEarthFeature('physical', 'coastline',  '50m',
                                         edgecolor='#444', facecolor='none', linewidth=0.6)
_BORDERS = cfeature.NaturalEarthFeature('cultural',
                                         'admin_0_boundary_lines_land', '50m',
                                         edgecolor='#999', facecolor='none', linewidth=0.4)


def _add_map_features(ax):
    """Add ocean, land, coastlines and borders to a Cartopy axes."""
    ax.add_feature(_OCEAN,   zorder=0)
    ax.add_feature(_LAND,    zorder=1)
    ax.add_feature(_COAST,   zorder=2)
    ax.add_feature(_BORDERS, zorder=2)


def plot_event_frequency_map(full_data, member_idx=0,
                              save_path='figures/freq_map.png'):
    """
    Map of extreme-event frequency (pixel-wise timestep count) for one
    member, with event barycentres pooled across all members.

    Parameters
    ----------
    full_data   : list of member dicts (from the .pkl file).
    member_idx  : int   Which member's frequency map to use as background.
    save_path   : str
    """
    d        = full_data[member_idx]
    all_lats = np.concatenate([m['location'][:, 0] for m in full_data])
    all_lons = np.concatenate([m['location'][:, 1] for m in full_data])
    freq_map = d['event_frequency_map']
    lats     = freq_map.lat.values
    lons     = freq_map.lon.values
    vals     = freq_map.values.astype(float)
    vals[vals == 0] = np.nan

    fig, ax = plt.subplots(figsize=(14, 6),
                            subplot_kw={'projection': _PROJ})
    _add_map_features(ax)
    pcm = ax.pcolormesh(lons, lats, vals, cmap='YlOrRd', vmin=1,
                         transform=_PROJ, zorder=3)
    ax.scatter(all_lons, all_lats, s=12, c='#2c3e50', alpha=0.5,
               transform=_PROJ, zorder=4, label='Event barycentre')
    plt.colorbar(pcm, ax=ax, label='Event count (timesteps)', shrink=0.7, pad=0.02)
    ax.set_global()
    ax.set_title(f'Extreme event frequency — member {d["member"]}',
                 fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


def plot_gmt_tas(d, e_idx, start_year=1980, end_year=2014,
                 save_path='figures/gmt_relationships.png'):
    """
    Two-panel figure: time series of Tas and GMT (left) and
    Tas vs GMT scatter with regression line (right), restricted
    to the [start_year, end_year] period.

    Parameters
    ----------
    d          : member data dict.
    e_idx      : int   Event index to highlight.
    start_year : int   First year of the plot window (default 1980).
    end_year   : int   Last  year of the plot window (default 2014).
    save_path  : str
    """
    import matplotlib.gridspec as gridspec
    from sklearn.linear_model import LinearRegression

    times_dt = pd.to_datetime(d['times'])
    mask     = (times_dt.year >= start_year) & (times_dt.year <= end_year)
    t_idx    = np.where(mask)[0]

    tas_full = d['f_tas'][:, e_idx]
    gmt_full = d['gmt4_f']
    tas_e    = tas_full[t_idx]
    gmt_e    = gmt_full[t_idx]
    times_e  = times_dt[t_idx]

    t_obs    = d['idx_f'][e_idx]
    ev_lat, ev_lon = d['location'][e_idx]

    reg      = LinearRegression().fit(gmt_e[:, None], tas_e)
    r        = np.corrcoef(gmt_e, tas_e)[0, 1]
    gmt_line = np.linspace(gmt_e.min(), gmt_e.max(), 200)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 0.05, 1], wspace=0.3)

    # ── Time series ───────────────────────────────────────────────
    ax_ts  = fig.add_subplot(gs[0])
    ax_ts2 = ax_ts.twinx()
    ax_ts.plot(times_e, tas_e, color='#c0392b', lw=1.0, alpha=0.7, label='Tas')
    ax_ts2.plot(times_e, gmt_e, color='#2980b9', lw=1.4, label='GMT (4yr)')
    if t_obs in t_idx:
        ax_ts.axvline(times_dt[t_obs], color='black', lw=1.5, ls='--', label='Event')
    ax_ts.set_ylabel('Tas anomaly (K)', color='#c0392b')
    ax_ts2.set_ylabel('GMT anomaly (K)', color='#2980b9')
    ax_ts.set_xlabel('Year')
    ax_ts.set_title(f'Event {e_idx}  |  lat={ev_lat:.1f}°  lon={ev_lon:.1f}°  '
                    f'({start_year}–{end_year})', fontweight='bold')
    lines1, lab1 = ax_ts.get_legend_handles_labels()
    lines2, lab2 = ax_ts2.get_legend_handles_labels()
    ax_ts.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc='upper left')
    ax_ts.grid(True, alpha=0.2)

    # ── Scatter ───────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[2])
    ax_sc.scatter(gmt_e, tas_e, s=6, alpha=0.35, color='#7f8c8d')
    ax_sc.plot(gmt_line, reg.predict(gmt_line[:, None]), color='#c0392b', lw=2,
               label=f'slope = {reg.coef_[0]:.2f} K/K\nr = {r:.2f}')
    if t_obs in t_idx:
        pos = np.searchsorted(t_idx, t_obs)
        ax_sc.scatter(gmt_e[pos], tas_e[pos], s=80, color='black',
                      zorder=5, label='Event')
    ax_sc.set_xlabel('GMT anomaly (K)')
    ax_sc.set_ylabel('Tas anomaly (K)')
    ax_sc.legend(fontsize=9)
    ax_sc.grid(True, alpha=0.2)
    ax_sc.set_title('Tas vs GMT scatter', fontweight='bold')

    plt.suptitle('GMT ↔ Temperature relationships', fontweight='bold', y=1.02)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


def plot_distributions(axes, d, e_idx, is_factual, cf_dict, title_prefix,
                        window_before=72, window_after=12):
    """
    One subplot per method: Gaussian fits of factual vs counterfactual
    temperature distributions with PN annotation.

    Parameters
    ----------
    axes         : list of Axes, one per entry in cf_dict.
    d            : member data dict.
    e_idx        : int
    is_factual   : bool
    cf_dict      : {method_name: counterfactual_array (T,)}
    title_prefix : str   Added to every subplot title.
    window_before, window_after : int  PN window in months.
    """
    from scipy.stats import norm as _norm
    from data_utils import get_window

    t_obs, t0, t1 = get_window(d, e_idx, is_factual, window_before, window_after)
    val     = d['f_tas_vals'][e_idx] if is_factual else d['c_tas_vals'][e_idx]
    tas_run = d['f_tas'][:, e_idx]   if is_factual else d['c_tas'][:, e_idx]

    for ax, (name, cf) in zip(axes, cf_dict.items()):
        tas_win = tas_run[t0:t1]
        cf_win  = cf[t0:t1]
        xmin = min(tas_win.min(), cf_win.min()) - 0.3
        xmax = max(tas_win.max(), cf_win.max()) + 0.3
        x    = np.linspace(xmin, xmax, 300)

        mu_f,  s_f  = _norm.fit(tas_win)
        mu_cf, s_cf = _norm.fit(cf_win)
        p_f  = _norm.sf(val, mu_f,  s_f)
        p_cf = _norm.sf(val, mu_cf, s_cf)
        pn   = max(0.0, 1.0 - p_cf / p_f) if p_f > 0 else 0.0

        ax.fill_between(x, _norm.pdf(x, mu_f,  s_f),  alpha=0.4,
                        color='#c0392b', label=f'Factual  μ={mu_f:.2f}')
        ax.fill_between(x, _norm.pdf(x, mu_cf, s_cf), alpha=0.4,
                        color='#2980b9', label=f'CF  μ={mu_cf:.2f}')
        ax.axvline(val, color='black', lw=1.5, ls='--', label=f'u={val:.2f}')
        ax.set_title(f'{title_prefix}\n{name}\nPN={pn:.3f}',
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)
        ax.set_xlabel('Temperature anomaly (K)')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.2)


def _get_analogue_data(d, e_idx, is_factual, n_analogues=50,
                        n_years_pool=50, half_width_deg=25.0,
                        window_before=72, window_after=12):
    """
    Run KNN inside the local SLP box and return per-analogue arrays.

    Returns
    -------
    anom_each : (N, nlb, nlo_b)  SLP anomaly vs box climatology
    mean_each : (N, nlb, nlo_b)  raw analogue SLP
    lat_box, lon_box             coordinate axes of the box
    ev_lat, ev_lon               event barycentre
    """
    from sklearn.neighbors import NearestNeighbors
    from data_utils import extract_local_slp, get_window

    slp_run  = d['f_slp'] if is_factual else d['c_slp']
    t_obs, _, _ = get_window(d, e_idx, is_factual, window_before, window_after)
    ev_lat, ev_lon   = d['location'][e_idx]
    slp_lat, slp_lon = d['slp_lat'], d['slp_lon']
    pool_end         = n_years_pool * 12

    slp_local = extract_local_slp(
        slp_run, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)

    # KNN directly on the full local SLP field — no PCA
    knn   = NearestNeighbors(n_neighbors=n_analogues).fit(slp_local[:pool_end])
    _, idx = knn.kneighbors(slp_local[t_obs].reshape(1, -1))
    ana_idx  = idx[0]

    lat_idx  = np.where((slp_lat >= ev_lat - half_width_deg) &
                        (slp_lat <= ev_lat + half_width_deg))[0]
    lon_idx  = np.where((slp_lon >= ev_lon - half_width_deg) &
                        (slp_lon <= ev_lon + half_width_deg))[0]
    nlb, nlo_b = len(lat_idx), len(lon_idx)
    lat_box  = slp_lat[lat_idx]
    lon_box  = slp_lon[lon_idx]

    clim      = slp_local[:pool_end].mean(axis=0)
    anom_each = (slp_local[ana_idx] - clim).reshape(n_analogues, nlb, nlo_b)
    mean_each =  slp_local[ana_idx].reshape(n_analogues, nlb, nlo_b)
    return anom_each, mean_each, lat_box, lon_box, ev_lat, ev_lon


def plot_analogue_maps(d, e_idx, suptitle,
                        save_path='figures/analogue_slp.png',
                        n_analogues=50, n_years_pool=50,
                        half_width_deg=25.0,
                        window_before=72, window_after=12):
    """
    Big grid: one panel per analogue for both factual (top) and
    counterfactual (bottom) runs.

    Each panel:
      pcolormesh — SLP anomaly vs box climatology (RdBu_r, shared scale)
      contours   — raw analogue SLP (thin black lines)
      star       — event barycentre
      Cartopy features zoomed to the box
    """
    import math
    from matplotlib.gridspec import GridSpec

    ncols   = math.ceil(math.sqrt(n_analogues))
    nrows_s = math.ceil(n_analogues / ncols)

    data_f  = _get_analogue_data(d, e_idx, True,  n_analogues, n_years_pool,
                                  half_width_deg, window_before, window_after)
    data_cf = _get_analogue_data(d, e_idx, False, n_analogues, n_years_pool,
                                  half_width_deg, window_before, window_after)
    vmax = max(
        np.nanpercentile(np.abs(data_f[0]),  98),
        np.nanpercentile(np.abs(data_cf[0]), 98),
    )

    panel_w, panel_h = 2.4, 2.2
    fig = plt.figure(figsize=(ncols * panel_w,
                               nrows_s * 2 * panel_h + 1.5))

    gs_top = GridSpec(nrows_s, ncols, figure=fig,
                      top=0.91, bottom=0.51, hspace=0.05, wspace=0.05)
    gs_bot = GridSpec(nrows_s, ncols, figure=fig,
                      top=0.47, bottom=0.07, hspace=0.05, wspace=0.05)

    for scen_label, scenario_data, gs in [
        ('Factual analogues',        data_f,  gs_top),
        ('Counterfactual analogues', data_cf, gs_bot),
    ]:
        anom_each, mean_each, lat_box, lon_box, ev_lat, ev_lon = scenario_data
        for k in range(n_analogues):
            row = k // ncols
            col = k % ncols
            ax  = fig.add_subplot(gs[row, col], projection=_PROJ)
            ax.set_extent([lon_box.min(), lon_box.max(),
                            lat_box.min(), lat_box.max()], crs=_PROJ)

            # Draw SLP anomaly first (zorder=1), then boundaries on top
            ax.pcolormesh(lon_box, lat_box, anom_each[k],
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           transform=_PROJ, zorder=1)
            lon2d, lat2d = np.meshgrid(lon_box, lat_box)
            try:
                ax.contour(lon2d, lat2d, mean_each[k], levels=6,
                            colors='black', linewidths=0.4,
                            transform=_PROJ, zorder=2, alpha=0.55)
            except Exception:
                pass
            # Land semi-transparent wash, then coast and borders on top
            ax.add_feature(_OCEAN,   zorder=0)
            ax.add_feature(_LAND,    zorder=3, alpha=0.15)
            ax.add_feature(_COAST,   zorder=4)
            ax.add_feature(_BORDERS, zorder=4)
            ax.scatter([ev_lon], [ev_lat], s=30, color='black',
                        marker='*', transform=_PROJ, zorder=5)
            if col == 0 and k == 0:
                ax.set_title(scen_label, fontsize=7, fontweight='bold', pad=2)
            else:
                ax.set_title(str(k + 1), fontsize=6, pad=1)

    sm = plt.cm.ScalarMappable(
        cmap='RdBu_r', norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cax = fig.add_axes([0.15, 0.02, 0.70, 0.018])
    fig.colorbar(sm, cax=cax, orientation='horizontal').set_label(
        f'SLP anomaly vs climatology (Pa)  —  box ±{half_width_deg}°  '
        f'| {n_analogues} nearest analogues', fontsize=8)

    fig.suptitle(suptitle, fontweight='bold', fontsize=11)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')


def plot_local_ridge_weights(d, e_idx, ridge_coef, suptitle,
                               save_path='figures/ridge_weights.png',
                               half_width_deg=40.0):
    """
    Ridge weights (no PCA) for the local DynAdj box, zoomed with
    country / ocean / sea boundaries visible on top of the colour field.

    Parameters
    ----------
    d           : member data dict.
    e_idx       : int
    ridge_coef  : array (n_pts_local,)  extras['ridge_l'].coef_
    suptitle    : str
    save_path   : str
    half_width_deg : float  must match the box used during fitting.
    """
    ev_lat, ev_lon   = d['location'][e_idx]
    slp_lat, slp_lon = d['slp_lat'], d['slp_lon']
    hw               = half_width_deg

    lat_idx  = np.where((slp_lat >= ev_lat - hw) & (slp_lat <= ev_lat + hw))[0]
    lon_idx  = np.where((slp_lon >= ev_lon - hw) & (slp_lon <= ev_lon + hw))[0]
    nlb, nlo_b = len(lat_idx), len(lon_idx)
    lat_box  = slp_lat[lat_idx]
    lon_box  = slp_lon[lon_idx]

    w_2d = np.full((nlb, nlo_b), np.nan)
    if ridge_coef.shape[0] == nlb * nlo_b:
        w_2d = ridge_coef.reshape(nlb, nlo_b)

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': _PROJ})
    ax.set_extent([lon_box.min(), lon_box.max(),
                   lat_box.min(), lat_box.max()], crs=_PROJ)

    vmax = np.nanpercentile(np.abs(w_2d[~np.isnan(w_2d)]), 98)
    pcm  = ax.pcolormesh(lon_box, lat_box, w_2d, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, transform=_PROJ, zorder=1)
    ax.add_feature(_LAND,    zorder=2, alpha=0.15)
    ax.add_feature(_COAST,   zorder=3)
    ax.add_feature(_BORDERS, zorder=3)
    ax.scatter([ev_lon], [ev_lat], s=150, color='black',
               marker='*', transform=_PROJ, zorder=4, label='Event barycentre')
    ax.legend(loc='lower left', fontsize=8)

    plt.colorbar(pcm, ax=ax,
                 label=f'Ridge coefficient  (box ±{hw}°, direct — no PCA)',
                 shrink=0.85, pad=0.02)
    ax.set_title(suptitle, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f'Saved -> {save_path}')
