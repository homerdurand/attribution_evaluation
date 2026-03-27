# src/attribution.py
"""
Core PN computation and built-in attribution methods.

All dynamical adjustment methods receive the global 3-D SLP array
(T, n_lat, n_lon) and call `extract_local_slp` from data_utils to cut
their chosen sub-domain at runtime.

Public API
----------
    compute_pn               : PN kernel shared by all methods.
    run_thermo_ml            : GMT-based thermodynamic adjustment.
    run_dyn_adj_global_pca   : Ridge on global SLP compressed with PCA.
    run_dyn_adj_local        : Ridge on a raw local SLP box (no PCA).
    run_dyn_adj_local_pca    : Ridge on a local SLP box + PCA.
"""

import numpy as np
from scipy.stats import norm, genextreme
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.decomposition import PCA
from src.data_utils import extract_local_slp
from src.deep_attributtor_v2 import *


# ---------------------------------------------------------------------------
# PN kernel
# ---------------------------------------------------------------------------

def compute_pn(factual_series, counterfactual_series, threshold, method='empirical', circ_ratio=1):
    """
    Compute the Probability of Necessity (PN) of an extreme event.

    PN = max(0, 1 - P(X > u | counterfactual) / P(X > u | factual))

    Parameters
    ----------
    factual_series        : array-like (n,)
    counterfactual_series : array-like (m,)
    threshold : float   Observed event value used as exceedance threshold u.
    method    : str     'empirical', 'gaussian', or 'gev'.

    Returns
    -------
    float   PN in [0, 1].  Returns 1e-20 when factual exceedance prob is 0.
    """
    if method == 'empirical':
        p_f  = np.mean(factual_series > threshold)
        p_cf = np.mean(counterfactual_series > threshold)

    elif method == 'gaussian':
        mu_f,  std_f  = norm.fit(factual_series)
        mu_cf, std_cf = norm.fit(counterfactual_series)
        p_f  = norm.sf(threshold, loc=mu_f,  scale=std_f)
        p_cf = norm.sf(threshold, loc=mu_cf, scale=std_cf)

    elif method == 'gev':
        shape_f,  loc_f,  scale_f  = genextreme.fit(factual_series)
        shape_cf, loc_cf, scale_cf = genextreme.fit(counterfactual_series)
        p_f  = genextreme.sf(threshold, shape_f,  loc=loc_f,  scale=scale_f)
        p_cf = genextreme.sf(threshold, shape_cf, loc=loc_cf, scale=scale_cf)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: 'empirical', 'gaussian', 'gev'.")

    if p_f <= 0:
        return 1e-20
    return max(0.0, 1.0 - (p_cf / p_f) * (1/circ_ratio))


# ---------------------------------------------------------------------------
# Thermodynamic attribution
# ---------------------------------------------------------------------------

def run_thermo_ml(tas, gmt, obs_val, t_range, mth):
    """
    Thermodynamic attribution via GMT-based linear regression.

    Constructs a counterfactual by removing the GMT-driven trend and
    replacing it with the pre-industrial GMT signal.

    Parameters
    ----------
    tas   : array (T,)      Temperature time series.
    gmt   : array (T,)      Smoothed GMT anomaly.
    obs_val : float           Observed event threshold.
    t_range : tuple (t0, t1)  Window for PN distribution.
    mth     : str             PN method.

    Returns
    -------
    float  PN value.
    """
    gmt_c = np.full_like(gmt, gmt[0])
    reg    = LinearRegression().fit(gmt[:, None], tas)
    tas_cf = tas - reg.predict(gmt[:, None]) + reg.predict(gmt_c[:, None])
    t0, t1 = t_range
    return compute_pn(tas[t0:t1], tas_cf[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box, no PCA
# ---------------------------------------------------------------------------

def run_adjusted_thermo_ml(tas, gmt, slp, slp_lat, slp_lon, ev_lat, ev_lon,
                      obs_val, t_range, mth,
                      half_width_deg=12.5, n_years=50, n_components=10):
    """
    Dynamical adjustment on a raw local SLP box (no PCA).

    Best suited for small boxes (<= 25x25 deg, ~100 pts on a 2.5-deg grid)
    where Ridge can be applied directly without dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float                    — data['location'][e_idx, 0]
    ev_lon         : float                    — data['location'][e_idx, 1]
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 12.5 -> 25x25 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    slp_local_clean = np.nan_to_num(slp_local)
    slp_local_past = slp_local_clean[:n_years*12, :]


    pls = PLSRegression(n_components=n_components)
    pls.fit(slp_local_past, tas[:n_years*12])
    slp_pls_components = pls.transform(slp_local_clean)

    predictors = np.hstack([gmt.reshape(-1, 1), slp_pls_components])
    reg = LinearRegression().fit(predictors, tas)

    predictors_counterfactual = np.hstack([
        np.full_like(gmt, gmt[0]).reshape(-1, 1), 
        slp_pls_components
    ])
    tas_dyn = reg.predict(predictors_counterfactual)
    t0, t1    = t_range
    return compute_pn(tas[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


def run_adjusted_thermo_dl(tas, gmt, slp, slp_lat, slp_lon, ev_lat, ev_lon,
                      t_obs, half_width_deg=12.5, k_clusters=10, latent_dim=16):
    """
    Dynamical adjustment on a raw local SLP box (no PCA).

    Best suited for small boxes (<= 25x25 deg, ~100 pts on a 2.5-deg grid)
    where Ridge can be applied directly without dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float                    — data['location'][e_idx, 0]
    ev_lon         : float                    — data['location'][e_idx, 1]
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 12.5 -> 25x25 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    Z = extract_local_slp(
        slp, slp_lat, slp_lon, 
        ev_lat, ev_lon, half_width_deg=half_width_deg, return_2d=False
    )
    Y = tas
    X = gmt
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)

    
    dcm = DeepCausalMediation(k_clusters=k_clusters, latent_dim=latent_dim, input_channels=1, spatial_dim=Z.shape[1:])
    dcm, _ = train_causal_model(X, Y, Z, dcm, epochs=200, batch_size=32, lr=1e-3, patience=15, val_split=0.2, 
                             tau_start=1.0, tau_min=0.1, lambda_kl_start=0.0, lambda_kl_max=1.0, lambda_kl_warmup=0.3, 
                             grad_clip=1.0, seed=42, num_workers=0)
    
    x_c = X[0]
    x_f = X[t_obs]
    y_th = Y[t_obs]
    # print(f'threshold data: {x_c}, {x_f}, {y_th}')
    p_c, p_f, pn = estimate_pn(dcm, x_c, x_f, y_th)
    
    return pn


# ---------------------------------------------------------------------------
# Dynamical adjustment — global SLP + PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_global_pca(tas, slp, obs_val, t_range, mth,
                            n_components=50, alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on global SLP compressed with PCA.

    Parameters
    ----------
    tas        : array (T,)
    slp        : array (T, n_lat, n_lon)  — data['f_slp']
    obs_val      : float
    t_range      : tuple (t0, t1)
    mth          : str
    n_components : int        PCA components to retain (default 50).
    alphas       : array-like RidgeCV regularisation grid.

    Returns
    -------
    float  PN value.
    """
    T        = slp.shape[0]
    slp_flat = np.nan_to_num(slp.reshape(T, -1))
    pca      = PCA(n_components=n_components)
    slp_pcs  = pca.fit_transform(slp_flat)
    reg      = RidgeCV(alphas=alphas).fit(slp_pcs, tas)
    tas_dyn  = reg.predict(slp_pcs)
    t0, t1   = t_range
    return compute_pn(tas[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box, no PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_local(tas, slp, slp_lat, slp_lon, ev_lat, ev_lon,
                      obs_val, t_range, mth,
                      half_width_deg=12.5, n_years=50,
                      alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a raw local SLP box (no PCA).

    Best suited for small boxes (<= 25x25 deg, ~100 pts on a 2.5-deg grid)
    where Ridge can be applied directly without dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float                    — data['location'][e_idx, 0]
    ev_lon         : float                    — data['location'][e_idx, 1]
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 12.5 -> 25x25 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    slp_local_past = slp_local[:n_years*12, :]
    reg       = RidgeCV(alphas=alphas).fit(slp_local_past, tas[:n_years*12])
    tas_dyn   = reg.predict(slp_local)
    t0, t1    = t_range
    return compute_pn(tas[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


def run_dml_dyn_adj_local(tas, gmt, slp, slp_lat, slp_lon, ev_lat, ev_lon,
                      obs_val, t_range, mth,
                      half_width_deg=12.5, alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a raw local SLP box (no PCA).

    Best suited for small boxes (<= 25x25 deg, ~100 pts on a 2.5-deg grid)
    where Ridge can be applied directly without dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float                    — data['location'][e_idx, 0]
    ev_lon         : float                    — data['location'][e_idx, 1]
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 12.5 -> 25x25 total).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    lr = LinearRegression()
    lr.fit(gmt.reshape(-1, 1), tas)
    res_tas = tas - lr.predict(gmt.reshape(-1, 1))
    lr.fit(gmt.reshape(-1, 1), slp_local)
    res_slp = slp_local - lr.predict(gmt.reshape(-1, 1))
    reg       = RidgeCV(alphas=alphas).fit(res_slp, res_tas)
    tas_dyn   = reg.predict(res_slp)
    t0, t1    = t_range
    return compute_pn(tas[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)
# ---------------------------------------------------------------------------
# Dynamical adjustment — local SLP box + PCA
# ---------------------------------------------------------------------------

def run_dyn_adj_local_pca(tas, slp, slp_lat, slp_lon, ev_lat, ev_lon,
                           obs_val, t_range, mth,
                           half_width_deg=25.0, n_components=20,
                           alphas=np.logspace(-5, 15, 50)):
    """
    Dynamical adjustment on a local SLP box compressed with PCA.

    Recommended for larger boxes (50x50 deg) where the raw grid is too
    high-dimensional for Ridge without prior dimensionality reduction.

    Parameters
    ----------
    tas_f          : array (T,)
    slp_f          : array (T, n_lat, n_lon)  — data['f_slp']
    slp_lat        : array (n_lat,)           — data['slp_lat']
    slp_lon        : array (n_lon,)           — data['slp_lon']
    ev_lat         : float
    ev_lon         : float
    obs_val        : float
    t_range        : tuple (t0, t1)
    mth            : str
    half_width_deg : float   Half-width in degrees (default 25.0 -> 50x50 total).
    n_components   : int     PCA components (default 20).
    alphas         : array-like

    Returns
    -------
    float  PN value.
    """
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    pca       = PCA(n_components=min(n_components, slp_local.shape[1]))
    slp_pcs   = pca.fit_transform(np.nan_to_num(slp_local))
    reg       = RidgeCV(alphas=alphas).fit(slp_pcs, tas)
    tas_dyn   = reg.predict(slp_pcs)
    t0, t1    = t_range
    return compute_pn(tas[t0:t1], tas_dyn[t0:t1], obs_val, method=mth)


def thermo_cf(tas, gmt):
    """
    Build a thermodynamic counterfactual temperature series.

    Removes the GMT-driven trend and replaces it with the pre-industrial
    level (first value of the factual GMT).

    Parameters
    ----------
    tas : array (T,)   Factual temperature time series.
    gmt : array (T,)   Smoothed factual GMT anomaly.

    Returns
    -------
    ndarray (T,)  Counterfactual temperature.
    """
    gmt_c = np.full_like(gmt, gmt[0])
    reg   = LinearRegression().fit(gmt[:, None], tas)
    return tas - reg.predict(gmt[:, None]) + reg.predict(gmt_c[:, None])


def pn_gaussian(tas_win, cf_win, threshold):
    """
    Quick PN estimate using Gaussian tail probabilities.

    Parameters
    ----------
    tas_win   : array (n,)  Factual temperature sample.
    cf_win    : array (m,)  Counterfactual temperature sample.
    threshold : float       Event exceedance threshold.

    Returns
    -------
    float  PN in [0, 1].
    """
    mu_f,  s_f  = norm.fit(tas_win)
    mu_cf, s_cf = norm.fit(cf_win)
    p_f  = norm.sf(threshold, mu_f,  s_f)
    p_cf = norm.sf(threshold, mu_cf, s_cf)
    return max(0.0, 1.0 - p_cf / p_f) if p_f > 0 else 0.0
