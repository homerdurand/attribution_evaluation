# src/analogues.py
"""
Analogue-based attribution methods.

All functions return a PN value computed over a neighbourhood of past
atmospheric states that resemble the target circulation.

Note on SLP input
-----------------
These methods expect a 2-D array (T, n_features). Typical options:
  - Global SLP flattened:  slp_2d = data['f_slp'].reshape(T, -1)
  - Local box:             slp_2d = extract_local_slp(data['f_slp'], ...)
  - PCA scores:            slp_2d = PCA(n).fit_transform(slp_flat)

Available methods
-----------------
    run_analogues          : Standard KNN on SLP features (+ optional GMT detrending).
    run_analogues_lasso    : KNN on Lasso-selected SLP dimensions.
    run_analogues_causal   : KNN on Mutual-Information selected SLP dimensions.
    run_analogues_ridge    : Local Ridge-corrected analogue matching.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

from src.attribution import compute_pn
from src.data_utils import extract_local_slp
from src.sparse_unconditional_analogues import KNNAnalogueAttributor


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def get_nonlinear_mb(X, y, threshold=0.01):
    """
    Select SLP features relevant to temperature via Mutual Information
    (nonlinear Markov Blanket approximation).

    Parameters
    ----------
    X         : array (T, n_features)
    y         : array (T,)   Temperature target.
    threshold : float        MI score below which a feature is discarded.

    Returns
    -------
    ndarray of int   Selected indices sorted by decreasing MI score.
    """
    mi_scores = mutual_info_regression(X, y, discrete_features=False)
    selected  = np.where(mi_scores > threshold)[0]
    return selected[np.argsort(mi_scores[selected])[::-1]]



def run_analogues_local(tas, slp, slp_lat, slp_lon,
                         ev_lat, ev_lon, obs_val, t_obs, t_range, mth,
                         half_width_deg=25.0, n_analogues=100, n_years=50,
                         metric='euclidean', algorithm='auto'):
    """
    Local analogue method: KNN search on a lat/lon box around the event
    barycentre, using the full time series as the analogue pool.

    The counterfactual distribution is the temperatures of the k nearest
    neighbours (direct, no detrending, no bias correction). The full
    record (all available years) is used as the pool so the search is
    not artificially limited to an early pre-warming period.

    Unlike the global analogue methods, the SLP distance is computed only
    within the local box, making it more sensitive to the regional
    circulation pattern that directly drives the event.

    Parameters
    ----------
    tas_f          : array (T,)               Full factual temperature series.
    gmt            : array (T,)               Smoothed GMT anomaly.
    slp_f          : array (T, n_lat, n_lon)  Full global SLP — data['f_slp']
    slp_lat        : array (n_lat,)           data['slp_lat']
    slp_lon        : array (n_lon,)           data['slp_lon']
    ev_lat         : float                    data['location'][e_idx, 0]
    ev_lon         : float                    data['location'][e_idx, 1]
    obs_val        : float                    Event threshold.
    t_obs          : int                      Time index of the event.
    t_range        : tuple (t0, t1)           Window for PN computation.
    mth            : str                      PN estimator.
    half_width_deg : float
        Half-width of the local SLP box in degrees (default 25.0 → 50×50°).
        Should match the spatial scale of the circulation feature driving
        the event; consistent with ANALOGUE_HALF_DEG in the notebooks.
    n_analogues    : int    Number of nearest neighbours (default 100).
    metric         : str    Distance metric for KNN (default 'euclidean').
    algorithm      : str    KNN algorithm.

    Returns
    -------
    float  PN value.
    """

    # Extract the local SLP box — (T, n_pts)
    slp_local = extract_local_slp(
        slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)

    # KNN on full record, excluding the event timestep itself
    T = slp_local.shape[0]
    pool_idx_past = np.array([t for t in range(n_years*12) if t != t_obs])
    pool_idx_present = np.array([t for t in range(t_range[0], t_range[1]) if t != t_obs])

    knn = NearestNeighbors(n_neighbors=n_analogues,
                            metric=metric, algorithm=algorithm)
    knn.fit(slp_local[pool_idx_past])
    _, idx_past = knn.kneighbors(slp_local[t_obs].reshape(1, -1))
    knn.fit(slp_local[pool_idx_present])
    _, idx_present = knn.kneighbors(slp_local[t_obs].reshape(1, -1))

    # Temperatures of the nearest neighbours → counterfactual distribution
    tas_past = tas[idx_past]
    tas_present = tas[idx_present]

    return compute_pn(tas_past, tas_present, obs_val, method=mth)



def run_analogues(tas, slp_feats, obs_val, t_obs, t_range, mth,
                  n_years=50, n_analogues=100,
                  metric='minkowski', algorithm='auto'):
    """
    Standard Global KNN analogue method adapted for Dual-Sample PN.
    """
    # Define pools: Past (early record) and Present (window around event)
    pool_idx_past = np.array([t for t in range(n_years * 12) if t != t_obs])
    pool_idx_present = np.array([t for t in range(t_range[0], t_range[1]) if t != t_obs])

    target = slp_feats[t_obs].reshape(1, -1)
    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)

    # 1. Past Search (Counterfactual)
    knn.fit(slp_feats[pool_idx_past])
    _, idx_past = knn.kneighbors(target)
    tas_past = tas[pool_idx_past][idx_past[0]]

    # 2. Present Search (Factual)
    knn.fit(slp_feats[pool_idx_present])
    _, idx_present = knn.kneighbors(target)
    tas_present = tas[pool_idx_present][idx_present[0]]

    return compute_pn(tas_present, tas_past, obs_val, method=mth)


def run_analogues_local_lasso(tas, slp, slp_lat, slp_lon, ev_lat, ev_lon, obs_val, t_obs, t_range, mth, 
                              half_width_deg=50.0, n_years=50, n_analogues=50, n_components=200, 
                              metric='minkowski', algorithm='auto'):
    """
    Local Lasso-selected analogues adapted for Dual-Sample PN.
    """
    # Extract local box and reduce dimensionality
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    pca = PCA(n_components=n_components)
    slp_pca = pca.fit_transform(slp_local)

    # Lasso feature selection (trained on full series to find stable circulation drivers)
    lasso = LassoCV(cv=10).fit(slp_pca.astype(np.float64), tas.astype(np.float64))
    selected = np.where(lasso.coef_ != 0)[0]
    if len(selected) == 0:
        selected = np.arange(slp_pca.shape[1])

    # Narrow to selected features
    X = slp_pca[:, selected]
    target = X[t_obs].reshape(1, -1)

    # Define pools
    pool_idx_past = np.array([t for t in range(n_years * 12) if t != t_obs])
    pool_idx_present = np.array([t for t in range(t_range[0], t_range[1]) if t != t_obs])

    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)

    # 1. Past Search
    knn.fit(X[pool_idx_past])
    _, idx_past = knn.kneighbors(target)
    tas_past = tas[pool_idx_past][idx_past[0]]

    # 2. Present Search
    knn.fit(X[pool_idx_present])
    _, idx_present = knn.kneighbors(target)
    tas_present = tas[pool_idx_present][idx_present[0]]

    return compute_pn(tas_present, tas_past, obs_val, method=mth)




def run_unconditional_analogues_lasso(tas, slp, slp_lat, slp_lon, ev_lat, ev_lon, obs_val, t_obs, t_range, mth, 
                              half_width_deg=50.0, n_years=50, n_analogues=50, n_components=200, 
                              metric='minkowski', algorithm='auto'):
    """
    1. Feature Scaling (essential for PCA/KNN)
    2. Leakage Prevention (Lasso trained on Past only)
    3. Log-stable Circulation Ratio with Median Heuristic
    """
    
    # --- 1. PRE-PROCESSING & SCALING ---
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    
    # PCA is distance-based; standardizing prevents grid-point variance from biasing components
    scaler = StandardScaler()
    slp_scaled = scaler.fit_transform(slp_local)
    
    pca = PCA(n_components=n_components)
    slp_pca = pca.fit_transform(slp_scaled)

    # --- 2. DEFINE POOLS EARLY ---
    # We define these now so we can train the Lasso without "seeing" the future
    pool_idx_past = np.array([t for t in range(n_years * 12) if t != t_obs])
    pool_idx_present = np.array([t for t in range(t_range[0], t_range[1]) if t != t_obs])

    # --- 3. LASSO (PREVENTING LEAKAGE) ---
    # FIX: Train Lasso only on the 'past' pool. 
    # This finds circulation drivers of temperature in a 'natural' state, 
    # avoiding inflation from the forced trend in the present period.
    lasso = LassoCV(cv=10).fit(
        slp_pca[pool_idx_past].astype(np.float64), 
        tas[pool_idx_past].astype(np.float64)
    )
    
    selected = np.where(lasso.coef_ != 0)[0]
    if len(selected) == 0:
        # Fallback: if Lasso nulls everything, take top 5 PCs to maintain some structure
        selected = np.arange(min(5, slp_pca.shape[1]))

    X = slp_pca[:, selected]
    d_dim = X.shape[1] 
    target = X[t_obs].reshape(1, -1)

    # --- 4. KNN SEARCH & DENSITY ESTIMATION ---
    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)

    # Past Search
    knn.fit(X[pool_idx_past])
    d_past_all, idx_past = knn.kneighbors(target)
    # Median heuristic for local density robustness
    d_k_past = np.median(d_past_all)

    # Present Search
    knn.fit(X[pool_idx_present])
    d_present_all, idx_present = knn.kneighbors(target)
    d_k_present = np.median(d_present_all)

    # --- 5. LOG-STABLE CIRCULATION RATIO ---
    n_c = len(pool_idx_past)
    n_f = len(pool_idx_present)
    
    # Using log-transform to prevent overflow with (dist_ratio)**d_dim
    # ln(R) = ln(n_c/n_f) + d * (ln(d_past) - ln(d_present))
    # We use d_k_past + 1e-9 to prevent log(0) in edge cases
    log_ratio = (np.log(n_c) - np.log(n_f)) + d_dim * (np.log(d_k_past + 1e-9) - np.log(d_k_present + 1e-9))
    
    # Cap the log_ratio to prevent extreme 'exp' results if dimensionality is high
    log_ratio = np.clip(log_ratio, -10, 10) 
    circulation_ratio = np.exp(log_ratio)
    # print(d_dim, circulation_ratio)

    # --- 6. TARGET DATA EXTRACTION ---
    tas_past = tas[pool_idx_past][idx_past[0]]
    tas_present = tas[pool_idx_present][idx_present[0]]

    return compute_pn(tas_present, tas_past, obs_val, method=mth, circ_ratio=circulation_ratio)


def run_analogues_causal_knn(tas, gmt, slp, slp_lat, slp_lon, ev_lat, ev_lon, obs_val, t_obs, t_range, mth, 
                             half_width_deg=25.0, n_years=50, n_analogues=100, n_components=20, 
                             metric='minkowski', algorithm='auto'):
    """
    Causal Forest selected analogues adapted for Dual-Sample PN.
    """
    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    slp_pca = PCA(n_components=n_components).fit_transform(StandardScaler().fit_transform(slp_local))

    # Causal Feature Selection
    est = CausalForestDML(model_y=RandomForestRegressor(n_estimators=100, max_depth=5),
                          model_t=RandomForestRegressor(n_estimators=100, max_depth=5),
                          cv=3)
    est.fit(tas.astype(np.float64), gmt.astype(np.float64), X=slp_pca.astype(np.float64))
    
    importances = est.feature_importances_
    selected = np.where(importances >= np.mean(importances))[0]
    if len(selected) == 0: selected = np.argsort(importances)[-3:]

    X = slp_pca[:, selected]
    target = X[t_obs].reshape(1, -1)

    # Define pools consistent with run_analogues_local
    pool_idx_past = np.array([t for t in range(n_years * 12) if t != t_obs])
    pool_idx_present = np.array([t for t in range(t_range[0], t_range[1]) if t != t_obs])

    knn = NearestNeighbors(n_neighbors=n_analogues, metric=metric, algorithm=algorithm)

    # 1. Past Search
    knn.fit(X[pool_idx_past])
    _, idx_past = knn.kneighbors(target)
    tas_past = tas[pool_idx_past][idx_past[0]]

    # 2. Present Search
    knn.fit(X[pool_idx_present])
    _, idx_present = knn.kneighbors(target)
    tas_present = tas[pool_idx_present][idx_present[0]]

    return compute_pn(tas_present, tas_past, obs_val, method=mth)


def preprocess_attribution_data(slp, slp_lat, slp_lon, ev_lat, ev_lon,
                                  half_width_deg=30.0, n_pcs=10):

    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    X = slp_local.reshape(slp_local.shape[0], -1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_pcs)
    pca.fit(X_scaled)
    X_pcs = pca.transform(X_scaled)

    return X_pcs



def run_knn_attributor(tas, slp, slp_lat, slp_lon,
                       ev_lat, ev_lon, obs_val, t_obs, t_range, mth,
                       half_width_deg=20.0,
                       lasso_method='lasso', n_analogues=None, n_components=10, adjust_circulation=True):
    """
    Wraps KNNAnalogueAttributor to match the run_analogues_local_lasso signature
    and return a single pn scalar compatible with ATTRIBUTION_METHODS.
    """
    
    past_idx    = np.arange(12 * 50)
    present_idx = np.arange(t_range[0], t_range[1])
    
    print(f'indices sizes: {len(past_idx)}, {len(present_idx)}')

    slp_local = extract_local_slp(slp, slp_lat, slp_lon, ev_lat, ev_lon, half_width_deg)
    X = slp_local.reshape(slp_local.shape[0], -1)

    attributor = KNNAnalogueAttributor(n_analogues=n_analogues, metric='euclidean')
    attributor.fit_features(X, tas, past_idx, method=lasso_method, n_components=n_components)

    res = attributor.compute_attribution(
        t_obs=int(t_obs),
        past_idx=past_idx,
        present_idx=present_idx,
        tas_data=tas,
        obs_val=obs_val,
    )
    
    
    if adjust_circulation:
        pn = res['pn']
    else:
        pn = 1 - 1/res['risk_ratio']
    return pn