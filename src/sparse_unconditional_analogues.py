import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import parcorr, gpdc

class KNNAnalogueAttributor:
    def __init__(self, n_analogues=20, metric='euclidean'):
        self.n_analogues = None
        self.metric = metric
        self.scaler = StandardScaler()
        self.X_reduced = None
        self.selected_indices = None
        self.projection_matrix = None
        self.train_X_scaled = None
        self.n_analogues = n_analogues,

    # --- OPTIMAL K SELECTION ---

    def _select_optimal_k(self, X_reduced, tas_data, past_idx, k_range=None):
        """
        Selects optimal n_analogues via leave-one-out cross-validation on past data.
        For each candidate k, uses KNN to predict tas from analogues and minimises MSE.
        """
        if k_range is None:
            k_range = np.arange(5, 51, 5)

        X_past = X_reduced[past_idx]
        y_past = tas_data[past_idx]
        mse_scores = []

        for k in k_range:
            errors = []
            # +1 because the query point itself will be its own nearest neighbour
            knn = NearestNeighbors(n_neighbors=k + 1).fit(X_past)
            distances, indices = knn.kneighbors(X_past)

            for i in range(len(X_past)):
                # Exclude the query point (index 0 is always self)
                neighbour_idx = indices[i, 1:k + 1]
                y_pred = np.mean(y_past[neighbour_idx])
                errors.append((y_past[i] - y_pred) ** 2)

            mse_scores.append(np.mean(errors))

        optimal_k = k_range[np.argmin(mse_scores)]
        return optimal_k, k_range, mse_scores

    # --- FEATURE REDUCTION & CAUSAL SELECTION ---

    def _kernel_pls(self, X, y, n_components=2, kernel='rbf', gamma=None):
        """
        Simplified Kernel PLS implementation.
        Computes the kernel matrix and performs PLS in the kernel space.
        """
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        
        K = rbf_kernel(X, X, gamma=gamma)
        
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(K_centered, y)
        
        return pls, gamma

    def fit_features(self, slp_data, tas_data, past_idx, method='pls', 
                     n_components=2, pcmci_params=None, kpls_gamma=None):
        """
        Methods: 'none', 'lasso', 'sir', 'pls', 'kpls', 'pcmci'
        If self.n_analogues is None, optimal k is selected automatically
        via LOO-CV on past data over the range (5, 50).
        """
        X_scaled = self.scaler.fit_transform(slp_data)
        self.train_X_scaled = X_scaled[past_idx]
        
        if method == 'kpls':
            self.kpls_model, self.gamma = self._kernel_pls(
                self.train_X_scaled, tas_data[past_idx], 
                n_components=n_components, gamma=kpls_gamma
            )
            K_all = rbf_kernel(X_scaled, self.train_X_scaled, gamma=self.gamma)
            n_train = self.train_X_scaled.shape[0]
            one_n = np.ones((K_all.shape[0], n_train)) / n_train
            K_all_centered = K_all - one_n @ rbf_kernel(self.train_X_scaled, self.train_X_scaled, gamma=self.gamma)
            self.X_reduced = self.kpls_model.transform(K_all_centered)

        elif method == 'pls':
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_scaled[past_idx], tas_data[past_idx])
            self.X_reduced = pls.transform(X_scaled)
            self.projection_matrix = pls.x_rotations_

        elif method == 'pcmci':
            params = pcmci_params or {}
            self.selected_indices = self._pcmci_selector(X_scaled[past_idx], tas_data[past_idx], **params)
            self.X_reduced = X_scaled[:, self.selected_indices]

        elif method == 'lasso':
            pca = PCA(n_components=n_components)
            X_pcs = pca.fit_transform(X_scaled[past_idx])
            lasso = LassoCV(cv=5).fit(X_pcs, tas_data[past_idx])
            self.selected_indices = np.where(lasso.coef_ != 0)[0]
            self.X_reduced = X_pcs[:, self.selected_indices]
            
        elif method == 'sir':
            X_std = X_scaled[past_idx]
            idx = np.argsort(tas_data[past_idx])
            slices = np.array_split(X_std[idx], 10)
            V = np.cov(np.array([np.mean(s, axis=0) for s in slices]), rowvar=False)
            _, eigvecs = np.linalg.eigh(V)
            self.projection_matrix = eigvecs[:, -n_components:]
            self.X_reduced = X_scaled @ self.projection_matrix
            
        else:
            self.X_reduced = X_scaled

        # Auto-select k after reduction space is established
        if self.n_analogues is None:
            self.n_analogues, self._k_range, self._k_mse = self._select_optimal_k(
                self.X_reduced, tas_data, past_idx
            )
            # print(f"[KNN] Auto-selected n_analogues = {self.n_analogues} "
                #   f"(LOO-CV MSE = {min(self._k_mse):.4f})")
            
        return self

    # --- ATTRIBUTION CORE ---

    def compute_attribution(self, t_obs, past_idx, present_idx, tas_data, obs_val):
        target_x = self.X_reduced[t_obs].reshape(1, -1)
        d = self.X_reduced.shape[1] 
        print(f'target shape : {target_x.shape}')

        knn_p = NearestNeighbors(n_neighbors=self.n_analogues).fit(self.X_reduced[past_idx])
        knn_f = NearestNeighbors(n_neighbors=self.n_analogues).fit(self.X_reduced[present_idx])
        
        dp, ip = knn_p.kneighbors(target_x)
        df, if_ = knn_f.kneighbors(target_x)

        d_k_p = np.median(dp)
        d_k_f = np.median(df)
        log_R = (np.log(len(past_idx)) - np.log(len(present_idx))) + \
                d * (np.log(d_k_p + 1e-9) - np.log(d_k_f + 1e-9))
        circ_ratio = np.exp(np.clip(log_R, -10, 10))

        p_c_cond = norm.sf(obs_val, *norm.fit(tas_data[past_idx][ip[0]]))
        p_f_cond = norm.sf(obs_val, *norm.fit(tas_data[present_idx][if_[0]]))

        p_c_uncond = p_c_cond 
        RR = p_f_cond / (p_c_uncond + 1e-12)
        pn = 1 - 1/(RR + 1e-12) * circ_ratio

        return {"pn": np.clip(pn, 0, 1), "risk_ratio": RR, "circ_ratio": circ_ratio, "d_dim": d}

    # --- SENSITIVITY ---

    def bootstrap_ci(self, t_obs, past_idx, present_idx, tas_data, obs_val, n_boot=100, full_data=True):
        res = []
        for _ in range(n_boot):
            p = np.random.choice(past_idx, len(past_idx), replace=True)
            f = np.random.choice(present_idx, len(present_idx), replace=True)
            res.append(self.compute_attribution(t_obs, p, f, tas_data, obs_val))
        if full_data:
            return pd.DataFrame(res)
        return pd.DataFrame(res).quantile([0.05, 0.5, 0.95])

    def sensitivity_k(self, t_obs, past_idx, present_idx, tas_data, obs_val, k_range=None):
        if k_range is None: k_range = np.arange(20, 101, 20)
        sens = []
        orig_k = self.n_analogues
        for k in k_range:
            self.n_analogues = k
            r = self.compute_attribution(t_obs, past_idx, present_idx, tas_data, obs_val)
            r['k'] = k
            sens.append(r)
        self.n_analogues = orig_k
        return pd.DataFrame(sens)