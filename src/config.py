# src/config.py
"""
Shared plot style configuration.

Usage
-----
    from config import STYLE_CONFIG
"""

STYLE_CONFIG = {
    'Simulation':         {'color': '#2c3e50', 'ls': '-',  'marker': 'o'},
    'Thermo ML':          {'color': '#2980b9', 'ls': '--', 'marker': 's'},
    'Thermo DL':          {'color': "#A9DF62", 'ls': '--', 'marker': 's'},
    'DynAdj Global PCA':  {'color': '#c0392b', 'ls': ':',  'marker': '^'},
    'DynAdj Local 50':    {'color': '#27ae60', 'ls': '-',  'marker': 'D'},
    'DynAdj Local 25':   {'color': '#8e44ad', 'ls': '--', 'marker': 'v'},
    'Analogues':          {'color': '#e67e22', 'ls': ':',  'marker': 'P'},
    'Analogues Lasso':    {'color': "#e622cc", 'ls': ':',  'marker': 'P'},
    'Analogues Conditional':    {'color': "#88507c", 'ls': ':',  'marker': 'P'},
    'Analogues Unconditional':    {'color': "#0829e7", 'ls': ':',  'marker': 'P'},
    'methods': {
        'empirical': {'alpha': 1.0, 'lw': 2.0},
        'gev':       {'alpha': 0.7, 'lw': 1.5},
        'gaussian':  {'alpha': 0.4, 'lw': 1.0},
    },
}
