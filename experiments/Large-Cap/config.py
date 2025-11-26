
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    'Random seed': 42,
    
    # Model and results save paths
    'Visual agent save path': f"./experiments/{experiment_name}/visual_models",
    'Numeric agent save path': f"./experiments/{experiment_name}/numeric_models", 
    'Results save path': f"./experiments/{experiment_name}/aggregated_results", 
    'Portfolio factors save path': f"./experiments/{experiment_name}/portfolio_factors", 
    
    # -------------------------------------------------------------------------
    # Hyperparameters (hardcoded from hyperparam tuning)
    # -------------------------------------------------------------------------
    
    'Lookback window': 7,
    'Deterministic' : True,
    'Checkpoint frequency': 100,
    
    'Visual agent hyperparameters': {
        'Learning rate': 0.0017513990778672951,
        'Batch size': 32,
        'Rollout steps': 2816,
        'Gamma': 0.9067390630376122,
        'GAE lambda': 0.8252833692036996,
        'Clip range': 0.13219658093665856,
        'Entropy coefficient': 5.341336091267625e-08,
        'VF coefficient': 0.10887409930876644,
        'Max grad norm': 0.722506417205221,
        'Epochs': 9,
        'Feature dim': 256
    },
    
    'Numeric agent hyperparameters': {
        'Learning rate': 1.0426317556898562e-05,
        'Batch size': 128,
        'Rollout steps': 3072,
        'Gamma': 0.9936267929071911,
        'GAE lambda': 0.8024631572035075,
        'Clip range': 0.25003219244480845,
        'Entropy coefficient': 1.2492665661830034e-08,
        'VF coefficient': 0.3916052035670908,
        'Max grad norm': 0.9992862769690893,
        'Epochs': 16
    },
    
    # -------------------------------------------------------------------------
    # Experiment Design
    # -------------------------------------------------------------------------
    
    'K folds': 5,
    'Walk throughs': 5,
    'Stratification type': 'random',
    
    # note: the modulus of training ratio and walk throughs must equal 
    # zero in order to get uniform training window sizes
    
    'Evaluation ratio' : 0.20,
    'Validation ratio' : 0.05,
    'Training ratio' : 0.75,
    
    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    
    # Time and frequency attributes
    'Start date': '2004-01-01',
    'End date': '2024-12-30',
    'Update frequency': '1d',
    
    # Features / target
    'Features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'Target': 'Close',
    'GAF periods': 14, 
    'SMA periods': 20,
    'RSI periods': 14,
    
    'Tickers' : ['AZO', 'CI', 'EMR', 'RCL', 'AEP', 'TRV', 'WMB', 'PWR', 'AMT', 'CDNS', 'GD', 'MMC', 'NEM', 'ORLY', 'SO', 'TT', 'DHR', 'GILD', 'HON', 'PLD', 'SYK', 'TXN', 'ABT', 'AMAT', 'AMGN', 'CAT', 'DIS', 'GS', 'T', 'TMO', 'AAPL', 'BRK.B', 'MU', 'ORCL', 'WFC', 'WMT', 'MAR', 'REGN', 'MCK', 'PH', 'KLAC', 'ETN', 'SPGI', 'UNP', 'MS', 'RTX', 'HD', 'PG', 'MSFT', 'JNJ']

}

