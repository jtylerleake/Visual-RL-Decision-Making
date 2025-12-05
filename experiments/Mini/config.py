
from common.modules import os
current_directory = os.path.dirname(os.path.abspath(__file__))
experiment_name = os.path.basename(current_directory)


EXPERIMENT_CONFIG = {
    
    'Experiment name': experiment_name,
    'Random seed': 42,
    
    # Model and results save paths
    'Visual agent save path': f"./experiments/{experiment_name}/visual_models",
    'Numeric agent save path': f"./experiments/{experiment_name}/numeric_models", 
    'Results save path': f"./experiments/{experiment_name}/aggregated_statistics", 
    'Portfolio factors save path': f"./experiments/{experiment_name}/portfolio_factors", 
    
    # -------------------------------------------------------------------------
    # Hyperparameters (hardcoded from hyperparam tuning)
    # -------------------------------------------------------------------------
    
    'Lookback window': 7,
    'Deterministic' : True,
    'Checkpoint frequency': 100,

    'Visual agent hyperparameters': {
        'Learning rate': 0.0004803451981619835,
        'Batch size': 64,
        'Rollout steps': 1280,
        'Gamma': 0.9455662787843444,
        'GAE lambda': 0.9075020500047098,
        'Clip range': 0.22685009929852115,
        'Entropy coefficient': 0.003963669759659864,
        'VF coefficient': 0.2625257871626315,
        'Max grad norm': 0.9177826027374991,
        'Epochs': 3,
        'Feature dim': 256
    },
    
    'Numeric agent hyperparameters': {
        'Learning rate': 0.001,
        'Batch size': 64,
        'Rollout steps': 2048,
        'Gamma': 0.99,
        'GAE lambda': 0.95,
        'Clip range': 0.2,
        'Entropy coefficient': 0.0,
        'VF coefficient': 0.5,
        'Max grad norm': 0.5,
        'Epochs': 10
    },
    
    # -------------------------------------------------------------------------
    # Experiment Design
    # -------------------------------------------------------------------------
    
    'K folds': 3,
    'Walk throughs': 3,
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
    'Start date': '2023-01-01',
    'End date': '2024-12-30',
    'Update frequency': '1d',
    
    # Features / target
    'Features': ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA', 'RSI', 'OBV'],
    'Target': 'Close',
    'GAF periods': 14, 
    'SMA periods': 20,
    'RSI periods': 14,

    # Stocks
    'Tickers' : ['BIIB', 'BRO', 'FE', 'PPL', 'STE', 'ADM', 'CSGP', 'EL', 'NUE', 'RJF']
}
