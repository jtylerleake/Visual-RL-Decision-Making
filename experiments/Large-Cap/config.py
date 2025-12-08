
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
    
    'Lookback window': 21, 
    'Deterministic' : True,
    'Checkpoint frequency': 100,
    
    'Visual agent hyperparameters': {
        "Learning rate": 0.007724648895904556,
        "Batch size": 256,
        "Rollout steps": 768,
        "Gamma": 0.9751541972871537,
        "GAE lambda": 0.8168697074108542,
        "Clip range": 0.13089765432305783,
        "Entropy coefficient": 0.0006810482071785461,
        "VF coefficient": 0.5189919855270405,
        "Max grad norm": 0.3397638968259331,
        "Epochs": 12,
        "Feature dim": 512
    },
    
    'Numeric agent hyperparameters': {
        "Learning rate": 2.9561060538346987e-05,
        "Batch size": 128,
        "Rollout steps": 1024,
        "Gamma": 0.9291269679440013,
        "GAE lambda": 0.9347733318317,
        "Clip range": 0.23051280976314076,
        "Entropy coefficient": 0.0030861439285987734,
        "VF coefficient": 0.6100714700055692,
        "Max grad norm": 0.974842040895288,
        "Epochs": 7
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
    'Start date': '2004-12-30',
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

