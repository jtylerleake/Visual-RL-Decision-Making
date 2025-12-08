
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
        "Learning rate": 0.009991695256726241,
        "Batch size": 256,
        "Rollout steps": 1024,
        "Gamma": 0.938727893940406,
        "GAE lambda": 0.8606305971607126,
        "Clip range": 0.25338881158413223,
        "Entropy coefficient": 1.0963324216939614e-08,
        "VF coefficient": 0.7193058918642087,
        "Max grad norm": 0.5663196576086293,
        "Epochs": 11,
        "Feature dim": 128
    },
    
    'Numeric agent hyperparameters': {
        "Learning rate": 0.00675157540532829,
        "Batch size": 128,
        "Rollout steps": 1280,
        "Gamma": 0.9304212319557053,
        "GAE lambda": 0.8746134006799539,
        "Clip range": 0.21776979693194481,
        "Entropy coefficient": 5.594233133621562e-05,
        "VF coefficient": 0.5018271818820856,
        "Max grad norm": 0.6058207330185965,
        "Epochs": 10
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
    
    # Stocks
    'Tickers' : ['ARE', 'APA', 'AVY', 'BALL', 'BAX', 'BBY', 'CHRW', 'CPT', 'DRI', 'DECK', 'ESS', 'FFIV', 'FRT', 'HAS', 'HPQ', 'INCY', 'IFF', 'IP', 'JBHT', 'LH', 'MAA', 'NTAP', 'NDSN', 'NTRS', 'NVR', 'PNW', 'PPG', 'PTC', 'PHM', 'RVTY', 'SWK', 'TROW', 'TXT', 'COO', 'TRMB', 'TYL', 'UHS', 'WAT', 'WST', 'CPB', 'IPG', 'TAP', 'MGM', 'BXP', 'SNA', 'LII', 'KIM', 'GPC', 'STLD', 'VRSN']
       
}
