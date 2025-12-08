
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
        "Learning rate": 0.006085574081564753,
        "Batch size": 64,
        "Rollout steps": 1792,
        "Gamma": 0.9135368005698759,
        "GAE lambda": 0.8855026209266923,
        "Clip range": 0.16509059764342734,
        "Entropy coefficient": 1.4050262440033276e-06,
        "VF coefficient": 0.5926283242982189,
        "Max grad norm": 0.3785815456992928,
        "Epochs": 10,
        "Feature dim": 512
    },
    
    'Numeric agent hyperparameters': {
        "Learning rate": 0.00019517592931225192,
        "Batch size": 256,
        "Rollout steps": 768,
        "Gamma": 0.9785264241085438,
        "GAE lambda": 0.9397893992745946,
        "Clip range": 0.13644509294911933,
        "Entropy coefficient": 0.0831500254925209,
        "VF coefficient": 0.1087491937734965,
        "Max grad norm": 0.9054911898340441,
        "Epochs": 15
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
    'Tickers' : ['BIIB', 'BRO', 'FE', 'PPL', 'STE', 'ADM', 'CSGP', 'EL', 'NUE', 'RJF', 'STT', 'A', 'DHI', 'EBAY', 'EQT', 'HSY', 'RMD', 'ROK', 'VMC', 'VTR', 'D', 'GWW', 'OKE', 'PSA', 'URI', 'AFL', 'FDX', 'SPG', 'SRE', 'TFC', 'TSCO', 'FITB', 'MCHP', 'ROL', 'MTB', 'CTSH', 'WAB', 'CCL', 'ACGL', 'YUM', 'ROST', 'ROP', 'EA', 'PCAR', 'XEL', 'CMI', 'MSI', 'ADSK', 'ALL', 'NSC']
       
}
