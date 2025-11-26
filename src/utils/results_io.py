
from common.modules import os, json, np, pd, glob, Dict, Any
import pickle
import gzip
from src.utils.logging import get_logger, log_function_call
from datetime import datetime


def _convert_numpy_types(obj):
    """Recursively convert np types to native python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj


@log_function_call
def save_results(
    results: Dict,
    experiment_name: str,
    filepath: str = None,
    format: str = 'json',
    compress: bool = True,
    run_id: int = None
) -> str:
    """Save experiment results to pickle/json file"""
    logger = get_logger(experiment_name, run_id=run_id)
    
    try:
        
        if format == 'pickle':
            if compress:
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Results saved to {filepath}")
            
        elif format == 'json':
            # convert numpy types to native python types
            json_results = _convert_numpy_types(results)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filepath}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise


@log_function_call
def load_results(
    filepath: str,
    experiment_name: str = None
) -> Dict:

    logger = get_logger(experiment_name) if experiment_name else get_logger("load_results")
    
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        # Determine format from file extension
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Results loaded from {filepath} (JSON format)")
            
        elif filepath.endswith('.pkl.gz') or filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Results loaded from {filepath} (compressed pickle format)")
            
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            logger.info(f"Results loaded from {filepath} (pickle format)")
            
        else:
            # Try pickle first, then JSON
            try:
                with open(filepath, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"Results loaded from {filepath} (pickle format, auto-detected)")
            except:
                with open(filepath, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"Results loaded from {filepath} (JSON format, auto-detected)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        raise


def get_latest_results(
    experiment_name: str,
    format: str = 'pickle'
) -> str:

    results_dir = os.path.join("experiments", experiment_name, "results")
    
    if not os.path.exists(results_dir):
        return None
    
    # Get all result files matching the format
    if format == 'pickle':
        pattern = "*-results.pkl*"
    else:
        pattern = "*-results.json"
    
    files = glob.glob(os.path.join(results_dir, pattern))
    
    if not files:
        return None
    
    # Return the most recently modified file
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

