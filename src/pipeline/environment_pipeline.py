
from common.modules import np, pd, List, Dict
from common.modules import GramianAngularField, MinMaxScaler
from common.modules import StocksEnv, DummyVecEnv, Monitor, spaces
from src.utils.logging import get_logger


# Global functions for multiprocessing compatibility

def make_single_stock_env(
    ticker: str,  
    features: List, 
    targets: List, 
    gaf_periods: int, 
    lookback_window: int, 
    logger = None,
    modality: str = 'image'
) -> Monitor:
    
    """
    Global function to create a single stock trading environment. This 
    function is outside the class to avoid pickle issues with multiprocessing
    
    :param ticker (str): stock to create env for
    :param features (list): feature set
    :param targets (list): target variable set
    :param gaf_periods (int): observation size
    :param lookback_window (int): agent lookback
    :param modality (str): image or numeric type
    
    """
    
    try:
        
        # create custom environment
        class SingleStockEnv(StocksEnv):
            """Custom environment for single stock trading with GAF data"""
            
            def __init__(self_inner):
                
                # create a dummy df to override parent class temporarily
                dummy = pd.DataFrame({
                    'dummy': [0] * len(features),
                    'price': targets,
                })
                frame_bound = (lookback_window, len(features))
                super().__init__(dummy, lookback_window, frame_bound)
                
                if modality == 'image': 
                    c = features[0].shape[0]
                    h = w = gaf_periods
                    self_inner.observation_space = spaces.Box(
                        low = 0.0,
                        high = 1.0, # min/max normalized
                        shape=(c, h, w),  # cnn expects (channels, height, width) format
                        dtype = np.float32
                    )
                    
                if modality == 'numeric': 
                    h = gaf_periods
                    w = features[0].shape[1]
                    self_inner.observation_space = spaces.Box(
                        low = 0.0,
                        high = 1.0,
                        shape = (h,w),
                        dtype = np.float32
                    )
                
            def _get_observation(self_inner):
                current_obs = self_inner.signal_features[self_inner._current_tick]
                return current_obs
                
            def _process_data(self_inner):
                i = self_inner.frame_bound[0] - self_inner.window_size
                j = self_inner.frame_bound[1]
                prices = np.array(targets[i:j])
                sequences = np.array(features[i:j])
                return prices, sequences

        return Monitor(SingleStockEnv()) # return monitor wrapped environment
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating environment for {ticker}: {e}")
        raise


class EnvironmentPipeline:
    
    """
    Data pipleine for assembling single-stock environments and multi-stock 
    vectorized environments from timeseries data
    
    :param experiment_name (str): experiment config being run
    :param timeseries (Dict): raw tabular stock data to process
    :param features (List): stock data features to use in environment creation
    :param target (str): response variable
    :param gaf_periods (int): window size for gaf matrix creation
    :param lookback_window (int): number of historical env frames for agent
    
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        timeseries:  Dict,
        features: List[str], 
        target: str, 
        gaf_periods: None,
        lookback_window: int, 
        run_id = None
    ) -> None:
        
        self.experiment_name = experiment_name
        self.timeseries = timeseries
        self.tickers = timeseries.keys()
        self.n_tickers = len(self.tickers)
        self.features = features
        self.target = target
        self.gaf_periods = gaf_periods
        self.lookback_window = lookback_window
        self.gaf_data = {}
        self.numeric_data = {}
        self.image_environments = {}
        self.numeric_environments = {}
        self.image_vec_environment = None
        self.numeric_vec_environment = None
        self.run_id = run_id
        self.logger = get_logger(experiment_name, run_id=run_id)
        self.logger.info(f"Initialized environment pipeline with {self.n_tickers} tickers")
    
    def build_gaf_dataset(self) -> bool:
        """Transform the pre-loaded timeseries dataset into GAF dataset"""
        
        try:
            self.logger.info("Building GAF dataset")

            # pre-fit the gaf transformer once with sample data
            self.logger.info("Pre-fitting GAF transformer...")
            sample_data = np.random.rand(1,self.gaf_periods)
            transformer = GramianAngularField(
                image_size = self.gaf_periods,
                method = 'summation'
            )
            transformer.fit(sample_data)
            self.logger.info("GAF transformer pre-fitted successfully")
            
            # assemble the gaf sequences and append to gaf_sequences
            for ticker, df in self.timeseries.items():
                
                self.logger.info(f"Processing GAF transformation for {ticker}")
                
                # prepare features for gaf transformation
                scaler = MinMaxScaler()
                feature_data = df[self.features].values
                scaled_feature_data = scaler.fit_transform(feature_data)

                # build and stack gaf image sequences
                gaf_sequences = self.build_gaf_sequences(
                    scaled_features = scaled_feature_data, 
                    gaf_transformer = transformer
                )
                
                if gaf_sequences:
                    self.gaf_data[ticker] = {
                        'sequences': gaf_sequences,
                        'targets': df[self.target].iloc[self.gaf_periods-1:].values.tolist()
                    }
                    self.logger.info(f"Created {len(gaf_sequences)} GAF sequences for {ticker}")
                else:
                    self.logger.warning(f"No GAF sequences created for {ticker}")
            
            if not self.gaf_data:
                self.logger.error("No GAF data created")
                return False
            
            self.logger.info(f"Successfully created GAF data for {len(self.gaf_data)} tickers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building GAF dataset: {e}")
            return False
    
    def build_gaf_sequences(
            self, 
            scaled_features: np.ndarray, 
            gaf_transformer: GramianAngularField
    ) -> List[np.ndarray]:
        """Create GAF sequences from scaled features data and transformer"""
        
        n_samples = len(scaled_features)
        n_features = len(self.features)
        
        # create all sliding window indices; extract all sliding windows
        indices = np.arange(self.gaf_periods-1, n_samples)
        windows = np.array([scaled_features[i-self.gaf_periods+1:i+1] for i in indices])
        # > shape = (n_windows, gaf_periods, n_features)
        
        # reshape for batch processing
        windows_reshaped = windows.transpose(0, 2, 1).reshape(-1, self.gaf_periods)
        # > shape = (n_windows * n_features, gaf_periods)
        
        # batch processing: transform a ll windows
        self.logger.info(f"Performing GAF transformation on {len(windows_reshaped)} windows...")
        all_gaf = gaf_transformer.transform(windows_reshaped)
        # shape = (n_windows * n_features, gaf_periods, gaf_periods)
        
        # vectorized reshape back to individual sequences
        all_gaf = all_gaf.reshape(len(indices), n_features, self.gaf_periods, self.gaf_periods)
        # shape = (n_windows, n_features, gaf_periods, gaf_periods)
        
        # convert to list of multi-channel gaf images
        gaf_sequences = []
        for i in range(len(indices)):
            stacked_gaf = np.stack([all_gaf[i, j] for j in range(n_features)], axis=0)
            gaf_sequences.append(stacked_gaf)
        
        self.logger.info(f"Created {len(gaf_sequences)} GAF sequences")
        return gaf_sequences
    
    def build_numeric_dataset(self) -> bool:
        """Transform pre-loaded timeseries into numeric sequences dataset"""
        
        try:
            self.logger.info("Building numeric dataset")
            
            # assemble the numeric sequences for each ticker
            for ticker, df in self.timeseries.items():
                
                self.logger.info(f"Processing numeric sequences for {ticker}")
                
                # prepare features for numeric sequences (scale same as GAF)
                scaler = MinMaxScaler()
                feature_data = df[self.features].values
                scaled_feature_data = scaler.fit_transform(feature_data)
                
                # build numeric sequences using sliding windows
                numeric_sequences = self.build_numeric_sequences(
                    scaled_features = scaled_feature_data
                )
                
                if numeric_sequences:
                    self.numeric_data[ticker] = {
                        'sequences': numeric_sequences,
                        'targets': df[self.target].iloc[self.gaf_periods-1:].values.tolist()
                    }
                    self.logger.info(f"Created {len(numeric_sequences)} numeric sequences for {ticker}")
                else:
                    self.logger.warning(f"No numeric sequences created for {ticker}")
            
            if not self.numeric_data:
                self.logger.error("No numeric data created")
                return False
            
            self.logger.info(f"Successfully created numeric data for {len(self.numeric_data)} tickers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building numeric dataset: {e}")
            return False
    
    def build_numeric_sequences(
        self, 
        scaled_features: np.ndarray
    ) -> List[np.ndarray]:
        """Create numeric sequences from scaled features data using sliding windows"""
        
        n_samples = len(scaled_features)
        
        # create all sliding window indices; extract all sliding windows
        indices = np.arange(self.gaf_periods-1, n_samples)
        windows = np.array([scaled_features[i-self.gaf_periods+1:i+1] for i in indices])
        # > shape = (n_windows, gaf_periods, n_features)
        
        # convert to list of numeric sequences (keep as tabular data)
        numeric_sequences = [windows[i] for i in range(len(indices))]
        
        self.logger.info(f"Created {len(numeric_sequences)} numeric sequences")
        return numeric_sequences
            
    def build_individual_environments(self, modality: str = 'image') -> bool:
        """Create 'SingleStockEnv' environments for each ticker"""
        
        try:
            self.logger.info("Creating individual trading environments")
            for ticker in self.tickers:
                self.logger.info(f"Creating environment for {ticker}")
                
                features = targets = None
                
                if modality == 'image':
                    features = self.gaf_data[ticker]['sequences']
                    targets = self.gaf_data[ticker]['targets']
                    
                if modality == 'numeric':
                    features = self.numeric_data[ticker]['sequences']
                    targets = self.numeric_data[ticker]['targets']
                
                env = make_single_stock_env(
                    ticker, 
                    features,
                    targets,
                    self.gaf_periods, 
                    self.lookback_window, 
                    logger = self.logger,
                    modality = modality
                )
                
                if modality == 'image': self.image_environments[ticker] = env
                if modality == 'numeric': self.numeric_environments[ticker] = env

                self.logger.info(f"Created environment for {ticker}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating environments: {e}")
            return False
    
    def build_vectorized_environment(self, modality) -> bool:
        """Create a multi-process vectorized environment containing all single
        stock environments using DummyVecEnv"""
        
        try:
            self.logger.info(f"""Creating vectorized environment with
            {len(self.tickers)} single stock environments""")
            
            # create environment factory functions using existing environments
            env_fns = []
            for ticker in self.tickers:
                if modality == 'image' and ticker in self.image_environments:
                    env_fns.append(lambda env=self.image_environments[ticker]: env)
                if modality == 'numeric' and ticker in self.numeric_environments:
                    env_fns.append(lambda env=self.numeric_environments[ticker]: env)
            if not env_fns:
                self.logger.error("No valid environment functions created")
                return False
            
            # create DummyVecEnv
            if modality == 'image': self.image_vec_environment = DummyVecEnv(env_fns)
            if modality == 'numeric': self.numeric_vec_environment = DummyVecEnv(env_fns)
            self.logger.info("Successfully created vectorized environments")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating vectorized environment: {e}")
            return False
    
    def exe_env_pipeline(self, modality: str = 'image') -> bool:
        """Build the multi-process vectorized environments"""
        
        try:
            
            self.logger.info("Building environment pipeline...")
            
            if modality == 'image': # build the gaf dataset 
                self.logger.info("Step 1: Building GAF dataset...")
                if not self.build_gaf_dataset():
                    self.logger.error("Failed to build GAF dataset")
                    return False
            
            if modality == 'numeric': # build the numeric dataset
                self.logger.info("Step 1: Building numeric dataset...")
                if not self.build_numeric_dataset():
                    self.logger.error("Failed to build numeric dataset")
                    return False
    
            # create individual environments
            self.logger.info("Step 2: Creating individual environments...")
            if not self.build_individual_environments(modality):
                self.logger.error("Failed to build GAF environments")
                return False
        
            # create the multi-process vectorized environment
            self.logger.info("Step 3: Creating vectorized environment...")
            if not self.build_vectorized_environment(modality):
                self.logger.error("Failed to build vectorized GAF environment")
                return False
            
            self.logger.info("GAF pipeline built successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error building GAF pipeline: {e}")
            return False



if __name__ == "__main__":
    
    from src.utils.configurations import load_config
    from src.pipeline.data_pipeline import DataPipeline
    
    experiment_name = 'Mini'
    experiment_config = load_config(experiment_name)
    
    DATA = DataPipeline(experiment_name)
    Data = DATA.exe_data_pipeline(experiment_config)
    
    ENV = EnvironmentPipeline(
        experiment_name,
        timeseries = Data,
        features = experiment_config.get('Features'),
        target = experiment_config.get('Target'),
        gaf_periods = experiment_config.get('GAF periods'),
        lookback_window = experiment_config.get('Lookback window')
    )
    
    ENV.exe_env_pipeline(modality = 'image')
    ENV.exe_env_pipeline(modality = 'numeric')
    
   