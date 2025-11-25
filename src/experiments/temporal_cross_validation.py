
from common.modules import np, pd, List, Dict, random, os, glob, PPO
from src.utils.logging import get_logger
from src.utils.metrics import compute_performance_metrics, aggregate_cross_validation_results
from src.utils.results_io import save_results
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.environment_pipeline import EnvironmentPipeline
from src.models.visual_agent import VisualAgent
from src.models.numeric_agent import NumericAgent
from src.models.benchmarks import MACD, SignR, BuyAndHold, Random


class TemporalCrossValidation:
    
    """
    Streamlined temporal walk-forward K-fold cross validation experiment.
    
    :param experiment_name (str): name of experiment config being run
    :param config (dict): experiment configuration dictionary
    :param run_id (int): experiment run differentiator id
    
    """
    
    def __init__(
        self, 
        experiment_name, 
        config,
        run_id = None,
    ) -> None:
        
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.logger = get_logger(experiment_name, run_id=run_id)
        self.config = config
        
        # cache config values for performance
        self.num_folds = config.get('K folds')
        self.num_time_windows = config.get('Time splits')
        self.stratification_type = config.get('Stratification type')
        self.random_seed = config.get('Random seed')
        self.walk_throughs = config.get('Walk throughs')
        self.features = config.get('Features')
        self.target = config.get('Target')
        self.gaf_periods = config.get('GAF periods')
        self.lookback_window = config.get('Lookback window')
        self.checkpoint_freq = config.get('Checkpoint frequency')
        self.deterministic = config.get('Deterministic', True)
        self.visual_save_path = config.get('Visual agent save path')
        self.numeric_save_path = config.get('Numeric agent save path')
        
        # seed reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self.logger.info("TemporalCrossValidation initialized")
    
    def get_folds(
        self, 
        stocks,
    ) -> Dict[int, List[str]]:
        """Partitions stocks into folds based on stratification type"""
        if not stocks or self.num_folds <= 1 or len(stocks) < self.num_folds:
            self.logger.error("Invalid fold configuration")
            return {}
        
        if self.stratification_type == 'random':
            shuffled = stocks.copy()
            random.shuffle(shuffled)
            fold_assignments = {}
            stocks_per_fold = len(stocks) // self.num_folds
            remainder = len(stocks) % self.num_folds
            
            start_idx = 0
            for fold in range(self.num_folds):
                fold_size = stocks_per_fold + (1 if fold < remainder else 0)
                end_idx = start_idx + fold_size
                fold_assignments[fold] = shuffled[start_idx:end_idx]
                start_idx = end_idx
            
            self.logger.info(f"Assigned {len(stocks)} stocks to {self.num_folds} folds")
            return fold_assignments
        elif self.stratification_type == 'sector balanced':
            raise NotImplementedError("Sector balanced stratification not implemented")
        else:
            self.logger.error(f"Unknown stratification: {self.stratification_type}")
            return {}
    
    def get_frame_bounds(
        self, 
        timeseries
    ) -> Dict[str, List]:
        """Creates frame bounds for temporal walk-forward windows"""
        df = next(iter(timeseries.values()))
        num_sequences = len(df) - self.gaf_periods + 1
        
        # Get and normalize ratios
        train_ratio = self.config.get('Training ratio')
        val_ratio = self.config.get('Validation ratio')
        eval_ratio = self.config.get('Evaluation ratio')
        total_ratio = train_ratio + val_ratio + eval_ratio
        
        if abs(total_ratio - 1.0) > 0.01:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            eval_ratio /= total_ratio
        
        # Calculate window sizes
        train_window_size = int((num_sequences * train_ratio) / self.walk_throughs) - 1
        val_window_size = int(num_sequences * val_ratio)
        test_window_size = int(num_sequences * eval_ratio)
        
        # Generate bounds for each walk-through
        train_bounds, val_bounds, eval_bounds = [], [], []
        walk_start = 0
        
        for _ in range(self.walk_throughs):
            train_end = walk_start + train_window_size
            val_start = train_end + 1
            val_end = val_start + val_window_size
            test_start = val_end + 1
            test_end = test_start + test_window_size
            
            train_bounds.append([walk_start, train_end])
            val_bounds.append([val_start, val_end])
            eval_bounds.append([test_start, test_end])
            walk_start = train_end + 1
        
        self.logger.info(f"Created frame bounds for {self.walk_throughs} walk-throughs")
        return {'training': train_bounds, 'validation': val_bounds, 'evaluation': eval_bounds}
    
    def _set_env_frame_bounds(
        self, 
        env_pipeline, 
        bounds,
        modalities: List[str] = ['image', 'numeric']
    ) -> None:
        """Helper to set frame bounds for all environments in a pipeline"""
        for modality in modalities:
            env_dict = getattr(env_pipeline, f'{modality}_environments', {})
            if not env_dict:
                continue
            for monitor in env_dict.values():
                env = monitor.env
                env.frame_bound = tuple(bounds)
                env._start_tick = bounds[0]
                env._end_tick = bounds[1]
                env._process_data()
                env.reset()
    
    def _create_env_pipeline(
        self, 
        timeseries,
    ) -> EnvironmentPipeline:
        """Create and build environment pipeline"""
        env = EnvironmentPipeline(
            experiment_name=self.experiment_name,
            timeseries=timeseries,
            features=self.features,
            target=self.target,
            gaf_periods=self.gaf_periods,
            lookback_window=self.lookback_window,
            run_id=self.run_id
        )
        
        if not env.exe_env_pipeline('image') or not env.exe_env_pipeline('numeric'):
            raise RuntimeError("Failed to build environments")
        
        return env
    
    def assemble_environments(
        self, 
        fold_idx,
        fold_assignments, 
        timeseries,
    ) -> tuple:
        """Build vectorized environments for training/testing"""
        test_stocks = fold_assignments[fold_idx]
        train_stocks = [s for fold_stocks in fold_assignments.values() 
                       for s in fold_stocks if s not in test_stocks]
        
        train_set = {s: timeseries[s] for s in train_stocks}
        test_set = {s: timeseries[s] for s in test_stocks}
        
        train_env = self._create_env_pipeline(train_set)
        test_env = self._create_env_pipeline(test_set)
        
        return train_env, test_env
    
    def select_best_checkpoint(
        self, 
        checkpoint_files,
        agent, 
        agent_class, 
        train_env, 
        modality
    ):
        """Evaluate checkpoints and return best model"""
        if not checkpoint_files or checkpoint_files == [None]:
            return agent
        
        best_agent = agent
        best_score = float('-inf')
        model_class = PPO
        vec_env = getattr(train_env, f'{modality}_vec_environment')
        
        for checkpoint_path in checkpoint_files:
            if not checkpoint_path:
                continue
            
            try:
                checkpoint_agent = agent_class(train_env, self.config)
                checkpoint_agent.model = model_class.load(checkpoint_path, env=vec_env)
                val_results = self.evaluate(checkpoint_agent, train_env, modality)
                
                if val_results:
                    avg_return = np.mean([m.get('cumulative return', 0) 
                                        for m in val_results.values()])
                    if avg_return > best_score:
                        best_score = avg_return
                        best_agent = checkpoint_agent
            except Exception as e:
                self.logger.warning(f"Error loading checkpoint {checkpoint_path}: {e}")
                continue
        
        return best_agent
    
    def evaluate(
        self, 
        strategy, 
        environment, 
        modality,
    ) -> Dict:
        """Evaluate a strategy in an environment"""
        stock_metrics = {}
        # For benchmark strategies, use image environments (they work with raw data)
        if hasattr(strategy, 'strategy_type') and strategy.strategy_type == 'Benchmark':
            env_dict = getattr(environment, 'image_environments', {})
        else:
            env_dict = getattr(environment, f'{modality}_environments', {})
        
        for stock, monitor in env_dict.items():
            portfolio_factors = []
            episode_reward = 0
            actions = []
            
            env = monitor.env
            obs, info = env.reset()
            obs = obs[0] if isinstance(obs, tuple) else obs
            
            portfolio_factors.append(info.get('total_profit', 1.0))
            
            done = False
            step_count = 0
            
            while not done:
                # Prepare observation
                obs_batch = np.expand_dims(obs, axis=0) if len(obs.shape) == 3 else obs
                
                # Get action
                if hasattr(strategy, 'strategy_type') and strategy.strategy_type == 'Agent':
                    action, _ = strategy.model.predict(obs_batch, deterministic=self.deterministic)
                else:
                    # For benchmark strategies
                    data = environment.timeseries[stock]
                    last_action = actions[-1] if actions else None
                    action = strategy.predict(data, last_action, step_count)
                
                if isinstance(action, np.ndarray):
                    action = int(action.item()) if action.ndim == 0 else int(action[0])
                elif isinstance(action, list):
                    action = int(action[0]) if len(action) > 0 else 0
                else:
                    action = int(action)
                actions.append(action)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                portfolio_factors.append(info.get('total_profit', 1.0))
                episode_reward += reward
                step_count += 1
            
            # Compute metrics
            metrics = compute_performance_metrics(
                portfolio_factors = portfolio_factors,
                start_date = self.config.get('Start date'),
                end_date = self.config.get('End date'),
                sig_figs = 4
            )
            
            unique_actions, action_counts = np.unique(actions, return_counts=True)
            
            stock_metrics[stock] = {
                'portfolio factors': portfolio_factors,
                'episode reward': episode_reward,
                'actions': actions,
                'action distribution': dict(zip(unique_actions.tolist(), action_counts.tolist())),
                'num steps': step_count,
                'cumulative return': metrics['cumulative return'],
                'annualized return': metrics['annualized cumulative return'],
                'sharpe ratio': metrics['sharpe ratio'],
                'sortino ratio': metrics['sortino ratio'],
                'max drawdown': metrics['max drawdown'],
            }
        
        return stock_metrics
    
    def train_validate_test(
        self, 
        train_env, 
        test_env, 
        frame_bounds,
        fold_idx, 
        window_idx, 
        phase,
    ) -> Dict:
        """Train, validate, and test for a fold/window combination"""
        train_bounds = frame_bounds['training'][window_idx]
        val_bounds = frame_bounds['validation'][window_idx]
        eval_bounds = frame_bounds['evaluation'][window_idx]
        
        checkpoint_dirs = {
            'image': os.path.join(self.visual_save_path, 
                                 f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'),
            'numeric': os.path.join(self.numeric_save_path,
                                  f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints')
        }
        
        if phase == 'training':
            # Set training bounds
            self._set_env_frame_bounds(train_env, train_bounds)
            
            # Train agents
            visual_agent = VisualAgent(train_env, self.config)
            numeric_agent = NumericAgent(train_env, self.config)
            
            visual_agent.train(checkpoint_save_path=checkpoint_dirs['image'],
                              checkpoint_freq=self.checkpoint_freq)
            numeric_agent.train(checkpoint_save_path=checkpoint_dirs['numeric'],
                               checkpoint_freq=self.checkpoint_freq)
            return {}
        
        # Inference phase
        # Find checkpoints
        image_checkpoints = sorted(glob.glob(os.path.join(
            checkpoint_dirs['image'], 'checkpoint_*.zip'))) or [None]
        numeric_checkpoints = sorted(glob.glob(os.path.join(checkpoint_dirs['numeric'],
                                                            'checkpoint_*.zip'))) or [None]
        
        # Set validation bounds and evaluate
        self._set_env_frame_bounds(train_env, val_bounds)
        
        visual_agent = VisualAgent(train_env, self.config)
        numeric_agent = NumericAgent(train_env, self.config)
        
        best_visual = self.select_best_checkpoint(image_checkpoints, visual_agent,
                                                  VisualAgent, train_env, 'image')
        best_numeric = self.select_best_checkpoint(numeric_checkpoints, numeric_agent,
                                                   NumericAgent, train_env, 'numeric')
        
        # Set test bounds and evaluate
        self._set_env_frame_bounds(test_env, eval_bounds)
        
        visual_results = self.evaluate(best_visual, test_env, 'image')
        numeric_results = self.evaluate(best_numeric, test_env, 'numeric')
        
        # Evaluate baselines
        baseline_results = {}
        for baseline_class in [MACD, SignR, BuyAndHold, Random]:
            baseline = baseline_class()
            baseline_results[f'{baseline_class.__name__} results'] = \
                self.evaluate(baseline, test_env, 'image')
        
        return {
            'Visual agent results': visual_results,
            'Numeric agent results': numeric_results,
            **baseline_results
        }
    
    def exe_cross_validation(
        self, 
        fold_assignments,
        frame_bounds,
        timeseries, 
        phase
    ) -> Dict:
        """Execute cross-validation across folds and windows"""
        fold_results = {}
        
        for fold_idx in range(self.num_folds):
            self.logger.info(f"Processing fold {fold_idx+1}/{self.num_folds}")
            
            train_env, test_env = self.assemble_environments(fold_idx, fold_assignments, timeseries)
            window_results = {}
            
            for window_idx in range(self.walk_throughs):
                self.logger.info(f"  Window {window_idx+1}/{self.walk_throughs}")
                
                window_metrics = self.train_validate_test(
                    train_env, test_env, frame_bounds, fold_idx, window_idx, phase
                )
                
                if phase == 'inference' and window_metrics:
                    window_results[window_idx] = window_metrics
            
            if phase == 'inference':
                fold_results[fold_idx+1] = window_results
        
        return fold_results
    
    def exe_experiment(
        self, 
        phase,
    ) -> Dict:
        """Execute the complete cross-validation experiment"""
        self.logger.info("Starting Temporal Cross-Validation Experiment")
        
        try:
            # Step 1: Load data
            self.logger.info("Loading timeseries data...")
            data_pipeline = DataPipeline(self.experiment_name, run_id=self.run_id)
            timeseries_data = data_pipeline.exe_data_pipeline(self.config)
            if not timeseries_data:
                raise RuntimeError("Failed to load timeseries data")
            
            # Step 2: Create folds
            self.logger.info("Creating fold assignments...")
            stocks = list(timeseries_data.keys())
            fold_assignments = self.get_folds(stocks)
            if not fold_assignments:
                raise RuntimeError("Failed to create fold assignments")
            
            # Step 3: Create frame bounds
            self.logger.info("Creating temporal frame bounds...")
            frame_bounds = self.get_frame_bounds(timeseries_data)
            if not frame_bounds:
                raise RuntimeError("Failed to create frame bounds")
            
            # Step 4: Execute cross-validation
            self.logger.info("Executing cross-validation...")
            results = self.exe_cross_validation(fold_assignments, frame_bounds, 
                                               timeseries_data, phase)
            
            # Step 5: Aggregate results
            if phase == 'inference':
                self.logger.info("Aggregating results...")
                aggregated_results = aggregate_cross_validation_results(results)
                self.logger.info("Experiment completed successfully")
                return aggregated_results
            
            self.logger.info("Training phase completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in experiment: {e}")
            return {}
    
    def save_experiment_results(
        self, 
        results: Dict, 
        filepath: str = None, 
        format: str = 'pickle', 
        compress: bool = True
    ) -> str:
        """Save experiment results"""
        return save_results(
            results=results,
            experiment_name=self.experiment_name,
            filepath=filepath,
            format=format,
            compress=compress,
            run_id=self.run_id
        )


if __name__ == "__main__":
    
    from src.utils.configurations import load_config

    mini_config = load_config('Mini')
    mini_experiment = TemporalCrossValidation('Mini', mini_config)
    mini_experiment.exe_experiment('inference')

    '''
    large_cap_config = load_config('Large-Cap')
    medium_cap_config = load_config('Medium-Cap')
    small_cap_config = load_config('Small-Cap')
    
    large_cap_experiment = TemporalCrossValidation('Large-Cap', large_cap_config)
    medium_cap_experiment = TemporalCrossValidation('Medium-Cap', medium_cap_config)
    small_cap_experiment = TemporalCrossValidation('Small-Cap', small_cap_config)
    
    large_cap_experiment.exe_experiment('training')
    medium_cap_experiment.exe_experiment('training')
    small_cap_experiment.exe_experiment('training')
    '''
    
