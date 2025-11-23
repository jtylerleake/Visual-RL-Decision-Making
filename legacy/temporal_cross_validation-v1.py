
from common.modules import np, pd, List, Dict, random, os, glob, PPO
from src.utils.logging import log_function_call, get_logger
from src.utils.metrics import compute_performance_metrics, aggregate_cross_validation_results
from src.utils.results_io import save_results, load_results, get_latest_results
from src.pipeline.data_pipeline import DataPipeline
from src.pipeline.environment_pipeline import EnvironmentPipeline
from src.models.visual_agent import VisualAgent
from src.models.numeric_agent import NumericAgent
from src.models.benchmarks import MACD, SignR, BuyAndHold, Random


class TemporalCrossValidation:
    
    """
    Temporal walk-forward K-fold cross validation experiment. Model evaluation 
    metrics are collected and aggregated across folds/windows.
    
    :param experiment_name (str): name of experiment config being run
    :param config (dict): experiment configuration dictionary
    :param run_id (int): experiment run differentiator id
    """
    
    def __init__(
        self, 
        experiment_name: str, 
        config: Dict, 
        run_id = None
    ) -> None:
        try:
            # basic configuration
            self.experiment_name = experiment_name
            self.run_id = run_id
            self.logger = get_logger(experiment_name, run_id=run_id)
            self.config = config
            self.num_folds = self.config.get('K folds') 
            self.num_time_windows = self.config.get('Time splits')
            self.stratification_type = self.config.get('Stratification type')
            self.random_seed = self.config.get('Random seed')
            # seed reproducibility
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
        except Exception as e: 
            self.logger.error(f"""Could not initialize temporal cross validation. 
            Error with configuation file parameters: {e}""")
        self.logger.info("Experiment Manager initialized")
    
    @log_function_call
    def get_folds(
        self, 
        stocks: List[str]
    ) -> Dict[int, List[str]]:
        """Partitions stocks into folds based on stratification type"""
        
        try:
            if not stocks:
                self.logger.error("No stocks provided for fold assignment")
                return {}
            if self.num_folds <= 1:
                self.logger.error("Number of folds must be greater than 1")
                return {}
            if len(stocks) < self.num_folds:
                self.logger.error(f"""Number of stocks ({len(stocks)}) is less 
                than number of folds ({self.num_folds})""")
                return {}

            # partition using stratification type == random
            if self.stratification_type == 'random':
                shuffled_stocks = stocks.copy()
                random.shuffle(shuffled_stocks)
                fold_assignments = {}
                stocks_per_fold = len(stocks) // self.num_folds
                remainder = len(stocks) % self.num_folds
                start_idx = 0
                for fold in range(self.num_folds):
                    # add one extra stock to first 'remainder' folds
                    fold_size = stocks_per_fold + (1 if fold < remainder else 0)
                    end_idx = start_idx + fold_size
                    fold_assignments[fold] = shuffled_stocks[start_idx:end_idx]
                    start_idx = end_idx
                self.logger.info(f"""Randomly assigned {len(stocks)} stocks to 
                {self.num_folds} folds""")
            
            # partition using stratification type == sector balanced
            elif self.stratification_type == 'sector balanced':
                raise NotImplementedError
            
            else:
                self.logger.error(f"""Unknown stratification strategy: 
                {self.stratification_strategy}""")
                return {}
            
            self.fold_assignments = fold_assignments
            return fold_assignments
            
        except Exception as e:
            self.logger.error(f"Error assigning stocks to folds: {e}")
            return {}
    
    @log_function_call
    def get_frame_bounds(
        self, 
        timeseries
    ) -> Dict[str, List]:
        """Creates frame bounds for environments based on number of total data 
        points, temporal window size, and train/val/test ratios from config"""
        
        try:
            
            # fetch config parameters
            walk_throughs = self.config.get('Walk throughs')
            training_ratio = self.config.get('Training ratio')
            validation_ratio = self.config.get('Validation ratio')
            evaluation_ratio = self.config.get('Evaluation ratio')

            # fetch number of data points in dataframe
            df = timeseries[next(iter(timeseries))]
            num_gaf_imgs = len(df) - self.config.get('GAF periods') + 1
            
            # validate ratios sum to 1.0
            total_ratio = training_ratio + validation_ratio + evaluation_ratio
            if abs(total_ratio - 1.0) > 0.01:
                self.logger.warning(f"Ratios sum to {total_ratio}, not 1.0. Normalizing...")
                training_ratio /= total_ratio
                validation_ratio /= total_ratio
                evaluation_ratio /= total_ratio
                
            # calculate window sizes in data points
            train_window_size = int((num_gaf_imgs * training_ratio) / walk_throughs) - 1
            val_window_size = int(num_gaf_imgs * validation_ratio)
            test_window_size = int(num_gaf_imgs * evaluation_ratio)
            
            # calculate frame bounds for each walk-through
            train_bounds = []
            validation_bounds = []
            evaluation_bounds = []
            
            walk_start = 0
            for walk_idx in range(walk_throughs):
                train_start = walk_start
                train_end = train_start + train_window_size
                val_start = train_end + 1
                val_end = val_start + val_window_size
                test_start = val_end + 1
                test_end = test_start + test_window_size
                train_bounds.append([train_start, train_end])
                validation_bounds.append([val_start, val_end])
                evaluation_bounds.append([test_start, test_end])
                walk_start = train_end + 1
            
            # aggregate bounds and return
            frame_bounds = {
                'training': train_bounds,
                'validation': validation_bounds,
                'evaluation': evaluation_bounds
            }
            
            self.logger.info(f"""Created frame bounds for {len(train_bounds)} walk-throughs""")
            return frame_bounds
            
        except Exception as e:
            self.logger.error(f"Error creating temporal window frame bounds: {e}")
            return {}
    
    @log_function_call
    def exe_experiment(self, phase : str = ['training', 'inference']) -> Dict:
        """Execute the cross-validation with temporal walk-through experiment"""
        
        self.logger.h1("Executing Temporal Walk-Forward RL Performance Experiment")
        try:
            
            # Step 1: load and preprocess data
            self.logger.info("Step 1: Preparing timeseries data pipeline...")
            timeseries_pipeline = DataPipeline(self.experiment_name, run_id=self.run_id)
            timeseries_data = timeseries_pipeline.exe_data_pipeline(self.config)
            if not timeseries_data:
                self.logger.error("Failed to prepare stock data")
                return {}
            
            # Step 2: get fold assignments
            self.logger.info("Step 2: Creating fold assignments...")
            stocks = list(timeseries_data.keys())
            fold_assignments = self.get_folds(stocks)
            if not fold_assignments:
                self.logger.error("Failed to create fold assignments")
                return {}
            
            # Step 3: get temporal window frame bounds
            self.logger.info("Step 3: Creating temporal window frame bounds...")
            frame_bounds = self.get_frame_bounds(timeseries_data)
            if not frame_bounds:
                self.logger.error("Failed to create temporal window frame bounds")
                return {}
            
            
            
            # Step 4: execute cross-validation
            self.logger.info("Step 4: Executing cross-validation...")
            results = self.exe_cross_validation(fold_assignments, frame_bounds, timeseries_data, phase)
            
            # Step 5: aggregate results
            self.logger.info("Step 5: Aggregating results...")
            aggregated_results = aggregate_cross_validation_results(results)
            
            self.logger.h1("Cross-Validation Experiment Completed")
            return aggregated_results
            
        except Exception as e:
            self.logger.error(f"Error executing cross-validation: {e}")
            return {}
    
    @log_function_call
    def exe_cross_validation(
        self, 
        fold_assignments: Dict[int, List[str]], 
        frame_bounds: Dict[str, List[List]],
        timeseries: Dict[str, pd.DataFrame],
        phase: str
    ) -> Dict:
        """Cross validaion overhead"""        
        try:
            
            # K-fold outer loop
            fold_results = {}
            
            for fold_idx in range(self.num_folds):
                window_results = {}
                
                train_env, test_env = self.assemble_environments(
                    fold_idx, 
                    fold_assignments, 
                    timeseries,
                )
                
                # Temporal walk-forward inner loop
                for window_idx in range(config.get('Walk throughs')):
                    self.logger.info(f"Training window {window_idx+1} in fold {fold_idx+1}")
                    
                    window_metrics = self.train_validate_test(
                        train_env,
                        test_env,
                        frame_bounds,
                        fold_idx, 
                        window_idx,
                        phase
                    )
                    
                    if phase == 'inference':
                        if window_metrics:
                            window_results[window_idx] = window_metrics
                            self.logger.info(f"Completed fold {fold_idx+1}, window {window_idx +1}")    
                        else:
                            self.logger.warning(f"Failed fold {fold_idx+1}, window {window_idx+1}")
                            
                    if phase == 'training':
                        continue
                        
                fold_results[fold_idx+1] = window_results
            return fold_results
            
        except Exception as e:
            self.logger.error(f"Error executing fold validation: {e}")
            return {}
        
    @log_function_call
    def assemble_environments(
        self,
        fold_idx,
        fold_assignments,
        timeseries, 
    ) -> tuple: 
        """Build the vectorized GAF environments for training/validation/testing"""
        
        try:
        
            # isolate the training/vaidation data from the test data
            test_stocks = fold_assignments[fold_idx] 
            train_stocks = [
                stock for stocks in fold_assignments.values() for stock 
                in stocks if stock not in test_stocks
            ]
            train_set = {stock: timeseries[stock] for stock in train_stocks}
            test_set = {stock: timeseries[stock] for stock in test_stocks}
            
            # initialize the training/validation/test environment classes
            train_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                timeseries = train_set,
                features = self.config.get('Features'),
                target = self.config.get('Target'),
                gaf_periods = self.config.get('GAF periods'),
                lookback_window = self.config.get('Lookback window'),
                run_id = self.run_id
            )
            
            test_env = EnvironmentPipeline(
                experiment_name = self.experiment_name,
                timeseries = test_set,
                features = self.config.get('Features'),
                target = self.config.get('Target'),
                gaf_periods = self.config.get('GAF periods'),
                lookback_window = self.config.get('Lookback window'),
                run_id = self.run_id
            )
            
            # assemble the vectorized GAF environments
            if not train_env.exe_env_pipeline('image') or not train_env.exe_env_pipeline('numeric'):
                self.logger.error("Failed to build training environments")
                return {}
            
            self.logger.info(f"""Successfully created training environment with 
            {train_env.image_vec_environment.num_envs} environments""")
            
            if not test_env.exe_env_pipeline('image') or not test_env.exe_env_pipeline('numeric'):
                self.logger.error("Failed to build test environments")
                return {}
            
            self.logger.info(f"""Successfully created training environment with 
            {test_env.image_vec_environment.num_envs} environments""")
    
            return train_env, test_env
        
        except Exception as e: 
            self.logger.error(f"Error creating environments: {e}")
            return {}
        
    @log_function_call
    def select_best_checkpoint(
        self,
        checkpoint_files,
        agent,
        agent_class,
        train_env,
        modality,
    ) -> tuple:
        """Evaluate all checkpoints on validation set and return the best one"""
        
        try:
        
            # evaluate all checkpoints on validation
            self.logger.info("Evaluating checkpoints on validation set...")
            best_checkpoint = None
            best_validation_score = float('-inf')
            validation_results_by_checkpoint = {}
            best_agent = None
            best_validation_results = None 
            
            for checkpoint_path in checkpoint_files:
                if checkpoint_path:
                    checkpoint_num = os.path.basename(checkpoint_path).replace('checkpoint_', '').replace('.zip', '')
                    self.logger.info(f"Evaluating checkpoint {checkpoint_num}...")
                    model_class = PPO['CnnPolicy'] if modality == 'image' else PPO['MlpPolicy']
                    checkpoint_agent = agent_class(train_env, self.config)
                    checkpoint_agent.model = model_class.load(checkpoint_path, env=train_env.vec_environment)
                    eval_agent = checkpoint_agent
                else:
                    self.logger.info("Evaluating the final model...")
                    eval_agent = agent
                
                # evaluate on validation set
                val_results = self.evaluate(eval_agent, train_env)
                
                # calculate validation score (using average cumulative return across stocks)
                if val_results:
                    avg_cumulative_return = np.mean([
                        stock_metrics.get('cumulative return', 0)
                        for stock_metrics in val_results.values()
                    ])
                    validation_results_by_checkpoint[checkpoint_path or 'final'] = {
                        'results': val_results,
                        'score': avg_cumulative_return
                    }
                    
                    if avg_cumulative_return > best_validation_score:
                        best_validation_score = avg_cumulative_return
                        best_checkpoint = checkpoint_path
                        best_agent = eval_agent
                        best_validation_results = val_results
            
            self.logger.info(f"""Best checkpoint: {best_checkpoint or 'final model'} with 
            validation score: {best_validation_score:.4f}""")
            
            return best_agent
            
        except Exception as e:
            self.logger.error(f"Error selecting best checkpoint: {e}")
            return None, agent, None, {} # fallback to final model
    
    @log_function_call
    def train_validate_test(
        self,
        train_env,
        test_env,
        frame_bounds,
        fold_idx, 
        window_idx,
        phase
    ) -> Dict:
        """Train, validate, and test a model for a fold/window combination"""
        
        try:
            
            self.logger.info(f"Training model for fold {fold_idx+1}, window {window_idx+1}")
            
            training_bounds = frame_bounds['training'][window_idx]
            validation_bounds = frame_bounds['validation'][window_idx]
            evaluation_bounds = frame_bounds['evaluation'][window_idx]
            
            image_checkpoint_dir = os.path.join(
                self.config.get('Visual agent save path'),
                f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'
            )
            
            numeric_checkpoint_dir = os.path.join(
                self.config.get('Numeric agent save path'),
                f'fold_{fold_idx+1}_window_{window_idx+1}_checkpoints'
            )
            
            visual_agent = VisualAgent(train_env, self.config)
            numeric_agent = NumericAgent(train_env, self.config)
            
            # >>> train the visual and numeric agents with checkpointing
            
            if phase == 'training': 
            
                # set the frame bounds for the training environment
                for ticker, monitor in train_env.image_environments.items():
                    env = monitor.env
                    env.frame_bound = (training_bounds[0], training_bounds[1])
                    env._start_tick = training_bounds[0]
                    env._end_tick = training_bounds[1]
                    env._process_data()
                    env.reset()
                
                # set the frame bounds for the training environment
                for ticker, monitor in train_env.numeric_environments.items():
                    env = monitor.env
                    env.frame_bound = (training_bounds[0], training_bounds[1])
                    env._start_tick = training_bounds[0]
                    env._end_tick = training_bounds[1]
                    env._process_data()
                    env.reset()
                    
                # train with checkpoint saving
                self.logger.info("Training the models...")
                checkpoint_freq = self.config.get('Checkpoint frequency')
            
                visual_agent.train(
                    checkpoint_save_path = image_checkpoint_dir,
                    checkpoint_freq = checkpoint_freq
                )
                
                numeric_agent.train(
                    checkpoint_save_path = numeric_checkpoint_dir,
                    checkpoint_freq = checkpoint_freq
                )
            
                return False
            
            # >>> load pre-trained models and run evaluation
            
            if phase == 'inference': 
            
                # find all visual model checkpoints
                self.logger.info("Searching for model checkpoints")
                image_checkpoint_files = sorted(glob.glob(os.path.join(image_checkpoint_dir, 'checkpoint_*.zip')))
                numeric_checkpoint_files = sorted(glob.glob(os.path.join(numeric_checkpoint_dir, 'checkpoint_*.zip')))
                if not image_checkpoint_files:
                    self.logger.warning("No visual model checkpoints found, using final model")
                    image_checkpoint_files = [None]  # use final model as fallback
                if not numeric_checkpoint_files:
                    self.logger.warning("No numeric model checkpoints found, using final model")
                    numeric_checkpoint_files = [None] 
                
                # set the validation environment frame bounds
                for ticker, monitor in train_env.image_environments.items():
                    env = monitor.env
                    env.frame_bound = (validation_bounds[0], validation_bounds[1])
                    env._process_data()
                    env._start_tick = validation_bounds[0]
                    env._end_tick = validation_bounds[1]
                    env.reset()
                    
                for ticker, monitor in train_env.numeric_environments.items():
                    env = monitor.env
                    env.frame_bound = (validation_bounds[0], validation_bounds[1])
                    env._process_data()
                    env._start_tick = validation_bounds[0]
                    env._end_tick = validation_bounds[1]
                    env.reset()
                
                # evaluate checkpoints and select best model
                best_visual_agent = self.select_best_checkpoint(
                    checkpoint_files = image_checkpoint_files,
                    agent = visual_agent,
                    train_env = train_env,
                    modality = 'image'
                )

                best_numeric_agent = self.select_best_checkpoint(
                    checkpoint_files = numeric_checkpoint_files,
                    agent = numeric_agent,
                    train_env = train_env,
                    modality = 'numeric'
                )
                
                # set the frame bounds for the test environment
                for ticker, monitor in test_env.image_environments.items():
                    env = monitor.env
                    env.frame_bound = (evaluation_bounds[0], evaluation_bounds[1])
                    env._start_tick = evaluation_bounds[0]
                    env._end_tick = evaluation_bounds[1]
                    env._process_data()
                    env.reset()
                    
                for ticker, monitor in test_env.numeric_environments.items():
                    env = monitor.env
                    env.frame_bound = (evaluation_bounds[0], evaluation_bounds[1])
                    env._start_tick = evaluation_bounds[0]
                    env._end_tick = evaluation_bounds[1]
                    env._process_data()
                    env.reset()
                    
                # test the best checkpoint
                self.logger.info("Evaluating best model on test set...")
                visual_test_results = self.evaluate(best_visual_agent, test_env)
                numeric_test_results = self.evaluate(best_numeric_agent, test_env)
    
                # run baseline models
                macd_results = self.evaluate(MACD(), test_env)
                signr_results = self.evaluate(SignR(), test_env)
                long_results = self.evaluate(BuyAndHold(), test_env)
                random_results = self.evaluate(Random(), test_env)
    
                # Step 8: combine results and return 
                fold_window_results = {
                    'Visual agent results': visual_test_results,
                    'Numeric agent results': numeric_test_results,
                    'MACD results' : macd_results,
                    'SignR results' : signr_results,
                    'Long results' : long_results,
                    'Random results' : random_results,
                }
                
                self.logger.info(f"""Completed training/validation/test for 
                fold {fold_idx+1}, window {window_idx+1}""")
                return fold_window_results
        
        except Exception as e:
            self.logger.error(f"Error during Train/Validate/Test: {e}")
            return {}
        
    @log_function_call
    def evaluate(
        self,
        strategy, 
        environment
    ) -> Dict: 
        """Evaluate a trained model or benchmark strategy in an environment.
        Update individual and aggregated performance metrics by environment"""

        try:
            stock_metrics = {}
            deterministic = self.config.get('Deterministic', True)
            
            # Stock-by-Stock Evaluation
            for stock, monitor in environment.environments.items():
                
                # initialize performance metrics
                portfolio_factors = []
                episode_reward = 0
                actions = []

                # reset the environment; return first observation and dataset
                env = monitor.env
                obs, info = env.reset() 
                obs = obs[0] if isinstance(obs, tuple) else obs

                # store initial portfolio value
                initial_portfolio_factor = info.get('total_profit', 1.0)
                portfolio_factors.append(initial_portfolio_factor)

                done = False
                step_count = 0
                while not done:
                    
                    # ensure observation has correct shape for model prediction
                    if len(obs.shape) == 3:
                        obs_batch = np.expand_dims(obs, axis=0)
                    else:
                        obs_batch = obs
                    
                    # action prediction
                    if strategy.strategy_type == 'Agent':
                        action, _ = strategy.model.predict(obs_batch, deterministic=deterministic)
                    if strategy.strategy_type == 'Benchmark':
                        data = environment.timeseries_data[stock]
                        last_action = None if step_count == 0 else actions[step_count-1]
                        action = strategy.predict(data, last_action, step_count)

                    if isinstance(action, (list, np.ndarray)):
                        action = int(action[0] if len(action) > 0 else action)
                    else:
                        action = int(action)
                    actions.append(action)

                    # step; reward; next state
                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                    # extract portfolio value from environment info
                    # gym_anytrading tracks normalized portfolio value in total_profit
                    portfolio_factor = info.get('total_profit')
                    portfolio_factors.append(portfolio_factor)
                    episode_reward += reward
                    step_count += 1

                # compute performance metrics from portfolio values
                metrics = compute_performance_metrics(
                    portfolio_factors = portfolio_factors,
                    start_date = self.config.get('Start date'), 
                    end_date = self.config.get('End date'),
                    sig_figs = 4,
                )

                # action distribution diagnostics
                unique_actions, action_counts = np.unique(actions, return_counts = True)
                action_distribution = dict(zip(unique_actions.tolist(), action_counts.tolist()))
                 
                stock_metrics[stock] = {
                    'portfolio factors': portfolio_factors,
                    'episode reward': episode_reward,
                    'actions': actions,
                    'action distribution': action_distribution,
                    'num steps': step_count,
                    'cumulative return': metrics['cumulative_return'],
                    'annualized return': metrics['annualized_cumulative_return'],
                    'sharpe ratio': metrics['sharpe_ratio'],
                    'sortino ratio': metrics['sortino_ratio'],
                    'max drawdown': metrics['max_drawdown'],
                }
                
            self.logger.info("Evaluation complete with performance metrics")
            return stock_metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return False
    
    @log_function_call
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
    
    experiment_name = 'Mini'
    config = load_config(experiment_name)
    
    EXP = TemporalCrossValidation(experiment_name, config)
    Results = EXP.exe_experiment('inference')
    
    # if Results:
    #     saved_path = dev.save_experiment_results(results, format='pickle', compress=True)
    #     print(f"\nResults saved to: {saved_path}")
    #     # print("\nTo load results later, use:")
    #     # print(f"  from src.utils.results_io import load_results")
    #     # print(f"  results = load_results('{saved_path}')")
    