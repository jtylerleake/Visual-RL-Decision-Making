
from common.modules import EvalCallback, DQN, A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from src.models.base import BaseStrategy
import os


class NumericAgent(BaseStrategy):
    
    """
    Overhead class for instantiating and training the numeric A2C reinforcement 
    learning model. Requires pre-built training env and an experiment config

    :param training_environment (): environment for model training
    :param config (Dict): experiment configuration dictionary
    """
        
    def __init__(
        self, 
        environment, 
        config
    ) -> None: 

        super().__init__(
            strategy_type = 'Agent', 
            strategy_name = 'Numeric A2C Agent'
        )

        self.config = config
        self.env = environment.numeric_vec_environment
        self.model = self.setup_model()

    def setup_model(
        self, 
    ) -> None:

        model = PPO
        policy = 'MlpPolicy'
        
        agent = model(
            policy,
            env = self.env, 
            learning_rate = self.config.get('Learning rate'),
            batch_size = self.config.get('Batch size'),
            n_steps = self.config.get('Rollout steps'),
            device = "auto",
            verbose = 1
        )
        
        return agent
    
    def train(
        self, 
        checkpoint_save_path: str = None,
        checkpoint_freq: int = None
    ) -> bool: 
        """Train the model with optional checkpoint saving"""
        
        epochs = self.config.get('Training epochs')
        callbacks = []
        
        if checkpoint_save_path and checkpoint_freq:
            os.makedirs(checkpoint_save_path, exist_ok=True)
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_save_path,
                name_prefix='checkpoint'
            )
            callbacks.append(checkpoint_callback)
        
        # combine callbacks (use CallbackList if multiple, otherwise single callback or None)
        if len(callbacks) == 0:
            callback = None
        elif len(callbacks) == 1:
            callback = callbacks[0]
        else:
            callback = CallbackList(callbacks)
        
        self.model.learn(total_timesteps = epochs, callback = callback)
        return True
   