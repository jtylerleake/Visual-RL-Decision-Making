
from common.modules import np, pd, random
from common.modules import TA
from src.models.base import BaseStrategy


class MACD (BaseStrategy):
    
    """
    Moving Average Convergence Divergence (MACD) trade strategy

    :param fast_period (int): short-term EMA period
    :param slow_period (int): long-term EMA period
    :param signal_period (int): signal line EMA period
    """
    
    def __init__(
        self, 
        fast_period = 12, 
        slow_period = 26, 
        signal_period = 9,
    ) -> None:

        super().__init__(
            strategy_name = 'MACD', 
            strategy_type = 'Benchmark'
        )

        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.macd_data = None
    
    def predict(
        self, 
        price_data: pd.DataFrame, 
        last_action: None,
        step_count: int
    ) -> int:
        """
        Compute MACD technical indicator for timeseries data and predict action 
        based on computed signals. Returns 0 = sell and 1 = buy. 
        """

        last_action = 0 if last_action == None else last_action

        min_periods = self.slow_period + self.signal_period

        if step_count < min_periods or step_count >= len(price_data):
            return last_action
        
        # calculate MACD for entire dataset once
        if self.macd_data is None:
            df = price_data.copy()
            self.macd_data = TA.MACD(
                df, 
                period_fast = self.fast_period, 
                period_slow = self.slow_period, 
                signal = self.signal_period
            )
        
        if step_count < 1 or step_count >= len(self.macd_data):
            return last_action
        
        # get MACD and signal values at time step
        macd = self.macd_data['MACD'].iloc[step_count]
        signal = self.macd_data['SIGNAL'].iloc[step_count]
        prev_macd = self.macd_data['MACD'].iloc[step_count - 1]
        prev_signal = self.macd_data['SIGNAL'].iloc[step_count - 1]
        
        # check for NaN values
        if pd.isna(macd) or pd.isna(signal) or pd.isna(prev_macd) or pd.isna(prev_signal):
            return last_action
        
        # Buy when MACD crosses above signal; sell when crosses below
        if prev_macd <= prev_signal and macd > signal:
            return 1
        elif prev_macd >= prev_signal and macd < signal:
            return 0
        else:
            return last_action


class SignR(BaseStrategy):
    
    """
    Sign of Returns, Sign(R), trade strategy. Predicts action based on the 
    sign of price returns: buy on positive returns, sell on negative returns.
    """
    
    def __init__(self) -> None:
        super().__init__(
            strategy_name='Sign(R)',
            strategy_type='Benchmark'
        )
    
    def predict(
        self,
        price_data: pd.DataFrame,
        last_action: None,
        step_count: int
    ) -> int:
        """
        Predict action based on sign of returns. Returns 0 = sell, 1 = buy
        """
        
        last_action = 0 if last_action is None else last_action
        
        if step_count < 1 or step_count >= len(price_data):
            return last_action
        
        # calculate return: price[t] - price[t-1]
        current_price = price_data['Close'].iloc[step_count]
        prev_price = price_data['Close'].iloc[step_count - 1]
        
        if pd.isna(current_price) or pd.isna(prev_price):
            return last_action
        
        # sign of return: positive = buy, negative = sell, zero = hold
        return_diff = current_price - prev_price
        if return_diff > 0:
            return 1
        elif return_diff < 0:
            return 0
        else:
            return last_action


class BuyAndHold(BaseStrategy):
    
    """
    Buy and Hold strategy: buy at first step, then hold thereafter
    """
    
    def __init__(self) -> None:
        super().__init__(
            strategy_name='Buy and Hold',
            strategy_type='Benchmark'
        )
    
    def predict(
        self,
        price_data: pd.DataFrame,
        last_action: None,
        step_count: int
    ) -> int:
        """Buy at step 0, then hold. Returns 0=sell, 1=buy"""
        return 1


class Random(BaseStrategy):
    
    """
    Random strategy: randomly choose buy or sell at each step
    """
    
    def __init__(self, seed = None) -> None:
        super().__init__(
            strategy_name='Random',
            strategy_type='Benchmark'
        )
        if seed is not None:
            random.seed(seed)
    
    def predict(
        self,
        price_data: pd.DataFrame,
        last_action: None,
        step_count: int
    ) -> int:
        """Randomly choose buy (1) or sell (0). Returns 0=sell, 1=buy"""
        return random.choice([0, 1])
    