
class BaseStrategy():

    """
    Base class for RL agents and benchmark trading strategies

    :param strategy_type (str): 'Agent' or 'Benchmark'
    :param strategy_name (str): technique name
    """

    def __init__(
        self, 
        strategy_name: str, 
        strategy_type: str = ['Agent', 'Benchmark'],
    ) -> None:
        self.strategy_type = strategy_type
        self.strategy_name = strategy_name