
from common.modules import np, pd, plt, sns, List, Dict, dt, json, os
from matplotlib.patches import Rectangle



def plot_normalized_test_lines(
    portfolio_factors_fpath,
    plt_save_directory,
    plot_by: str = 'Window',
    strategy: str = 'Visual agent',
    figsize: tuple = (14, 8),
    ) -> None:
    """Plot normalized profit factors as line charts for each temporal window"""
    
    try:

        # load the json file
        portfolio_factors_fpath = portfolio_factors_fpath.strip()        
        if not os.path.exists(portfolio_factors_fpath):
            raise FileNotFoundError(f"Portfolio factors file not found: {portfolio_factors_fpath}")
        with open(portfolio_factors_fpath, 'r', encoding='utf-8') as f:
            portfolio_data = json.load(f)
        
        # extract data for the strategy
        if strategy not in portfolio_data:
            raise ValueError(f"{strategy} not found in portfolio factors file")
        strategy_data = portfolio_data[strategy]
        
        # organize data by either fold or window
        grouped_data = {}
        
        for fold_id_str, window_results in strategy_data.items():
            for window_id_str, stock_results in window_results.items():
                window_id = int(window_id_str)
                for stock, subset in stock_results.items():
                    factors = subset['Factors']
                    if isinstance(factors, list) and len(factors) > 0:
                        factors_array = np.array(factors)
                        if window_id not in grouped_data: grouped_data[window_id] = []
                        grouped_data[window_id].append(factors_array)
        
        if not grouped_data:
            raise ValueError("No portfolio factor data found in file")
        
        # plot all portfolio factors as individual lines
        for window_idx, factors in grouped_data.items():
            
            # count total number of lines for color palette
            total_lines = sum(len(factor_list) for factor_list in grouped_data.values())
            
            # configure the plot style and create the plot
            sns.set_style("whitegrid")
            sns.set_palette("Paired", n_colors = total_lines)
            fig, ax = plt.subplots(figsize = figsize)
            
            time_steps = np.arange(len(factors[0])) # create time steps (x-axis)
            for factor_arr in factors: 
                ax.plot(
                    time_steps,
                    factor_arr,
                    linewidth = 2,
                    alpha = 0.7
                )
        
            # customize the plot
            ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('Profit Factor', fontsize=12, fontweight='bold')
            
            # title attributes
            title_suffix = f'Window {window_idx} ({total_lines} test stocks)'
            
            ax.set_title(
                f'{strategy.title()} Cumulative Performance on Test \n'
                + title_suffix,
                fontsize = 15,
                fontweight = 'bold',
                pad = 25,
                loc = 'center'
            )
            
            # add gridlines and horizontal line at 1.0 for reference
            ax.axhline(y = 1.0, color = 'black', linestyle = '--', linewidth = 1, alpha = 0.5, label = 'Baseline (1.0)')
            ax.grid(True, alpha = 0.3, linestyle = '--')
            ax.spines['bottom'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_color('black')
            ax.spines['left'].set_linewidth(1)
            plt.tight_layout()
            
            # save 
            save_dir = os.path.abspath(plt_save_directory)
            if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
            extension = f"window_{window_idx}_normalized_plot"
            plt_save_path = os.path.normpath(os.path.join(save_dir, extension))
            plt.savefig(plt_save_path, dpi = 300, bbox_inches = 'tight')
        
    except Exception as e:
        raise ValueError("Error generating plot: {e}")



def convert_to_latex_table(
    metric: str = 'cumulative return',
    aggregated_stats_fpath: str = None,
    save_directory: str = None,
    aggregation_level: str = 'fold',
    sig_figs: int = 4,
    n_folds: int = 5,
) -> str:
    """Convert aggregate statistics from cross-validation to LaTeX table"""
    
    try:
                
        if not os.path.exists(aggregated_stats_fpath):
            raise FileNotFoundError(f"Results file not found: {aggregated_stats_fpath}")
        with open(aggregated_stats_fpath, 'r', encoding='utf-8') as f:
            file_results = json.load(f)
            
        # extract the appropriate section from the file     
        section_key = 'Fold-wise' if aggregation_level.lower() == 'fold' else 'Window-wise'
        results = file_results.get(section_key, {})
        if not results: raise ValueError(f"No '{section_key}' section found in results file")
        
        # helper function to format a value
        def format_val(value, pct = False, default = 0.0):
            if value is None or np.isnan(value): return default
            if pct: 
                rounded = round(float(value) * 100, sig_figs)
                return f"{rounded:.2f}\%"
            else: 
                rounded = round(float(value), sig_figs)
                return f"{rounded:.3f}"
        
        # helper function to get metric value from aggregated results
        def get_metric(results, metric, strategy, stat):
            try:
                strategy_data = results.get(strategy, {})
                metric = strategy_data.get(metric, {})
                stat = metric.get(stat, "x")
                return stat
            except:
                return 0.0

        latex_lines = []
        strategy_order = ['Long', 'Random', 'MACD', 'Numeric agent', 'Visual agent']
        
        for strategy in strategy_order:
            
            F1 = format_val(get_metric(results, metric, strategy, '1'), pct = True)
            F2 = format_val(get_metric(results, metric, strategy, '2'), pct = True)
            F3 = format_val(get_metric(results, metric, strategy, '3'), pct = True)
            # F4 = format_val(get_metric(results, metric, strategy, '4'), pct = True)
            # F5 = format_val(get_metric(results, metric, strategy, '5'), pct = True)
            
            AVG = format_val(get_metric(results, metric, strategy, 'mean'), pct = True)
            STD = format_val(get_metric(results, metric, strategy, 'std'))
            MIN = format_val(get_metric(results, metric, strategy, 'min'), pct = True)
            MAX = format_val(get_metric(results, metric, strategy, 'max'), pct = True)
            
            # sharpe = format_val(get_metric(results, strategy, 'sharpe ratio', 'mean'))
            # sortino = format_val(get_metric(results, strategy, 'sortino ratio', 'mean'))
            # mdd = format_val(get_metric(results, strategy, 'max drawdown', 'mean'), pct = True)

            # format the row with uniform and minimal whitespace around &
            row = (
                f"{strategy:<20} & "
                f"{F1} & "
                f"{F2} & "
                f"{F3} & "
                # f"{F4} & "
                # f"{F5} & "
                f"{AVG} & "
                f"{STD} & "
                f"{MIN} & "
                f"{MAX} & \\\\"
            )
            
            latex_lines.append(row)
            
        latex_table = "\n".join(latex_lines)
        
        # save 
        save_dir = os.path.abspath(save_directory)
        if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        fname = section_key.split("-")[0].lower()
        extension = f"{metric}_{fname}_wise_results"
        table_save_path = os.path.normpath(os.path.join(save_dir, extension))
        with open(table_save_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)

    except Exception as e:
        raise ValueError(f"Error generating table: {e}")



def plot_best_model_overlay(
    portfolio_factors_fpath: str,
    plt_save_directory: str,
    figsize: tuple = (16, 10),
):

    try:

        # fetch the data files
        if not os.path.exists(portfolio_factors_fpath):
            raise FileNotFoundError(f"Results file not found: {portfolio_factors_fpath}")
        with open(portfolio_factors_fpath, 'r', encoding='utf-8') as f:
            portfolio_factors = json.load(f)['Visual agent']
        
        # search through all test results to find the best
        best_performance = -np.inf
        best_portfolio_factors = None
        best_info = None
        actions = []
        
        for fold_id, window_results in portfolio_factors.items():
            for window_id, stock_data in window_results.items():
                for stock, subset in stock_data.items():
                    terminal_factor = subset['Factors'][-1]
                    if terminal_factor > best_performance: 
                        best_performance = terminal_factor
                        best_portfolio_factors = subset['Factors']
                        best_info = (fold_id, window_id, stock)
                        actions = subset['Actions']
                        
        if not best_info:
            raise ValueError(f"No valid portfolio factors found in test results")  
        
        # helper function to trim timeseries
        def trim_dates(
            data: pd.DataFrame, 
            start_date_dt: pd.Timestamp, 
            end_date_dt: pd.Timestamp
        ) -> pd.DataFrame:
            """Trim data to experiment date range with proper timezone handling"""
            
            if data.empty:return data
            
            # handle timezone consistently
            timezone = data.index[0].tz if hasattr(data.index[0], 'tz') else None
            
            if timezone is not None:
                start_dt = start_date_dt.tz_localize(timezone) if start_date_dt.tz \
                    is None else start_date_dt
                end_dt = end_date_dt.tz_localize(timezone) if end_date_dt.tz \
                    is None else end_date_dt
            else:
                start_dt = start_date_dt.tz_localize('UTC') if start_date_dt.tz is \
                    None else start_date_dt
                end_dt = end_date_dt.tz_localize('UTC') if end_date_dt.tz is \
                    None else end_date_dt
                data.index = data.index.tz_localize('UTC')

            return data.loc[(data.index >= start_dt) & (data.index <= end_dt)]

        # fetch and trim data
        data_path = f"dataset/{stock}_data.csv"
        timeseries = pd.read_csv(data_path)
        plot_data = trim_dates(timeseries, x, y)
        
        if len(actions) > len(plot_data):
            actions = actions[:len(plot_data)]
        
        # set up the plot style
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        
        # create candlestick chart
        dates = plot_data.index 
        
        # plot candlesticks
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            
            open_price = row['Open']
            high_price = row['High']
            low_price = row['Low']
            close_price = row['Close']
            
            color = 'green' if close_price >= open_price else 'red'
            
            # draw the wick (high-low line)
            ax.plot([i, i], [low_price, high_price], color='black', linewidth=0.5, alpha=0.7)
            
            # draw the body (open-close rectangle)
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = Rectangle(
                (i - 0.3, body_bottom),
                0.6,
                body_height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            ax.add_patch(rect)
        
        # overlay buy/sell action lines
        buy_indices = []
        sell_indices = []
        
        for i, action in enumerate(actions):
            if action == 1: buy_indices.append(i)
            elif action == 0: sell_indices.append(i)
        
        # plot buy lines (green, vertical)
        if buy_indices:
            for idx in buy_indices:
                if idx < len(plot_data):
                    ax.axvline(
                        x=idx,
                        color='green',
                        linestyle='--',
                        linewidth=2.5,
                        alpha=0.8,
                        label='Buy' if idx == buy_indices[0] else '',
                        zorder=10
                    )
        
        # plot sell lines (red, vertical)
        if sell_indices:
            for idx in sell_indices:
                if idx < len(plot_data):
                    ax.axvline(
                        x=idx,
                        color='red',
                        linestyle='--',
                        linewidth=2.5,
                        alpha=0.8,
                        label='Sell' if idx == sell_indices[0] else '',
                        zorder=10
                    )
        
        # customize the plot
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax.set_title(
            f"""Best Model Performance: {stock} \n(Fold {best_info[fold_id]}, 
            Window {best_info[window_id]})""", 
            fontsize=15,
            fontweight='bold',
            pad=20
        )
        
        # set x-axis labels
        if isinstance(plot_data.index, pd.DatetimeIndex):
            ax.set_xticks(range(len(plot_data)))
            ax.set_xticklabels(
                [d.strftime('%Y-%m-%d') for d in plot_data.index], 
                rotation=45, 
                ha='right'
            )
        else:
            ax.set_xticks(range(0, len(plot_data), max(1, len(plot_data) // 10)))
        
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.show()
        
        # save 
        save_dir = os.path.abspath(plt_save_directory)
        if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
        extension = f"best_performance_plot"
        plt_save_path = os.path.normpath(os.path.join(save_dir, extension))
        plt.savefig(plt_save_path, dpi = 300, bbox_inches = 'tight')
        
    except Exception as e:
        raise ValueError(f"Error generating plot: {e}")






def plot_temporal_stability_lines(
    aggregated_results: Dict,
    strategy_name: str = 'test',
    metric_name: str = 'cumulative_return',
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """Plot temporal stability lines showing average metric values across windows for each fold"""

    try:
        # Extract fold_results from aggregated results
        fold_results = aggregated_results.get('fold_results', {})
        if not fold_results:
            raise ValueError("No fold_results found in aggregated_results")
        
        # Strategy and metric name mappings (reverse of what's in aggregate function)
        strategy_mapping_reverse = {
            'macd': 'macd results',
            'signr': 'signr results',
            'buyandhold': 'buyandhold results',
            'random': 'random results'
        }
        
        metric_mapping_reverse = {
            'episode_reward': 'episode reward',
            'cumulative_return': 'cumulative return',
            'annualized_return': 'annualized return',
            'sharpe_ratio': 'sharpe ratio',
            'sortino_ratio': 'sortino ratio',
            'max_drawdown': 'max drawdown'
        }
        
        strategy_key = strategy_mapping_reverse.get(strategy_name, f'{strategy_name} results')
        metric_key = metric_mapping_reverse.get(metric_name, metric_name)
        
        # Extract window-level averages per fold
        fold_window_data = {}  # {fold_id: {window_id: [values]}}
        
        for fold_id, window_results in fold_results.items():
            fold_window_data[fold_id] = {}
            
            for window_id, fold_window_results in window_results.items():
                # Windows are 0-indexed in fold_results, convert to 1-indexed
                window_key = window_id + 1 if isinstance(window_id, int) else window_id
                
                # Get strategy results for this window
                strategy_result = fold_window_results.get(strategy_key, {})
                if not isinstance(strategy_result, dict):
                    continue
                
                # Collect metric values for all stocks in this window
                values = []
                for stock, stock_metrics in strategy_result.items():
                    if isinstance(stock_metrics, dict) and metric_key in stock_metrics:
                        value = stock_metrics[metric_key]
                        if value is not None and not np.isnan(value):
                            values.append(float(value))
                
                # Store values for this window in this fold
                if values:
                    fold_window_data[fold_id][window_key] = values
        
        # Prepare data for plotting
        plot_data = []
        for fold_id, window_data in fold_window_data.items():
            for window_id in sorted(window_data.keys()):
                values = window_data[window_id]
                if values:
                    avg_value = np.mean(values)
                    plot_data.append({
                        'Fold': f'Fold {fold_id}',
                        'Window': window_id,
                        'Average Metric': avg_value
                    })
        
        if not plot_data:
            raise ValueError(f"No data found for strategy '{strategy_name}' and metric '{metric_name}'")
        
        # Convert to DataFrame for seaborn
        df = pd.DataFrame(plot_data)
        
        # Set up the plot style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create the figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot lines using seaborn lineplot
        sns.lineplot(
            data=df,
            x='Window',
            y='Average Metric',
            hue='Fold',
            marker='o',
            linewidth=2.5,
            markersize=8,
            ax=ax,
            legend='full'
        )
        
        # Customize the plot
        ax.set_xlabel('Window Index', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Average {metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Temporal Stability: {strategy_name.upper()} - {metric_name.replace("_", " ").title()}',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # Set x-axis to show integer window indices
        unique_windows = sorted(df['Window'].unique())
        ax.set_xticks(unique_windows)
        ax.set_xticklabels([int(w) for w in unique_windows])
        
        # Customize legend
        ax.legend(
            title='Fold',
            loc='best',
            frameon=True,
            fancybox=True,
            shadow=True,
            title_fontsize=11,
            fontsize=10
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        
        return fig
        
    except Exception as e:
        raise
