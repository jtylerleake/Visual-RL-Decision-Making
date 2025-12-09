
if __name__ == "__main__":

    from src.utils.figures import *

    # experiments=("Large-Cap", "Medium-Cap", "Small-Cap")

    experiments=["Mini"]
    
    for experiment in experiments: 

        aggregated_statistics_path = f"""C:/Users/Jtyler/Downloads/visual-reinforcement-fin-decision-making/experiments/{experiment}/aggregated_statistics"""
        portfolio_factors_path = f"""C:/Users/Jtyler/Downloads/visual-reinforcement-fin-decision-making/experiments/{experiment}/portfolio_factors"""
        figures_save_directory = f"""C:/Users/Jtyler/Downloads/visual-reinforcement-fin-decision-making/experiments/{experiment}/figures"""
        tables_save_directory = f"""C:/Users/Jtyler/Downloads/visual-reinforcement-fin-decision-making/experiments/{experiment}/tables"""

        # table conversion

        convert_to_latex_table(
            metric = 'cumulative return',
            aggregated_stats_fpath = aggregated_statistics_path,
            save_directory = tables_save_directory,
            aggregation_level = 'fold',
            sig_figs = 3
        )

        convert_to_latex_table(
            metric = 'cumulative return',
            aggregated_stats_fpath = aggregated_statistics_path,
            save_directory = tables_save_directory,
            aggregation_level = 'window',
            sig_figs = 3
        )

        # plot creation

        '''
        plot_normalized_test_lines(
            portfolio_factors_fpath = portfolio_factors_path, 
            plt_save_directory = figures_save_directory
        )

        plot_best_model_overlay(
            portfolio_factors_fpath = portfolio_factors_fpath,
            save_path = figures_save_directory,
            figsize = (16, 10),
        )    
        '''

