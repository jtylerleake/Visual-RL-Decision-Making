#!/usr/bin/env python3
"""
Script to analyze dataset-cache files and identify date ranges for each ticker.
Prints a table with ticker, start date, and end date.
"""

my_tickers = [

    'ARE', 'APA', 'AVY', 'BALL', 'BAX', 'BBY', 'CHRW', 'CPT', 'DRI', 'DECK', 'ESS', 'FFIV', 'FRT', 'HAS', 'HPQ', 'INCY', 'IFF', 'IP', 'JBHT', 'LH', 'MAA', 'NTAP', 'NDSN', 'NTRS', 'NVR', 'PNW', 'PPG', 'PTC', 'PHM', 'RVTY', 'SWK', 'TROW', 'TXT', 'COO', 'TRMB', 'TYL', 'UHS', 'WAT', 'WST', 'CPB', 'IPG', 'TAP', 'MGM', 'BXP', 'SNA', 'PSKY', 'KIM', 'GPC', 'HUBB', 'VRSN', 'BIIB', 'BRO', 'FE', 'PPL', 'STE', 'ADM', 'CSGP', 'EL', 'NUE', 'RJF', 'STT', 'A', 'DHI', 'EBAY', 'EQT', 'HSY', 'RMD', 'ROK', 'VMC', 'VTR', 'D', 'GWW', 'OKE', 'PSA', 'URI', 'AFL', 'FDX', 'SPG', 'SRE', 'TFC', 'TSCO', 'FITB', 'MCHP', 'ROL', 'MTB', 'CTSH', 'WAB', 'CCL', 'ACGL', 'YUM', 'ROST', 'ROP', 'EA', 'PCAR', 'XEL', 'CMI', 'MSI', 'ADSK', 'ALL', 'NSC', 'AZO', 'CI', 'EMR', 'RCL', 'AEP', 'TRV', 'WMB', 'SNPS', 'AMT', 'CDNS', 'GD', 'MMC', 'NEM', 'ORLY', 'SO', 'TT', 'DHR', 'GILD', 'HON', 'PLD', 'SYK', 'TXN', 'ABT', 'AMAT', 'AMGN', 'CAT', 'DIS', 'GS', 'T', 'TMO', 'AAPL', 'BRK.B', 'MU', 'ORCL', 'WFC', 'WMT', 'MAR', 'REGN', 'MCK', 'PH', 'KLAC', 'ETN', 'SPGI', 'UNP', 'MS', 'RTX', 'HD', 'PG', 'MSFT', 'JNJ'

]

For_Dev = [
    'CPB', 'IPG', 'TAP', 'MGM', 'BXP', 'SNA', 'PSKY', 'KIM', 'GPC', 'HUBB', 'VRSN', 'TSCO', 'FITB', 'MCHP', 'ROL', 'MTB', 'CTSH', 'WAB', 'CCL', 'ACGL', 'YUM', 'ROST', 'ROP', 'EA', 'PCAR', 'XEL', 'CMI', 'MSI', 'ADSK', 'ALL', 'NSC', 'GILD', 'PLD', 'TXN', 'ABT', 'AMAT', 'AMGN', 'CAT', 'DIS', 'GS', 'T', 'TMO', 'AAPL', 'BRK.B', 'MU', 'ORCL', 'WFC', 'WMT', 'MAR', 'REGN', 'MCK', 'PH', 'KLAC', 'ETN', 'SPGI', 'UNP', 'MS', 'RTX', 'HD', 'PG', 'MSFT', 'JNJ'
]

import os
import pandas as pd
from pathlib import Path


def extract_ticker_from_filename(filename):
    """Extract ticker symbol from filename (e.g., 'A_data.csv' -> 'A')."""
    return filename.replace('_data.csv', '')


def get_date_range(csv_path):
    """Read CSV file and return min and max dates."""
    try:
        df = pd.read_csv(csv_path)
        if 'date' not in df.columns:
            return None, None
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        return min_date, max_date
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, None


def main():
    """Main function to analyze all data files."""
    dataset_cache_dir = Path('dataset-cache')
    
    if not dataset_cache_dir.exists():
        print(f"Error: {dataset_cache_dir} directory not found!")
        return
    
    # Get all CSV files
    csv_files = sorted(dataset_cache_dir.glob('*_data.csv'))
    
    if not csv_files:
        print(f"No data files found in {dataset_cache_dir}")
        return
    
    # Process each file
    results = []
    for csv_file in csv_files:
        ticker = extract_ticker_from_filename(csv_file.name)
        min_date, max_date = get_date_range(csv_file)
        
        if min_date is not None and max_date is not None:
            results.append({
                'Ticker': ticker,
                'Start Date': min_date.strftime('%Y-%m-%d'),
                'End Date': max_date.strftime('%Y-%m-%d')
            })
        else:
            results.append({
                'Ticker': ticker,
                'Start Date': 'ERROR',
                'End Date': 'ERROR'
            })
    
    # Convert to DataFrame for nice formatting
    df_results = pd.DataFrame(results)
    
    # Print table
    print("\nDataset Date Range Analysis")
    print("=" * 80)
    print(df_results.to_string(index=False))
    print("=" * 80)
    print(f"\nTotal tickers analyzed: {len(results)}")
    
    # Print summary statistics
    if results and results[0]['Start Date'] != 'ERROR':
        all_start_dates = [pd.to_datetime(r['Start Date']) for r in results if r['Start Date'] != 'ERROR']
        all_end_dates = [pd.to_datetime(r['End Date']) for r in results if r['End Date'] != 'ERROR']
        
        if all_start_dates and all_end_dates:
            overall_min = min(all_start_dates)
            overall_max = max(all_end_dates)
            print(f"\nOverall date range: {overall_min.strftime('%Y-%m-%d')} to {overall_max.strftime('%Y-%m-%d')}")


if __name__ == '__main__':
    main()
