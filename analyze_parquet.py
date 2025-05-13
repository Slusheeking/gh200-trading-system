import pandas as pd
import glob
import numpy as np
from datetime import timedelta

def analyze_parquet_files():
    # Get all day and minute parquet files
    day_files = glob.glob('data/cache/bars/*/day.parquet')
    minute_files = glob.glob('data/cache/bars/*/minute.parquet')
    
    print('Number of stocks with day data: {}'.format(len(day_files)))
    print('Number of stocks with minute data: {}'.format(len(minute_files)))
    
    # Analyze day files
    print('\nDay data analysis:')
    day_row_counts = []
    day_start_dates = []
    day_end_dates = []
    
    for file in day_files:
        stock = file.split('/')[3]
        try:
            df = pd.read_parquet(file)
            day_row_counts.append(df.shape[0])
            
            # Only add valid timestamps
            if not df.empty and pd.api.types.is_datetime64_any_dtype(df.index):
                day_start_dates.append(df.index.min())
                day_end_dates.append(df.index.max())
            
            # Print details for first 5 stocks
            if len(day_row_counts) <= 5:
                print('{}: {} rows, date range: {} to {}'.format(stock, df.shape[0], df.index.min() if not df.empty else "N/A", df.index.max() if not df.empty else "N/A"))
        except Exception as e:
            print(f'{stock}: Error - {str(e)}')
    
    # Analyze minute files
    print('\nMinute data analysis:')
    minute_row_counts = []
    minute_start_dates = []
    minute_end_dates = []
    
    for file in minute_files:
        stock = file.split('/')[3]
        try:
            df = pd.read_parquet(file)
            minute_row_counts.append(df.shape[0])
            
            # Only add valid timestamps
            if not df.empty and pd.api.types.is_datetime64_any_dtype(df.index):
                minute_start_dates.append(df.index.min())
                minute_end_dates.append(df.index.max())
            
            # Print details for first 5 stocks
            if len(minute_row_counts) <= 5:
                print('{}: {} rows, date range: {} to {}'.format(stock, df.shape[0], df.index.min() if not df.empty else "N/A", df.index.max() if not df.empty else "N/A"))
        except Exception as e:
            print(f'{stock}: Error - {str(e)}')
    
    # Statistical analysis of day data
    print('\nDay data statistics:')
    if day_row_counts:
        print(f'Row count - Min: {min(day_row_counts)}, Max: {max(day_row_counts)}, Mean: {np.mean(day_row_counts):.2f}, Median: {np.median(day_row_counts)}')
        if day_start_dates:
            print(f'Earliest start date: {min(day_start_dates)}')
            print(f'Latest end date: {max(day_end_dates)}')
        else:
            print("No valid date ranges found in day data")
    
    # Statistical analysis of minute data
    print('\nMinute data statistics:')
    if minute_row_counts:
        print(f'Row count - Min: {min(minute_row_counts)}, Max: {max(minute_row_counts)}, Mean: {np.mean(minute_row_counts):.2f}, Median: {np.median(minute_row_counts)}')
        if minute_start_dates:
            print(f'Earliest start date: {min(minute_start_dates)}')
            print(f'Latest end date: {max(minute_end_dates)}')
        else:
            print("No valid date ranges found in minute data")
    
    # Check for missing values in a sample file
    print('\nChecking for missing values in STOCK1:')
    try:
        df_day = pd.read_parquet('data/cache/bars/STOCK1/day.parquet')
        print('Missing values in day.parquet:')
        print(df_day.isna().sum())
        
        df_minute = pd.read_parquet('data/cache/bars/STOCK1/minute.parquet')
        print('\nMissing values in minute.parquet:')
        print(df_minute.isna().sum())
    except Exception as e:
        print(f'Error checking missing values: {str(e)}')
    
    # Check data types and value ranges
    print('\nData types and value ranges for STOCK1:')
    try:
        print('Day data:')
        print(f'Data types: {df_day.dtypes}')
        print('Value ranges:')
        for col in df_day.columns:
            print(f'  {col}: {df_day[col].min()} to {df_day[col].max()}')
        
        print('\nMinute data:')
        print(f'Data types: {df_minute.dtypes}')
        print('Value ranges (sample):')
        for col in df_minute.columns:
            print(f'  {col}: {df_minute[col].min()} to {df_minute[col].max()}')
    except Exception as e:
        print(f'Error checking data types and ranges: {str(e)}')
    
    # Analyze time intervals in minute data
    print('\nAnalyzing time intervals in STOCK1 minute data:')
    try:
        # Sort by timestamp to ensure correct interval calculation
        df_minute_sorted = df_minute.sort_index()
        
        # Calculate time differences between consecutive rows
        time_diffs = df_minute_sorted.index.to_series().diff().dropna()
        
        # Convert to minutes for easier interpretation
        time_diffs_minutes = time_diffs.dt.total_seconds() / 60
        
        print(f'Time interval statistics (in minutes):')
        print(f'  Min: {time_diffs_minutes.min()}')
        print(f'  Max: {time_diffs_minutes.max()}')
        print(f'  Mean: {time_diffs_minutes.mean():.2f}')
        print(f'  Median: {time_diffs_minutes.median()}')
        
        # Count occurrences of each interval
        interval_counts = time_diffs_minutes.value_counts().sort_index()
        print(f'Most common intervals (minutes: count):')
        for interval, count in interval_counts.head(5).items():
            print(f'  {interval}: {count}')
        
        # Check for gaps in data
        large_gaps = time_diffs[time_diffs > timedelta(minutes=5)]
        if not large_gaps.empty:
            print(f'Found {len(large_gaps)} gaps larger than 5 minutes')
            print('Sample of gaps:')
            for timestamp, gap in large_gaps.head(3).items():
                print(f'  Gap of {gap} at {timestamp}')
    except Exception as e:
        print(f'Error analyzing time intervals: {str(e)}')
    
    # Compare data consistency between stocks
    print('\nComparing data between stocks:')
    try:
        # Compare a few stocks for the same date range
        stocks_to_compare = ['STOCK1', 'STOCK2', 'STOCK3']
        date_to_check = '2025-03-10'  # Choose a date that should be present in most stocks
        
        print(f'Data for {date_to_check}:')
        for stock in stocks_to_compare:
            try:
                df = pd.read_parquet(f'data/cache/bars/{stock}/day.parquet')
                date_matches = df.index.strftime('%Y-%m-%d') == date_to_check
                if any(date_matches):
                    day_data = df[date_matches]
                    print(f'  {stock}: Close price: {day_data["close"].values[0]}, Volume: {day_data["volume"].values[0]}')
                else:
                    print(f'  {stock}: No data for {date_to_check}')
            except Exception as e:
                print(f'  {stock}: Error - {str(e)}')
    except Exception as e:
        print(f'Error comparing stocks: {str(e)}')

if __name__ == "__main__":
    analyze_parquet_files()
