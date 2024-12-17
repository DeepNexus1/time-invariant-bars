import numpy as np
import pandas as pd
from numba import jit
import datetime

start_time = pd.Timestamp.now()


path = '.csv' # path to your data file here, ideally tick data from which you can build bars

df = pd.read_csv(path, parse_dates=['DateTime'], usecols = ['DateTime', 'Bid', 'Ask', 'Volume'])
# skiprows=range(1, 250000000) : optional parameter 
# depending on the size of your data file and memory constraints, you may need to add parameter skiprows when creating your dataframe

df = df.rename(columns={'DateTime': 'Date'})

first_date = df.iloc[0]['Date']
reset_date = first_date.replace(hour=16,
                                minute=59,
                                second=59) + \
pd.Timedelta("{} days".format(6 - first_date.dayofweek))
reset_date = reset_date.timestamp() * 1e3
bars = []
num_lines = len(df)
dates = df['Date'].values.astype(np.float64) // 10**6
gap = np.float64(pd.Timedelta("7 days").total_seconds() * 10**3)
df['Price'] = ((df.Ask + df.Bid) / 2)  # price is bid/ask midpoint
prices = df['Price'].values
volumes = df['Volume'].values
df['Spread'] = (((df['Ask'] - df['Bid']) / df['Price']) * 100)  # normalized spread
spreads = df['Spread'].values

# use jit decorators via numba to speed up the processing
# if you encounter a forced exit from the script without a csv being created at the end, try commenting out the jit decorators and re-running the program

@jit
def create_range_bar(range_prices, range_volumes, range_spreads, last_date, reset, complete):
        return   np.array([range_prices[0], range_prices[-1], max(range_prices), min(range_prices), 
                           len(range_prices), np.sum(range_volumes), 
                           range_spreads[0], range_spreads[-1], max(range_spreads), min(range_spreads),
                           last_date, reset, complete])
    
    
@jit(forceobj=True)   
def get_bars_from_df(prices, dates, volumes, spreads, reset_date, gap, return_val):
    bars = []
    start_val = None
    start_idx = None

    for i in range(len(prices)):

        val = prices[i]
        date = dates[i]

        if not start_val:
            start_val = val
            start_idx = i

        elif date > reset_date:
            end_idx = i
            range_prices = prices[start_idx:end_idx+1]
            range_volumes = volumes[start_idx:end_idx+1]
            range_spreads = spreads[start_idx:end_idx+1]
            last_date = dates[end_idx]
            bars.append(create_range_bar(range_prices, range_volumes, range_spreads, last_date, 1., 1.))

            reset_date += gap
            start_idx = None
            start_val = None

        elif abs(np.log(val / start_val)) > return_val:
            end_idx = i
            range_prices = prices[start_idx:end_idx+1]
            range_volumes = volumes[start_idx:end_idx+1]
            range_spreads = spreads[start_idx:end_idx+1]
            last_date = dates[end_idx]
            bars.append(create_range_bar(range_prices, range_volumes, range_spreads, last_date, 0, 1))

            start_idx = None
            start_val = None

    if end_idx != i:
        end_idx = i
        range_prices = prices[start_idx:end_idx+1]
        range_volumes = volumes[start_idx:end_idx+1]
        range_spreads = spreads[start_idx:end_idx+1]
        last_date = dates[end_idx]
        bars.append(create_range_bar(range_prices, range_volumes, range_spreads, last_date, 0, 1))

    bars = np.stack(bars)
    return bars

bars = get_bars_from_df(prices, dates, volumes, spreads, reset_date, gap, 0.01)
    
bars_df = pd.DataFrame(data=bars,
                       columns=['open', 'close', 'high', 'low', 
                                'ticks_number', 'sum_volume',
                                'open_spread', 'close_spread', 'max_spread', 'min_spread',
                                'date', 'reset_date', 'complete'])
bars_df['date'] = pd.to_datetime(bars_df['date'] * 10**6)

# reorder columns...
bars_df = bars_df[['date', 'open', 'high', 'low', 'close',
                                'ticks_number', 'sum_volume',
                                'open_spread', 'close_spread', 'max_spread', 'min_spread',
                                 'reset_date', 'complete']]

bars_df.to_csv('.csv', index=False) # save your new data file here; save as .csv or .h5, whatever your preference

end_time = pd.Timestamp.now()

print("Completed in: ", end_time - start_time)


