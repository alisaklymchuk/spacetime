# General Setup
import os
import sys

from os.path import join

# Local imports from spacetime
project_dir = './spacetime'
sys.path.insert(0, os.path.abspath(project_dir)) 
# The data science trinity, we might not use all of them
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from utils.logging import print_config
from pprint import pprint
from datetime import timedelta

# Hacky args via an OmegaConf config
args = """
seed: 42
"""
args = OmegaConf.create(args)


import yfinance as yf

yf_ticker_list = [
    yf.Ticker('XRP-GBP'),
    yf.Ticker('XMR-GBP'),
    yf.Ticker('LTC-GBP'),
    yf.Ticker('XLM-GBP')
]

start_date = '2024-01-01'   # Format: 'YYYY-MM-DD'
end_date = '2025-10-20'     # Format: 'YYYY-MM-DD'

# Option 2: Specify end date and lookback days (comment out Option 1 if using this)
# end_date = '2024-12-31'
# days_lookback = 90

end_date = pd.to_datetime(end_date)

# If using lookback approach, calculate start_date
if 'days_lookback' in locals():
    start_date = end_date - timedelta(days=days_lookback)
else:
    start_date = pd.to_datetime(start_date)

print ()

print(f'Target date range: {start_date} to {end_date}')
print(f'Window size: {(end_date - start_date).days} days\n')

dfs_dict = {}

# Calculate how many days to fetch (add buffer for safety)
days_needed = (end_date - start_date).days # + 30  # +30 day buffer
fetch_period = f'{days_needed}d'

for ticker in yf_ticker_list:
    print (f'\rFetching {ticker.ticker}', end='', flush=True)

    df = ticker.history(start=start_date, end=end_date, interval='1h', auto_adjust=True)
    # drop timezone if you like
    try:
        df = df.tz_convert(None)   # if tz-aware
    except Exception:
        pass

    df = df.reset_index()

    # normalize name to 'Date'
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})
    elif 'date' in df.columns:
        df = df.rename(columns={'date': 'Date'})
    # else keep if already 'Date'

    # Ensure Date is datetime type
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    if len(df) == 0:
        print(f'\nWarning: No data for {ticker.ticker} in specified date range')
        continue

    dfs_dict[ticker.ticker] = df
    # dfs_list.append(df)

print(f'\n\nFetched data for {len(dfs_dict)} tickers')

print('\nAligning dataframes...')

# Step 2: Create a common date range from the union of all dates
all_dates = pd.DatetimeIndex([])
for df in dfs_dict.values():
    all_dates = all_dates.union(df['Date'])

# Filter common dates to our target range (in case of any outliers)
all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
# Sort the common date index
all_dates = all_dates.sort_values()

print(f'Common date index: {len(all_dates)} timestamps')

aligned_dfs = {}

for ticker_name, df in dfs_dict.items():
    # Set Date as index for reindexing
    df = df.set_index('Date')
    
    # Reindex to common dates
    df_aligned = df.reindex(all_dates)
    
    # Interpolate missing values
    # Use 'time' method for time-series data (considers temporal distance)
    df_aligned = df_aligned.interpolate(method='time', limit_direction='both')
    
    # For any remaining NaNs at the edges, forward/backward fill
    # df_aligned = df_aligned.fillna(method='ffill').fillna(method='bfill')
    df_aligned = df_aligned.ffill().bfill()
    
    # Reset index to get Date back as a column
    df_aligned = df_aligned.reset_index().rename(columns={'index': 'Date'})
    
    aligned_dfs[ticker_name] = df_aligned

# Convert back to list if needed
dfs_list = list(aligned_dfs.values())

print(f'Aligned {len(dfs_list)} dataframes to {len(all_dates)} rows each')

# Verify alignment
print('\nVerification:')
for ticker_name, df in aligned_dfs.items():
    date_min = df['Date'].min()
    date_max = df['Date'].max()
    print(f'{ticker_name}: {len(df)} rows, Date range: {date_min} to {date_max}')

print()
# print ('\r' + ' '*100)

'''
yf_data = yf.Ticker('BCH-GBP')
data = yf_data.history(period='max', start='2020-01-01',
                       auto_adjust=True)
df = pd.DataFrame(data).reset_index()
'''

# Visualize closing prices
'''
for idx, df in enumerate(dfs_list):
    plt.plot(df['Date'], df['Close'])
    plt.savefig(f"plot{idx}.png", dpi=300, bbox_inches='tight')
'''

print ('saving plots...')

for idx, df in enumerate(dfs_list):
    plt.figure(figsize=(8, 4))
    plt.plot(df['Date'], df['Close'], label=f'Series {idx}')
    plt.title(f"Price History {idx}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot{idx}.png", dpi=300, bbox_inches='tight')
    plt.close()

# 2️⃣ Create a single united plot
plt.figure(figsize=(10, 5))
for idx, df in enumerate(dfs_list):
    plt.plot(df['Date'], df['Close'], label=f'Series {idx}')

plt.title("Combined Price History")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.tight_layout()
plt.savefig("plots_all.png", dpi=300, bbox_inches='tight')
plt.close()

args.lag = 192          # We'll use the prior 12 calendar weeks as inputs
args.horizon = 48       # We'll then try to predict out the next 20 available days (4ish working weeks)
args.target = 'Close'   # Pick one feature to forecast 

# Windows of samples
samples = [w.to_numpy() for w in df[args.target].rolling(window=args.lag + args.horizon)][args.lag + args.horizon - 1:]
# Dates for each sample
dates = [w for w in df['Date'].rolling(window=args.lag + args.horizon)][args.lag + args.horizon - 1:]

# print(samples[0].shape())
# import sys
# sys.exit()

import datetime

'''
test_year = 2025  
test_date = datetime.date(test_year, 1, min(args.horizon, 30)) 

## Convert 'Date' to datetime object
df['Date'] = pd.to_datetime(df['Date']).dt.date

## Find indices corresponding to test year dates
test_ix = len(dates) - df[df['Date'] >= test_date].shape[0]
'''

#Training and validation splits

# Check that the horizon dates are roughly in 2022
#pprint(dates[test_ix][args.lag:args.lag + args.horizon])

def train_val_split(data_indices, val_ratio=0.1):
    train_ratio = 1 - val_ratio
    last_train_index = int(np.round(len(data_indices) * train_ratio))
    return data_indices[:last_train_index], data_indices[last_train_index:]

'''
# Split data indices for train and val sets
train_indices, val_indices = train_val_split(np.arange(test_ix))
train_samples = np.array(samples[:val_indices[0]])
val_samples = np.array(samples[val_indices[0]:val_indices[-1]])
'''

# Sanity check the splits by plotting the last horizon term in each sample
'''
ix = -1
plt.plot(train_samples[:, ix], label=f'{args.target} (train)', alpha=1)
plt.plot(np.arange(len(train_samples), len(train_samples) + len(val_samples)), 
         val_samples[:, ix], label=f'{args.target} (val)', alpha=1)
plt.legend()
plt.savefig("split.png", dpi=300, bbox_inches='tight')
'''

import copy
import torch
from torch.utils.data import Dataset, DataLoader

def time_embedding(timestamp: float) -> torch.Tensor:
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR   = 60 * SECONDS_PER_MINUTE
    SECONDS_PER_DAY    = 24 * SECONDS_PER_HOUR
    SECONDS_PER_WEEK   = 7 * SECONDS_PER_DAY
    SECONDS_PER_YEAR   = 365.25 * SECONDS_PER_DAY
    SECONDS_PER_LUNAR  = 29.5306 * SECONDS_PER_DAY

    # def angle(t: float, period: float) -> float:
    #     return 2 * torch.pi * (t % period) / period

    periods = [
        SECONDS_PER_YEAR,
        SECONDS_PER_LUNAR,
        SECONDS_PER_WEEK,
        SECONDS_PER_DAY,
    ]

    t = torch.tensor(timestamp, dtype=torch.float32)
    angles = 2 * torch.pi * (t % torch.tensor(periods)) / torch.tensor(periods)
    encoding = torch.stack([torch.sin(angles), torch.cos(angles)], dim=1).flatten()
    return encoding

class MultiSeriesStockPriceDataset(Dataset):
    def __init__(self, 
                 data_list: list[np.array], 
                 lag: int, 
                 horizon: int,
                 timestamps: np.array = None,
                 use_time_encoding: bool = False):
        """
        Args:
            data_list: List of numpy arrays, each with shape (num_samples, window_size)
            lag: Number of historical timesteps
            horizon: Number of future timesteps to predict
            timestamps: Array of Unix timestamps for each sample, shape (num_samples, window_size)
            use_time_encoding: Whether to append time encodings to features
        """
        super().__init__()
        # Convert list of arrays to tensors and stack along new dimension
        # Shape: (num_samples, window_size, num_series)
        self.data_x = torch.stack([
            torch.tensor(data).float() for data in data_list
        ], dim=-1)
        
        self.data_y = copy.deepcopy(self.data_x[:, -horizon:, :])
        self.lag = lag
        self.horizon = horizon
        self.use_time_encoding = use_time_encoding
        self.timestamps = timestamps
        
        # Pre-compute time encodings if needed
        if use_time_encoding:
            if timestamps is None:
                raise ValueError("timestamps must be provided when use_time_encoding=True")
            self.time_encodings = self._compute_time_encodings()
        
    def _compute_time_encodings(self):
        """
        Compute time encodings for all samples and timesteps.
        
        Returns:
            Tensor of shape (num_samples, window_size, 8)
            where 8 = 4 periods * 2 (sin, cos)
        """
        num_samples, window_size = self.timestamps.shape
        encodings = torch.zeros(num_samples, window_size, 8)

        print ('computing time embeddings')

        for i in range(num_samples):
            print (f'\r{i + 1} of {num_samples}', end='', flush=True)
            for j in range(window_size):
                encodings[i, j] = time_embedding(self.timestamps[i, j])
        print()
        return encodings
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, idx):
        x = self.data_x[idx].clone()  # Shape: (lag+horizon, num_series)
        y = self.data_y[idx]  # Shape: (horizon, num_series)
        
        if self.use_time_encoding:
            # Append time encodings to features
            time_enc = self.time_encodings[idx]  # Shape: (lag+horizon, 8)
            x = torch.cat([x, time_enc], dim=-1)  # Shape: (lag+horizon, num_series + 8)
        
        x[-self.horizon:, :len(self.data_y[0, 0])] = 0  # Mask input horizon terms (only price data, not time)
        
        return x, y, (self.lag, self.horizon)
    
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x

def load_data(df_list: list[pd.DataFrame],
              lag: int,
              horizon: int,
              target: str,
              val_ratio: float,
              test_year_month_day: list[int],
              use_time_encoding: bool = False,
              date_column: str = 'Date',
              **dataloader_kwargs):
    """
    Load synchronized data from multiple dataframes.
    
    Args:
        df_list: List of dataframes with same dates and structure
        lag: Number of historical timesteps
        horizon: Number of future timesteps to predict
        target: Column name containing the target values
        val_ratio: Ratio of validation data
        test_year_month_day: [year, month, day] for test split
        use_time_encoding: Whether to add cyclical time encodings
        date_column: Name of the date column in dataframes
        **dataloader_kwargs: Additional arguments for DataLoader
    
    Returns:
        List of dataloaders [train_loader, val_loader, test_loader]
    """
    # Ensure all dataframes have the same dates
    all_samples = []
    
    for df in df_list:
        # Convert day-wise data into sequences of lag + horizon terms
        samples = [w.to_numpy() for w in df[target].rolling(window=lag + horizon)][lag + horizon - 1:]
        all_samples.append(samples)
    
    # Use dates from first dataframe (assuming all are synchronized)
    df_list[0][date_column] = pd.to_datetime(df_list[0][date_column])
    dates = [w for w in df_list[0][date_column].rolling(window=lag + horizon)][lag + horizon - 1:]
    
    # Prepare timestamps if time encoding is requested
    timestamps = None
    if use_time_encoding:
        # Convert dates to Unix timestamps for each window
        timestamps = []
        for date_window in dates:
            window_timestamps = [d.timestamp() for d in date_window]
            timestamps.append(window_timestamps)
        timestamps = np.array(timestamps)  # Shape: (num_samples, window_size)
    
    # Set aside test samples by date
    test_date = pd.to_datetime(datetime.datetime(*test_year_month_day)).date()

    df_list[0]['_date_only'] = df_list[0][date_column].dt.date
    test_ix = len(dates) - df_list[0][df_list[0]['_date_only'] >= test_date].shape[0]
    
    # Split each series
    test_samples_list = [np.array(samples[test_ix:]) for samples in all_samples]
    test_timestamps = timestamps[test_ix:] if timestamps is not None else None
    
    # Get training + validation samples

    train_indices, val_indices = train_val_split(
        np.arange(len(dates[:test_ix])), 
        val_ratio=val_ratio  # Explicitly pass as keyword arg
    )

    train_samples_list = [np.array(samples[:val_indices[0]]) for samples in all_samples]
    val_samples_list = [np.array(samples[val_indices[0]:val_indices[-1]]) for samples in all_samples]
    
    train_timestamps = timestamps[:val_indices[0]] if timestamps is not None else None
    val_timestamps = timestamps[val_indices[0]:val_indices[-1]] if timestamps is not None else None
    
    # PyTorch datasets and dataloaders
    datasets = [
        MultiSeriesStockPriceDataset(
            sample_list, 
            lag, 
            horizon,
            timestamps=ts,
            use_time_encoding=use_time_encoding
        )
        for sample_list, ts in [
            (train_samples_list, train_timestamps),
            (val_samples_list, val_timestamps),
            (test_samples_list, test_timestamps)
        ]
    ]
    
    dataloaders = [DataLoader(dataset, shuffle=True if ix == 0 else False, **dataloader_kwargs)
                   for ix, dataset in enumerate(datasets)]
    
    return dataloaders

# Function to visualize samples over time
def visualize_data(dataloaders, sample_idx, sample_dim=0,
                   splits=['train', 'val', 'test'], title=None):
    assert len(splits) == len(dataloaders)
    start_idx = 0
    for idx, split in enumerate(splits):
        y = dataloaders[idx].dataset.data_x[:, sample_idx, sample_dim]
        x = np.arange(len(y)) + start_idx
        plt.plot(x, y, label=split)
        start_idx += len(x)
    plt.title(title)
    plt.legend()
    plt.savefig("vis.png")

# Again we use OmegaConf bc it's great
dataset_configs = f"""
lag: {args.lag}
horizon: {args.horizon}
target: Close
val_ratio: 0.03
test_year_month_day:
- 2025
- 10
- 1
use_time_encoding: true
"""
dataset_configs = OmegaConf.create(dataset_configs)

dataloader_configs = """
batch_size: 32
num_workers: 2
pin_memory: true
"""
dataloader_configs = OmegaConf.create(dataloader_configs)

# Load and visualize data
torch.manual_seed(args.seed)

print ('buildiing dataset...')
dataloaders = load_data(dfs_list, **dataset_configs, **dataloader_configs)
train_loader, val_loader, test_loader = dataloaders

#visualize_data(dataloaders, sample_idx=0)

# We've got 4 main components to specify: 
# 1. The embedding / input projection (e.g., an MLP)
# 2. The encoder block ("open-loop" / convolutional SpaceTime SSMs go here)
# 3. The decoder block ("closed-loop" / recurrent SpaceTime SSMs go here)
# 4. The output projection (e.g., an MLP)

config_dir = 'spacetime/configs/'

embedding_config = """
method: linear_mod
kwargs:
  input_dim: 4
  embedding_dim: 256
"""
embedding_config = OmegaConf.create(embedding_config)

encoder_config = """
blocks:
- input_dim: 256
  pre_config: 'ssm/preprocess/residual'
  ssm_config: 'ssm/companion_preprocess'
  mlp_config: 'mlp/default'
  skip_connection: true
  skip_preprocess: false
"""
encoder_config = OmegaConf.create(encoder_config)

decoder_config = """
blocks:
- input_dim: 256
  pre_config: 'ssm/preprocess/none'
  ssm_config: 'ssm/closed_loop/companion'
  mlp_config: 'mlp/identity'
  skip_connection: false
  skip_preprocess: false
"""
decoder_config = OmegaConf.create(decoder_config)

output_config = """
input_dim: 256
output_dim: 1
method: mlp
kwargs:
  input_dim: 256
  output_dim: 1
  activation: gelu
  dropout: 0.2
  layernorm: false
  n_layers: 1
  n_activations: 1
  pre_activation: true
  input_shape: bld
  skip_connection: false
  average_pool: null
"""
output_config = OmegaConf.create(output_config)

from model.network import SpaceTime
from setup import seed_everything

# Initialize SpaceTime encoder and decoder preprocessing, SSM, and MLP components
# - These are referenced as paths in the above encoder and decoder configs


def init_encoder_decoder_config(config, config_dir):
    for ix, _config in enumerate(config['blocks']):
        # Load preprocess kernel configs
        c_path = join(config_dir, f"{_config['pre_config']}.yaml")
        _config['pre_config'] = OmegaConf.load(c_path)
        # Load SSM kernel configs
        c_path = join(config_dir, f"{_config['ssm_config']}.yaml")
        _config['ssm_config'] = OmegaConf.load(c_path)
        # Load MLP configs
        c_path = join(config_dir, f"{_config['mlp_config']}.yaml")
        _config['mlp_config'] = OmegaConf.load(c_path)
    return config

encoder_config = init_encoder_decoder_config(encoder_config, join(config_dir, 'model'))
decoder_config = init_encoder_decoder_config(decoder_config, join(config_dir, 'model'))

# Initialize SpaceTime model
model_configs = {
    'embedding_config': embedding_config,
    'encoder_config': encoder_config,
    'decoder_config': decoder_config,
    'output_config': output_config,
    'lag': dataset_configs.lag,
    'horizon': dataset_configs.horizon
}
seed_everything(args.seed)

from utils.config import print_config  # View OmegaConf configs

from loss import get_loss
from data_transforms import get_data_transforms
from optimizer import get_optimizer, get_scheduler
from setup.configs.optimizer import get_optimizer_config, get_scheduler_config

from train import train_model, evaluate_model, plot_forecasts

arg_config = f"""
lag: {dataset_configs.lag}
horizon: {dataset_configs.horizon}
features: S
lr: 1e-4
weight_decay: 1e-4
dropout: 0.25
criterion_weights:
- 10
- 1
- 10
optimizer: adamw
scheduler: timm_cosine
max_epochs: 500
early_stopping_epochs: 20
data_transform: mean
loss: informer_rmse
val_metric: informer_rmse
seed: 42
dataset: sp500
variant: null
model: SpaceTime
"""
class Args():
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)
            
args = Args(OmegaConf.create(arg_config))
# GPU
args.device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

# These others are not super important
args.checkpoint_dir = './checkpoints'
args.log_dir = './log_dir'
args.variant = None
args.no_wandb = True
args.dataset_type = 'informer'  # for standard forecasting
args.log_epoch = 1000

seed_everything(args.seed)
model = SpaceTime(**model_configs)  # Reset model from here

model.set_lag(args.lag)
model.set_horizon(args.horizon)
    
# Initialize optimizer and scheduler
optimizer = get_optimizer(model, get_optimizer_config(args, config_dir))
scheduler = get_scheduler(model, optimizer, get_scheduler_config(args, config_dir))
    
# Loss objectives
criterions = {name: get_loss(name) for name in ['rmse', 'mse', 'mae']}
eval_criterions = criterions
for name in ['rmse', 'mse', 'mae']:
    eval_criterions[f'informer_{name}'] = get_loss(f'informer_{name}')
    
# Data transforms, e.g., normalization
input_transform, output_transform = get_data_transforms(args.data_transform, args.lag)

from setup import initialize_experiment

initialize_experiment(args, experiment_name_id='',
                      best_train_metric=1e10, 
                      best_val_metric=1e10)

# Actually train model
splits = ['train', 'val', 'test']
dataloaders_by_split = {split: dataloaders[ix] 
                        for ix, split in enumerate(splits)}

model = train_model(model, optimizer, scheduler, dataloaders_by_split, 
                    criterions, max_epochs=args.max_epochs, config=args, 
                    input_transform=input_transform,
                    output_transform=output_transform,
                    val_metric=args.val_metric, wandb=None, 
                    return_best=True, early_stopping_epochs=args.early_stopping_epochs) 

from dataloaders import get_evaluation_loaders
from train.evaluate import plot_forecasts

eval_splits = ['eval_train', 'val', 'test']
eval_loaders = get_evaluation_loaders(dataloaders, batch_size=dataloader_configs.batch_size)
eval_loaders_by_split = {split: eval_loaders[ix] for ix, split in
                         enumerate(eval_splits)}
model, log_metrics, total_y = evaluate_model(model, dataloaders=eval_loaders_by_split, 
                                             optimizer=optimizer, scheduler=scheduler, 
                                             criterions=eval_criterions, config=args,
                                             epoch=args.best_val_metric_epoch, 
                                             input_transform=input_transform, 
                                             output_transform=output_transform,
                                             val_metric=args.val_metric, wandb=None,
                                             train=False)
n_plots = len(splits) # train, val, test
fig, axes = plt.subplots(1, n_plots, figsize=(6.4 * n_plots, 4.8))

plot_forecasts(total_y, splits=eval_splits, axes=axes)