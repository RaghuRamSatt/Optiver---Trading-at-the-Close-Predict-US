#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().system('pip uninstall -q -y torchaudio torchdata torchtext torchvision')

get_ipython().system('pip install -q "torch<2.0.0"  -f /kaggle/input/pytorch-tabular-python-package/ --no-index')
get_ipython().system('pip install pytorch_tabular -f /kaggle/input/pytorch-tabular-python-package/ --no-index')
get_ipython().system('pip list | grep torch')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[10]:



import gc
import os
import time
import warnings
from itertools import combinations
from warnings import simplefilter


import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
#from pytorch_tabular import TabularModel

warnings.filterwarnings("ignore")
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# In[11]:


max_lookback = np.nan
# split_day = 435  # Split day for time series data

df_train = pd.read_csv("/kaggle/input/optiver-trading-at-the-close/train.csv")
df_train.dropna(subset=["target"], inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_shape = df_train.shape


# In[12]:


def reduce_mem_usage(df, verbose=1):
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)

    if verbose:
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        decrease = 100 * (start_mem - end_mem) / start_mem
        print(f"Decreased by {decrease:.2f}%")

    return df


# In[13]:


from numba import njit, prange

@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


# In[14]:


def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    features_array = compute_triplet_imbalance(df_values, comb_indices)

    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    return pd.DataFrame(features_array, columns=columns)


# In[15]:


import torch

IS_CUDA = torch.cuda.is_available()
NB_CARDS = torch.cuda.device_count()
print(f"{IS_CUDA=} with {NB_CARDS=}")

def imbalance_features(df):
    if IS_CUDA:
        import cudf
        df = cudf.from_pandas(df)
    
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]

    # V1 features
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("ask_price + bid_price")/2
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval("imbalance_size-matched_size")/df.eval("matched_size+imbalance_size")
    df["size_imbalance"] = df.eval("bid_size / ask_size")
    
    # Create features for pairwise price imbalances
    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")
        
    # V2 features
    # Calculate additional features
    df["imbalance_momentum"] = df.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / df['matched_size']
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(['stock_id'])['price_spread'].diff()
    df['price_pressure'] = df['imbalance_size'] * (df['ask_price'] - df['bid_price'])
    df['market_urgency'] = df['price_spread'] * df['liquidity_imbalance']
    df['depth_pressure'] = (df['ask_size'] - df['bid_size']) * (df['far_price'] - df['near_price'])
    
    # Calculate various statistical aggregation features
    
    # V3 features
    # Calculate shifted and return features for specific columns
    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby('stock_id')[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby('stock_id')[col].pct_change(window)
    
    # Calculate diff features for specific columns
    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size']:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    if IS_CUDA:
        df = df.to_pandas()
    # Replace infinite values with 0
    return df.replace([np.inf, -np.inf], 0)


# In[16]:


def numba_imb_features(df):
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    
    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)
        
    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values
    return df


# In[17]:


def other_features(df):
    df["dow"] = df["date_id"] % 5  
    df["seconds"] = df["seconds_in_bucket"] % 60  
    df["minute"] = df["seconds_in_bucket"] // 60  
    df['time_to_market_close'] = 540 - df['seconds_in_bucket']

    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())
    return df


# In[18]:


def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    
    df = imbalance_features(df)
    df = numba_imb_features(df)
    df = other_features(df)
    gc.collect()  # Perform garbage collection to free up memory
    
    feature_name = [i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]]
    return df[feature_name]


# In[19]:


g_ask_size_median = df_train.groupby("stock_id")["ask_size"].median()
g_ask_size_std = df_train.groupby("stock_id")["ask_size"].std()
g_ask_size_min = df_train.groupby("stock_id")["ask_size"].min()
g_ask_size_max = df_train.groupby("stock_id")["ask_size"].max()
g_bid_size_median = df_train.groupby("stock_id")["bid_size"].median()
g_bid_size_std = df_train.groupby("stock_id")["bid_size"].std()
g_bid_size_min = df_train.groupby("stock_id")["bid_size"].min()
g_bid_size_max = df_train.groupby("stock_id")["bid_size"].max()
global_stock_id_feats = {
    "median_size": g_bid_size_median + g_ask_size_median,
    "std_size": g_bid_size_std + g_ask_size_std,
    "ptp_size": g_bid_size_max - g_bid_size_min,
    "median_price": g_bid_size_median + g_ask_size_median,
    "std_price": g_bid_size_std + g_ask_size_std,
    "ptp_price": g_bid_size_max - g_ask_size_min,
}
df_train_feats = generate_all_features(df_train)
print("Build Online Train Feats Finished.")

df_train_feats = reduce_mem_usage(df_train_feats)
df_train_target = df_train['target'].astype(np.float16)

df_train_date_ids = df_train['date_id'].values

del df_train
gc.collect()


# In[20]:


# df_train.fillna(df_train.median(), inplace=True)

df_train_feats['far_price'].fillna(0, inplace=True)
df_train_feats['near_price'].fillna(1, inplace=True)

cols_group_by = ['stock_id', 'imbalance_buy_sell_flag']
train_grouped_median = df_train_feats.groupby(cols_group_by).transform('median')
df_train_feats.fillna(train_grouped_median, inplace=True)
print(df_train_feats.isnull().sum().sum())


# In[21]:


TARGET_NAME = "target"
CAT_FEATURES = ["stock_id"] + ["dow"] + [c for c in df_train_feats.columns if c.startswith("imbalance_buy_sell_flag")]
NUM_FEATURES = [c for c in df_train_feats.columns if c not in CAT_FEATURES]
FEATURE_NAMES = CAT_FEATURES + NUM_FEATURES
print(f"Feature length = {len(FEATURE_NAMES)} as \n{sorted(CAT_FEATURES)}\nand\n{sorted(NUM_FEATURES)}")


# In[22]:


from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.models import FTTransformerConfig, TabTransformerConfig

data_config = DataConfig(
    target=[TARGET_NAME],
    continuous_cols=NUM_FEATURES,
    categorical_cols=CAT_FEATURES,
    #normalize_continuous_features=True,
)
model_config = FTTransformerConfig(
    task="regression",
    #input_embed_dim=8,
    num_attn_blocks=4,
    num_heads=2,
    loss="L1Loss",
    metrics=["mean_absolute_error"],
)
trainer_config = TrainerConfig(
#     accelerator="cpu",
#     devices=os.cpu_count(),
    accelerator="gpu",
    devices=1,
    batch_size=1024,
    accumulate_grad_batches=4,
    max_epochs=5,
    early_stopping="valid_loss",
    early_stopping_patience=3,
    checkpoints="valid_loss",
    progress_bar=False,
)
optimizer_config = OptimizerConfig()

TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

PATH_SAVE_MODELS = '.'  # Directory to save models
#os.makedirs(model_save_path, exist_ok=True)


# In[ ]:


from sklearn.metrics import mean_absolute_error

# The total number of date_ids is 480, we split them into 5 folds with a gap of 5 days in between
num_folds, gap = 5, 5
fold_size = 480 // num_folds
models, scores = [], []

for i in range(num_folds):
    start = i * fold_size
    end = start + fold_size
    
    # Define the training and testing sets by date_id
    if i < num_folds - 1: 
        purged_start = end - 2
        purged_end = end + gap + 2
        train_indices = (df_train_date_ids >= start) & (df_train_date_ids < purged_start) | (df_train_date_ids > purged_end)
    else:
        train_indices = (df_train_date_ids >= start) & (df_train_date_ids < end)
    
    test_indices = (df_train_date_ids >= end) & (df_train_date_ids < end + fold_size)
    
    spl_params = dict(n=min(sum(train_indices), 300_000), random_state=42)
    fold_train = df_train_feats[FEATURE_NAMES][train_indices].sample(**spl_params)
    fold_train[TARGET_NAME] = df_train_target[train_indices].sample(**spl_params)
    fold_valid = df_train_feats[FEATURE_NAMES][test_indices]
    fold_valid[TARGET_NAME] = df_train_target[test_indices]

    print(f"Fold {i+1} Model Training")
    # Train a TabNet model for the current fold
    model_config = FTTransformerConfig(
        task="regression",
        input_embed_dim=16,
        num_attn_blocks=1 + i // 2,
        num_heads=2,
        loss="L1Loss",
        metrics=["mean_absolute_error"],
    )
    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    model.fit(train=fold_train, validation=fold_valid)
    # Append the model to the list
    models.append(model)
    # Free up memory by deleting fold specific variables
    del fold_train

    # Evaluate model performance on the validation set
    fold_predictions = model.predict(fold_valid, device="cuda", include_input_features=False)
    fold_score = mean_absolute_error(fold_predictions.values, fold_valid[TARGET_NAME].values)
    scores.append(fold_score)
    print(f"Fold {i+1} MAE: {fold_score}")
    # Free up memory by deleting fold specific variables
    del fold_valid
    
    # Save the model to a file
    model_path = os.path.join(PATH_SAVE_MODELS, f'tabular_{i+1}')
    os.makedirs(model_path, exist_ok=True)
    model.save_model(model_path)
    model.save_model_for_inference(model_path + ".pt")
    print(f"Model for fold {i+1} saved to {model_path}")
    
    gc.collect(), time.sleep(5)
    #torch.cuda.empty_cache()
    get_ipython().system('rm */*.sav')

# Now 'models' holds the trained models for each fold and 'scores' holds the validation scores
print(f"Average MAE across all folds: {np.mean(scores)}")


# In[ ]:


spl_params = dict(n=min(len(df_train_feats), 400_000), random_state=42)
all_train = df_train_feats[FEATURE_NAMES].sample(**spl_params)
all_train[TARGET_NAME] = df_train_target.sample(**spl_params)

print(f"Training final model with ...")
# Train the final model on the entire dataset
model = tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
model.fit(train=all_train)

# Save the final model to a file
model_path = os.path.join(PATH_SAVE_MODELS, 'tabular_final')
os.makedirs(model_path, exist_ok=True)
model.save_model(model_path)
model.save_model_for_inference(model_path + ".pt")
print(f"Model final saved to {model_path}")

# Append the final model to the list of models
models.append(model)
get_ipython().system('rm */*.sav')


# In[ ]:


import optiver2023
env = optiver2023.make_env()
iter_test = env.iter_test()

def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices) / np.sum(std_error)
    out = prices - std_error * step
    return out


# In[ ]:


counter = 0
y_min, y_max = -64, 64
qps, predictions = [], []
cache = pd.DataFrame()

# Weights for each fold model
model_weights = [1. / len(models)] * len(models) 

for (test, revealed_targets, sample_prediction) in iter_test:
    now_time = time.time()
    cache = pd.concat([cache, test], ignore_index=True, axis=0)
    if counter > 0:
        cache = cache.groupby(['stock_id']).tail(21).sort_values(
            by=['date_id', 'seconds_in_bucket', 'stock_id']).reset_index(drop=True)
    feat = generate_all_features(cache)[-len(test):]
    feat.fillna(train_grouped_median, inplace=True)

    # Generate predictions for each model and calculate the weighted average
    predictions = np.zeros(len(test))
    for model, weight in zip(models, model_weights):
        predictions += weight * model.predict(
            feat, device="cuda", include_input_features=False).values[:, 0]

    predictions = zero_sum(predictions, test['bid_size'] + test['ask_size'])
    clipped_predictions = np.clip(predictions, y_min, y_max)
    sample_prediction['target'] = clipped_predictions
    env.predict(sample_prediction)
    qps.append(time.time() - now_time)
    if counter % 10 == 0:
        print(counter, 'qps:', np.mean(qps))
    if counter < 3:
        display(sample_prediction.head())
    counter += 1

time_cost = 1.146 * np.mean(qps)
print(f"The code will take approximately {np.round(time_cost, 4)} hours to reason about")


# In[ ]:


get_ipython().system('head submission.csv')

