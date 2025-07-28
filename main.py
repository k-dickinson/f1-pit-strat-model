import os
import pandas as pd
import numpy as np
import fastf1
from fastf1 import plotting
from fastf1.core import Laps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from datetime import timedelta

# 1. CONFIGURATION
CACHE_DIR = 'fastf1_cache/2023/'
# Dynamically find available races in the cache directory
RACES = [d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))]
SESSIONS = ['Race']

# 2. DATA LOADING
# Helper to load laps for a given race
def load_laps_for_race(race_dir):
    session_info_path = os.path.join(CACHE_DIR, race_dir, '2023-07-30_Race', 'session_info.ff1pkl')
    if not os.path.exists(session_info_path):
        # Try to find the session subdir
        subdirs = os.listdir(os.path.join(CACHE_DIR, race_dir))
        for sub in subdirs:
            if 'Race' in sub:
                session_info_path = os.path.join(CACHE_DIR, race_dir, sub, 'session_info.ff1pkl')
                break
    if not os.path.exists(session_info_path):
        print(f"Session info not found for {race_dir}")
        return None
    # Use FastF1 to load session
    event = fastf1.get_event(2023, race_dir.split('_')[1] + ' ' + race_dir.split('_')[2])
    session = event.get_session('R')
    session.load()
    laps = session.laps
    return laps

# 3. FEATURE ENGINEERING
def create_features(laps: Laps):
    df = laps.copy().reset_index(drop=True)
    # Basic features
    df['LapNumber'] = df['LapNumber']
    df['Compound'] = df['Compound'].astype('category').cat.codes
    df['Stint'] = df['Stint']
    df['TrackStatus'] = df['TrackStatus'].fillna(0).astype(int)
    df['IsPersonalBest'] = df['IsPersonalBest'].astype(int)
    # Add more features as needed
    return df

# 4. LABEL CREATION (Next pit lap)
def add_next_pit_label(df):
    df = df.sort_values(['Driver', 'LapNumber'])
    df['NextPitLap'] = np.nan
    for drv in df['Driver'].unique():
        drv_df = df[df['Driver'] == drv]
        pit_laps = drv_df[drv_df['PitInTime'].notnull()]['LapNumber'].values
        for idx, row in drv_df.iterrows():
            future_pits = pit_laps[pit_laps > row['LapNumber']]
            if len(future_pits) > 0:
                df.at[idx, 'NextPitLap'] = future_pits[0]
    # Target: how many laps until next pit
    df['LapsToNextPit'] = df['NextPitLap'] - df['LapNumber']
    df = df.dropna(subset=['LapsToNextPit'])
    return df

# 5. DATASET ASSEMBLY
all_laps = []
for race in RACES:
    laps = load_laps_for_race(race)
    if laps is not None:
        df = create_features(laps)
        df = add_next_pit_label(df)
        df = df.copy()  # Ensure we are not assigning to a view
        df.loc[:, 'Race'] = race  # Use .loc to avoid SettingWithCopyWarning
        all_laps.append(df)
if not all_laps:
    raise RuntimeError('No race data loaded!')
data = pd.concat(all_laps, ignore_index=True)

# 6. TIME SERIES SPLIT
if len(RACES) >= 2:
    train_data = data[data['Race'] == RACES[0]]
    test_data = data[data['Race'] == RACES[1]]
else:
    # Only one race available, split by lap number (e.g., 70% train, 30% test)
    race = RACES[0]
    race_data = data[data['Race'] == race].sort_values(by='LapNumber')
    split_idx = int(0.7 * len(race_data))
    train_data = race_data.iloc[:split_idx]
    test_data = race_data.iloc[split_idx:]

FEATURES = ['LapNumber', 'Compound', 'Stint', 'TrackStatus', 'IsPersonalBest']
TARGET = 'LapsToNextPit'

X_train = train_data[FEATURES]
y_train = train_data[TARGET]
X_test = test_data[FEATURES]
y_test = test_data[TARGET]

# 7. LIGHTGBM TRAINING WITH EARLY STOPPING
train_set = lgb.Dataset(X_train, y_train)
valid_set = lgb.Dataset(X_test, y_test, reference=train_set)
params = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'seed': 42,
    'max_depth': 4,
    'min_data_in_leaf': 20,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'num_leaves': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1
}
model = lgb.train(
    params,
    train_set,
    valid_sets=[train_set, valid_set],
    valid_names=['train', 'valid'],
    num_boost_round=200,
    callbacks=[
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=50)
    ]
)

# 8. EVALUATION
preds = model.predict(X_test, num_iteration=model.best_iteration)
train_preds = model.predict(X_train, num_iteration=model.best_iteration)
mae = mean_absolute_error(y_test, preds)
train_mae = mean_absolute_error(y_train, train_preds)

# Percent of exact predictions (rounded to nearest integer)
y_test_arr = np.array(y_test)
correct = np.sum(np.round(preds) == y_test_arr)
percent_correct = 100 * correct / len(y_test_arr)

# Naive baseline: always predict mean of y_train
naive_pred = np.full_like(y_test, y_train.mean())
naive_mae = mean_absolute_error(y_test, naive_pred)
improvement = 100 * (naive_mae - mae) / naive_mae if naive_mae != 0 else 0

# --- Compute additional metrics for display ---
within_metrics = []
for n in [1, 2, 3]:
    within_n = 100 * np.mean(np.abs(np.round(preds) - y_test_arr) <= n)
    within_metrics.append(within_n)

# 9. FEATURE IMPORTANCE + METRICS SIDE BY SIDE (CLEANED UP)
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
lgb.plot_importance(model, ax=ax1)
ax1.set_title('Feature Importance')

# Hide axis for metrics panel
ax2.axis('off')
metrics_text = (
    f"Train MAE: {train_mae:.3f} laps\n"
    f"Test MAE: {mae:.3f} laps\n"
    f"Model improvement over naive: {improvement:.2f}%\n"
    f"Percent correct (exact lap): {percent_correct:.2f}%\n"
    f"Percent within ±1 lap: {within_metrics[0]:.2f}%\n"
    f"Percent within ±2 laps: {within_metrics[1]:.2f}%\n"
    f"Percent within ±3 laps: {within_metrics[2]:.2f}%"
)
ax2.text(0.01, 0.98, metrics_text, fontsize=14, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.9), linespacing=1.5)
plt.tight_layout()
plt.show()

# 10. SAVE MODEL (optional)
# model.save_model('pitstop_lgbm.txt')

# 11. NOTES
# - This is a minimal pipeline. For production, add more features, error handling, and experiment tracking.
# - For more races, expand RACES and automate splits.
# - For live prediction, use only laps up to current lap for each car.
# If overfitting persists, try removing 'LapNumber' from FEATURES.
