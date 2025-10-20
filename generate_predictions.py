"""
RiskGrid - Generate Predictions CSV with Ultra Features
Creates ultra features and generates predictions
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("=" * 80)
print(" " * 20 + "RISKGRID PREDICTIONS GENERATOR")
print("=" * 80)
# STEP 1: LOAD FEATURE DATA

print("\n[STEP 1] Loading feature matrix...")

try:
    df = pd.read_csv("data/features/feature_matrix.csv")
    df['time_window'] = pd.to_datetime(df['time_window'])
    print(f"  ✓ Loaded {len(df):,} observations")
    print(f"  ✓ Existing columns: {len(df.columns)}")
except Exception as e:
    print(f"  ❌ Error: {e}")
    exit(1)
# STEP 2: CREATE ULTRA FEATURES
print("\n[STEP 2] Creating ultra-advanced features...")

# Chicago city center
CHICAGO_CENTER_LAT = 41.8781
CHICAGO_CENTER_LON = -87.6298

# Check if ultra features already exist
if 'dist_to_center' not in df.columns:
    print("  • Creating spatial features...")
    df['dist_to_center'] = np.sqrt(
        (df['cell_center_lat'] - CHICAGO_CENTER_LAT)**2 + 
        (df['cell_center_lon'] - CHICAGO_CENTER_LON)**2
    )

if 'hour_fourier_1' not in df.columns:
    print("  • Creating Fourier features...")
    df['hour_fourier_1'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_fourier_2'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['hour_fourier_3'] = np.sin(4 * np.pi * df['hour'] / 24)
    df['hour_fourier_4'] = np.cos(4 * np.pi * df['hour'] / 24)
    df['week_fourier_1'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_fourier_2'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

if 'is_rush_hour' not in df.columns:
    print("  • Creating temporal indicators...")
    df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                          (df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_prime_crime_time'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)

if 'ewma_3' not in df.columns:
    print("  • Creating EWMA features...")
    df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
    df['ewma_3'] = df.groupby('cell_id')['incident_count'].transform(
        lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
    )
    df['ewma_7'] = df.groupby('cell_id')['incident_count'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
    )

if 'volatility_3' not in df.columns:
    print("  • Creating volatility features...")
    df['volatility_3'] = df.groupby('cell_id')['incident_count'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std().shift(1)
    )
    df['volatility_7'] = df.groupby('cell_id')['incident_count'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std().shift(1)
    )

if 'acceleration' not in df.columns:
    print("  • Creating acceleration features...")
    df['acceleration'] = df.groupby('cell_id')['momentum'].transform(
        lambda x: x.diff(1)
    )

if 'lag1_x_ewma3' not in df.columns:
    print("  • Creating interaction features...")
    df['lag1_x_ewma3'] = df['lag_1'] * df['ewma_3']
    df['trend_x_volatility'] = df['trend_short'] * df['volatility_3']
    df['neighbor_x_hour'] = df['neighbor_mean'] * df['hour']
    df['cellavg_x_weekend'] = df['cell_avg_incidents'] * df['is_weekend']
    df['dist_x_night'] = df['dist_to_center'] * df['is_night']

if 'lag1_percentile' not in df.columns:
    print("  • Creating percentile features...")
    df['lag1_percentile'] = df.groupby('cell_id')['lag_1'].transform(
        lambda x: x.rank(pct=True)
    )

# Fill NaN values
ultra_features = ['dist_to_center', 'hour_fourier_1', 'hour_fourier_2', 'hour_fourier_3', 
                  'hour_fourier_4', 'week_fourier_1', 'week_fourier_2', 'is_rush_hour',
                  'is_night', 'is_prime_crime_time', 'ewma_3', 'ewma_7', 'volatility_3',
                  'volatility_7', 'acceleration', 'lag1_x_ewma3', 'trend_x_volatility',
                  'neighbor_x_hour', 'cellavg_x_weekend', 'dist_x_night', 'lag1_percentile']

df[ultra_features] = df[ultra_features].fillna(0)

print(f"  ✓ Created {len(ultra_features)} ultra features")
print(f"  ✓ Total columns now: {len(df.columns)}")
# STEP 3: LOAD MODEL AND SCALER
print("\n[STEP 3] Loading trained model...")

try:
    model = joblib.load("models/ultra/xgb_ultra.pkl")
    scaler = joblib.load("models/ultra/ultra_scaler.pkl")
    feature_names = joblib.load("models/ultra/ultra_feature_names.pkl")
    print(f"  ✓ Loaded XGBoost Ultra model")
    print(f"  ✓ Model expects {len(feature_names)} features")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
# STEP 4: PREPARE DATA
print("\n[STEP 4] Preparing data for prediction...")

# Check if all required features exist
missing_features = [f for f in feature_names if f not in df.columns]

if missing_features:
    print(f"Warning: {len(missing_features)} features still missing:")
    for feat in missing_features[:10]:
        print(f"     • {feat}")
    if len(missing_features) > 10:
        print(f"     ... and {len(missing_features) - 10} more")
    
    print("\n  Creating placeholder features with zeros...")
    for feat in missing_features:
        df[feat] = 0

# Get features in correct order
try:
    X = df[feature_names].copy()
    print(f"  ✓ Prepared {len(X):,} samples with {len(feature_names)} features")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
# STEP 5: SCALE AND PREDICT
print("\n[STEP 5] Scaling features and generating predictions...")

X_scaled = scaler.transform(X)
print(f"  ✓ Features scaled")

predictions = model.predict(X_scaled)
print(f"  ✓ Generated {len(predictions):,} predictions")
print(f"     Min: {predictions.min():.3f}")
print(f"     Max: {predictions.max():.3f}")
print(f"     Mean: {predictions.mean():.3f}")
print(f"     Median: {np.median(predictions):.3f}")
# STEP 6: CREATE RESULTS DATAFRAME
print("\n[STEP 6] Creating results dataframe...")

# Keep only necessary columns for dashboard
keep_cols = ['cell_id', 'time_window', 'cell_center_lat', 'cell_center_lon',
             'grid_lat_idx', 'grid_lon_idx', 'target', 'hour', 'day_of_week', 
             'is_weekend']

# Add optional columns if they exist
optional_cols = ['cell_avg_incidents', 'rolling_mean_14', 'lag_1', 'momentum', 
                 'neighbor_mean', 'hour_avg_incidents', 'month', 'day_of_month']
for col in optional_cols:
    if col in df.columns:
        keep_cols.append(col)

results_df = df[keep_cols].copy()
results_df['ultra_predicted'] = predictions
results_df['ultra_error'] = abs(results_df['target'] - predictions)

# Risk levels
results_df['risk_level'] = pd.cut(
    predictions,
    bins=[0, 1, 3, 100],
    labels=['Low', 'Medium', 'High']
)

print(f"  ✓ Created results with {len(results_df.columns)} columns")
# STEP 7: SPLIT INTO TRAIN/VAL/TEST
print("\n[STEP 7] Splitting into train/val/test sets...")

results_df = results_df.sort_values('time_window').reset_index(drop=True)

train_end = int(len(results_df) * 0.70)
val_end = int(len(results_df) * 0.85)

train_df = results_df.iloc[:train_end]
val_df = results_df.iloc[train_end:val_end]
test_df = results_df.iloc[val_end:]

print(f"  • Train: {len(train_df):,} ({train_df['time_window'].min().date()} to {train_df['time_window'].max().date()})")
print(f"  • Val:   {len(val_df):,} ({val_df['time_window'].min().date()} to {val_df['time_window'].max().date()})")
print(f"  • Test:  {len(test_df):,} ({test_df['time_window'].min().date()} to {test_df['time_window'].max().date()})")
# STEP 8: SAVE FILES
print("\n[STEP 8] Saving prediction files...")

os.makedirs("outputs/ultra", exist_ok=True)

# Save full predictions
results_df.to_csv("outputs/ultra/ultra_predictions.csv", index=False)
print(f"  ✓ Saved: outputs/ultra/ultra_predictions.csv ({len(results_df):,} rows)")

# Save test predictions for compatibility
test_df.to_csv("outputs/test_predictions.csv", index=False)
print(f"  ✓ Saved: outputs/test_predictions.csv ({len(test_df):,} rows)")
# STEP 9: VALIDATION
print("\n[STEP 9] Validation...")

test_rmse = np.sqrt(mean_squared_error(test_df['target'], test_df['ultra_predicted']))
test_mae = mean_absolute_error(test_df['target'], test_df['ultra_predicted'])
test_r2 = r2_score(test_df['target'], test_df['ultra_predicted'])

print(f"\n  Test Set Performance:")
print(f"     RMSE: {test_rmse:.4f}")
print(f"     MAE:  {test_mae:.4f}")
print(f"     R²:   {test_r2:.4f}")

# Risk distribution
risk_counts = results_df['risk_level'].value_counts()
print(f"\n  Risk Distribution:")
for level in ['Low', 'Medium', 'High']:
    if level in risk_counts:
        count = risk_counts[level]
        pct = count / len(results_df) * 100
        print(f"     {level:8s}: {count:8,} ({pct:5.1f}%)")

# Top hotspots
print(f"\n  Top 5 Predicted Hotspots (Test Set):")
top_5 = test_df.nlargest(5, 'ultra_predicted')[['cell_id', 'ultra_predicted', 'target', 'ultra_error']]
for idx, row in top_5.iterrows():
    print(f"     Cell {row['cell_id']:15s}: Pred={row['ultra_predicted']:5.2f}, "
          f"Actual={row['target']:3.0f}, Error={row['ultra_error']:5.2f}")
# FINAL SUMMARY
print("\n" + "=" * 80)
print("✓ PREDICTIONS GENERATED SUCCESSFULLY!")
print("=" * 80)

print(f"""
Output Files Created:
   • outputs/ultra/ultra_predictions.csv ({len(results_df):,} rows)
   • outputs/test_predictions.csv ({len(test_df):,} rows)
Performance Summary:
   • Test RMSE: {test_rmse:.4f}
   • Test MAE:  {test_mae:.4f}
   • Test R²:   {test_r2:.4f}
   • High-risk cells: {(test_df['ultra_predicted'] >= 5).sum()}
""")

print("=" * 80)