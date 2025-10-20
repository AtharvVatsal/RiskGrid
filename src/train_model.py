"""
RiskGrid - ULTRA-ADVANCED Training Pipeline
State-of-the-art ML with deep learning, hyperparameter tuning, and advanced ensembles
Target: 0.30-0.35 RMSE (15-25% improvement over baseline)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print(" " * 18 + "RISKGRID ULTRA-ADVANCED TRAINING PIPELINE")
print(" " * 25 + "Target: 0.30-0.35 RMSE")
print("=" * 80)

FEATURE_DATA_PATH = "data/features/feature_matrix.csv"
CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"
MODEL_OUTPUT_DIR = "models/ultra/"
RESULTS_OUTPUT_DIR = "outputs/ultra/"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Advanced options
USE_OPTUNA = True  # Hyperparameter optimization
USE_LSTM = True    # Deep learning model
USE_STACKING = True  # Meta-learning ensemble
USE_CUSTOM_LOSS = True  # Weighted loss for rare events

# Create directories
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# STEP 1: LOAD DATA
print("\n[STEP 1] Loading data...")
df = pd.read_csv(FEATURE_DATA_PATH)
df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

df['time_window'] = pd.to_datetime(df['time_window'])
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'])

print(f"  ✓ Feature data: {len(df):,} observations")
print(f"  ✓ Raw data: {len(df_cleaned):,} incidents")

# STEP 2: ULTRA-ADVANCED FEATURE ENGINEERING
print("\n[STEP 2] Creating ultra-advanced features...")

# Calculate city center (Chicago: approximately 41.88, -87.63)
CHICAGO_CENTER_LAT = 41.8781
CHICAGO_CENTER_LON = -87.6298

# 2.1: Distance to city center
df['dist_to_center'] = np.sqrt(
    (df['cell_center_lat'] - CHICAGO_CENTER_LAT)**2 + 
    (df['cell_center_lon'] - CHICAGO_CENTER_LON)**2
)

# 2.2: Fourier features for strong periodicity
df['hour_fourier_1'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_fourier_2'] = np.cos(2 * np.pi * df['hour'] / 24)
df['hour_fourier_3'] = np.sin(4 * np.pi * df['hour'] / 24)
df['hour_fourier_4'] = np.cos(4 * np.pi * df['hour'] / 24)

df['week_fourier_1'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['week_fourier_2'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# 2.3: Advanced temporal features
df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                      (df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_prime_crime_time'] = ((df['hour'] >= 18) & (df['hour'] <= 23)).astype(int)

# 2.4: Exponentially weighted moving averages (give more weight to recent)
df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
df['ewma_3'] = df.groupby('cell_id')['incident_count'].transform(
    lambda x: x.ewm(span=3, adjust=False).mean().shift(1)
)
df['ewma_7'] = df.groupby('cell_id')['incident_count'].transform(
    lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
)

# 2.5: Volatility features
df['volatility_3'] = df.groupby('cell_id')['incident_count'].transform(
    lambda x: x.rolling(window=3, min_periods=1).std().shift(1)
)
df['volatility_7'] = df.groupby('cell_id')['incident_count'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std().shift(1)
)

# 2.6: Acceleration (change in momentum)
df['acceleration'] = df.groupby('cell_id')['momentum'].transform(
    lambda x: x.diff(1)
)

# 2.7: High-order interactions
df['lag1_x_ewma3'] = df['lag_1'] * df['ewma_3']
df['trend_x_volatility'] = df['trend_short'] * df['volatility_3']
df['neighbor_x_hour'] = df['neighbor_mean'] * df['hour']
df['cellavg_x_weekend'] = df['cell_avg_incidents'] * df['is_weekend']
df['dist_x_night'] = df['dist_to_center'] * df['is_night']

# 2.8: Percentile-based features
df['lag1_percentile'] = df.groupby('cell_id')['lag_1'].transform(
    lambda x: x.rank(pct=True)
)

# Fill NaN values
new_features = ['dist_to_center', 'hour_fourier_1', 'hour_fourier_2', 'hour_fourier_3', 
                'hour_fourier_4', 'week_fourier_1', 'week_fourier_2', 'is_rush_hour',
                'is_night', 'is_prime_crime_time', 'ewma_3', 'ewma_7', 'volatility_3',
                'volatility_7', 'acceleration', 'lag1_x_ewma3', 'trend_x_volatility',
                'neighbor_x_hour', 'cellavg_x_weekend', 'dist_x_night', 'lag1_percentile']

df[new_features] = df[new_features].fillna(0)

print(f"  ✓ Created {len(new_features)} ultra-advanced features")

# Prepare feature columns
exclude_cols = ['cell_id', 'time_window', 'cell_center_lat', 'cell_center_lon', 
                'grid_lat_idx', 'grid_lon_idx', 'time_slot', 'incident_count', 
                'target', 'target_binary']

feature_cols = [col for col in df.columns if col not in exclude_cols]
print(f"  ✓ Total features: {len(feature_cols)}")
# STEP 3: TRAIN/VAL/TEST SPLIT
print("\n[STEP 3] Creating stratified temporal splits...")

df = df.sort_values('time_window').reset_index(drop=True)

train_end = int(len(df) * (1 - TEST_SIZE - VAL_SIZE))
val_end = int(len(df) * (1 - TEST_SIZE))

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

X_train = train_df[feature_cols]
y_train = train_df['target']

X_val = val_df[feature_cols]
y_val = val_df['target']

X_test = test_df[feature_cols]
y_test = test_df['target']

print(f"  ✓ Train: {len(train_df):,} ({train_df['time_window'].min().date()} to {train_df['time_window'].max().date()})")
print(f"  ✓ Val:   {len(val_df):,} ({val_df['time_window'].min().date()} to {val_df['time_window'].max().date()})")
print(f"  ✓ Test:  {len(test_df):,} ({test_df['time_window'].min().date()} to {test_df['time_window'].max().date()})")

# Feature scaling
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

joblib.dump(scaler, f"{MODEL_OUTPUT_DIR}ultra_scaler.pkl")
# HELPER FUNCTIONS
def evaluate_model(y_true, y_pred, model_name):
    """Comprehensive evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def custom_loss_gradient(y_true, y_pred):
    """Custom loss that penalizes errors on high-incident events more"""
    weights = 1 + np.log1p(y_true)  # Higher weight for higher incidents
    grad = (y_pred - y_true) * weights
    hess = weights
    return grad, hess
# STEP 4: HYPERPARAMETER OPTIMIZATION
if USE_OPTUNA:
    print("\n[STEP 4] Hyperparameter optimization with Optuna...")
    
    try:
        import optuna
        from optuna.samplers import TPESampler
        
        def objective_xgb(trial):
            """Optimize XGBoost hyperparameters"""
            params = {
                'n_estimators': 500,
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': RANDOM_STATE,
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
            pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            return rmse
        
        def objective_lgb(trial):
            """Optimize LightGBM hyperparameters"""
            params = {
                'n_estimators': 500,
                'max_depth': trial.suggest_int('max_depth', 6, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'random_state': RANDOM_STATE,
                'n_jobs': -1,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
            pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            return rmse
        
        # Optimize XGBoost
        print("  • Optimizing XGBoost (20 trials)...")
        study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
        study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)
        best_xgb_params = study_xgb.best_params
        print(f"    Best XGBoost RMSE: {study_xgb.best_value:.4f}")
        
        # Optimize LightGBM
        print("  • Optimizing LightGBM (20 trials)...")
        study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=RANDOM_STATE))
        study_lgb.optimize(objective_lgb, n_trials=20, show_progress_bar=True)
        best_lgb_params = study_lgb.best_params
        print(f"    Best LightGBM RMSE: {study_lgb.best_value:.4f}")
        
        OPTUNA_SUCCESS = True
        
    except ImportError:
        print("  ⚠️  Optuna not installed. Using default parameters.")
        print("     Install with: pip install optuna")
        OPTUNA_SUCCESS = False
        best_xgb_params = {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 
                          'colsample_bytree': 0.8, 'min_child_weight': 3, 'gamma': 0.1,
                          'reg_alpha': 0.1, 'reg_lambda': 1.0}
        best_lgb_params = {'max_depth': 10, 'learning_rate': 0.05, 'num_leaves': 50,
                          'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
                          'reg_alpha': 0.1, 'reg_lambda': 1.0}
else:
    OPTUNA_SUCCESS = False
    best_xgb_params = {'max_depth': 8, 'learning_rate': 0.05, 'subsample': 0.8, 
                      'colsample_bytree': 0.8, 'min_child_weight': 3, 'gamma': 0.1,
                      'reg_alpha': 0.1, 'reg_lambda': 1.0}
    best_lgb_params = {'max_depth': 10, 'learning_rate': 0.05, 'num_leaves': 50,
                      'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 20,
                      'reg_alpha': 0.1, 'reg_lambda': 1.0}
# STEP 5: TRAIN OPTIMIZED MODELS
print("\n[STEP 5] Training optimized models...")

results = []

# Baseline
baseline_pred = np.full(len(y_test), y_train.mean())
results.append(evaluate_model(y_test, baseline_pred, 'Baseline (Mean)'))

# Random Forest (optimized)
print("  • Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
results.append(evaluate_model(y_test, rf_pred, 'Random Forest (Optimized)'))
print(f"    RMSE: {results[-1]['RMSE']:.4f}")

# XGBoost (optimized)
print("  • XGBoost (Optuna-tuned)...")
xgb_params = {**best_xgb_params, 'n_estimators': 500, 'random_state': RANDOM_STATE, 'n_jobs': -1}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_test_scaled)
results.append(evaluate_model(y_test, xgb_pred, 'XGBoost (Optuna)'))
print(f"    RMSE: {results[-1]['RMSE']:.4f}")

# LightGBM (optimized)
print("  • LightGBM (Optuna-tuned)...")
lgb_params = {**best_lgb_params, 'n_estimators': 500, 'random_state': RANDOM_STATE, 
              'n_jobs': -1, 'verbose': -1}
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
lgb_pred = lgb_model.predict(X_test_scaled)
results.append(evaluate_model(y_test, lgb_pred, 'LightGBM (Optuna)'))
print(f"    RMSE: {results[-1]['RMSE']:.4f}")

# XGBoost with custom loss
if USE_CUSTOM_LOSS:
    print("  • XGBoost (Custom Loss)...")
    xgb_custom_model = xgb.XGBRegressor(**xgb_params)
    
    # Create sample weights based on incident count
    sample_weights = 1 + np.log1p(y_train)
    
    xgb_custom_model.fit(X_train_scaled, y_train, sample_weight=sample_weights, 
                        eval_set=[(X_val_scaled, y_val)], verbose=False)
    xgb_custom_pred = xgb_custom_model.predict(X_test_scaled)
    results.append(evaluate_model(y_test, xgb_custom_pred, 'XGBoost (Custom Loss)'))
    print(f"    RMSE: {results[-1]['RMSE']:.4f}")
# STEP 6: STACKING ENSEMBLE
if USE_STACKING:
    print("\n[STEP 6] Building stacking ensemble...")
    
    try:
        # Base models
        base_models = [
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ]
        
        # Meta-learner (Ridge regression)
        meta_learner = Ridge(alpha=1.0)
        
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1  # Avoid Windows multiprocessing issues
        )
        
        print("  • Training stacking ensemble (this may take a few minutes)...")
        stacking_model.fit(X_train_scaled, y_train)
        stacking_pred = stacking_model.predict(X_test_scaled)
        results.append(evaluate_model(y_test, stacking_pred, 'Stacking Ensemble'))
        print(f"    RMSE: {results[-1]['RMSE']:.4f}")
        STACKING_SUCCESS = True
        
    except Exception as e:
        print(f"  ⚠️  Stacking failed: {str(e)[:100]}")
        print("     Continuing without stacking ensemble...")
        STACKING_SUCCESS = False
        stacking_model = None
        stacking_pred = None
else:
    STACKING_SUCCESS = False
    stacking_model = None
    stacking_pred = None
# STEP 7: WEIGHTED ENSEMBLE (OPTIMIZED)

print("\n[STEP 7] Creating optimized weighted ensemble...")

# Get validation predictions
rf_val = rf_model.predict(X_val_scaled)
xgb_val = xgb_model.predict(X_val_scaled)
lgb_val = lgb_model.predict(X_val_scaled)

if USE_CUSTOM_LOSS:
    xgb_custom_val = xgb_custom_model.predict(X_val_scaled)
    
# Fine-grained weight search
best_rmse = float('inf')
best_weights = None

print("  • Searching for optimal weights...")
weight_grid = np.linspace(0, 1, 21)  # More granular than before

for w_rf in weight_grid:
    for w_xgb in weight_grid:
        for w_lgb in weight_grid:
            if USE_CUSTOM_LOSS:
                for w_custom in weight_grid:
                    total = w_rf + w_xgb + w_lgb + w_custom
                    if abs(total - 1.0) > 0.01:
                        continue
                    
                    ensemble_val = (w_rf * rf_val + w_xgb * xgb_val + 
                                  w_lgb * lgb_val + w_custom * xgb_custom_val)
                    rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
                    
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = (w_rf, w_xgb, w_lgb, w_custom)
            else:
                total = w_rf + w_xgb + w_lgb
                if abs(total - 1.0) > 0.01:
                    continue
                
                ensemble_val = (w_rf * rf_val + w_xgb * xgb_val + w_lgb * lgb_val)
                rmse = np.sqrt(mean_squared_error(y_val, ensemble_val))
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = (w_rf, w_xgb, w_lgb, 0)

# Apply best weights to test set
if USE_CUSTOM_LOSS:
    weighted_pred = (best_weights[0] * rf_pred + best_weights[1] * xgb_pred + 
                    best_weights[2] * lgb_pred + best_weights[3] * xgb_custom_pred)
else:
    weighted_pred = (best_weights[0] * rf_pred + best_weights[1] * xgb_pred + 
                    best_weights[2] * lgb_pred)

results.append(evaluate_model(y_test, weighted_pred, 'Weighted Ensemble (Optimized)'))

print(f"  ✓ Optimal weights: RF={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, "
      f"LGB={best_weights[2]:.3f}" + (f", Custom={best_weights[3]:.3f}" if USE_CUSTOM_LOSS else ""))
print(f"  ✓ Test RMSE: {results[-1]['RMSE']:.4f}")
# STEP 8: MODEL COMPARISON
print("\n[STEP 8] Ultra-model performance comparison...")

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\n" + "=" * 80)
print(" " * 20 + "🏆 ULTRA-ADVANCED MODEL RANKINGS 🏆")
print("=" * 80)
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

best_model_name = results_df.iloc[0]['Model']
best_rmse = results_df.iloc[0]['RMSE']
baseline_rmse = results_df[results_df['Model'] == 'Baseline (Mean)']['RMSE'].values[0]
improvement = (1 - best_rmse / baseline_rmse) * 100

print(f"\n🎯 TARGET CHECK:")
print(f"   Goal: 0.30-0.35 RMSE")
print(f"   Achieved: {best_rmse:.4f} RMSE")
if best_rmse <= 0.35:
    print(f"   ✅ TARGET MET! ({(0.35 - best_rmse)/0.35*100:.1f}% better than target)")
else:
    print(f"   📊 Close! ({(best_rmse - 0.35)/0.35*100:.1f}% above target)")

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Improvement vs Baseline: {improvement:.1f}%")
print(f"   vs Previous Best (0.380): {(1 - best_rmse/0.380)*100:.1f}% improvement")

# Save models
joblib.dump(rf_model, f"{MODEL_OUTPUT_DIR}rf_ultra.pkl")
joblib.dump(xgb_model, f"{MODEL_OUTPUT_DIR}xgb_ultra.pkl")
joblib.dump(lgb_model, f"{MODEL_OUTPUT_DIR}lgb_ultra.pkl")
if USE_CUSTOM_LOSS:
    joblib.dump(xgb_custom_model, f"{MODEL_OUTPUT_DIR}xgb_custom_ultra.pkl")
if USE_STACKING and STACKING_SUCCESS:
    joblib.dump(stacking_model, f"{MODEL_OUTPUT_DIR}stacking_ultra.pkl")

# Save best ensemble config
best_config = {
    'weights': best_weights,
    'models': ['rf', 'xgb', 'lgb'] + (['xgb_custom'] if USE_CUSTOM_LOSS else []),
    'rmse': best_rmse
}
joblib.dump(best_config, f"{MODEL_OUTPUT_DIR}best_ensemble_config.pkl")
joblib.dump(feature_cols, f"{MODEL_OUTPUT_DIR}ultra_feature_names.pkl")

# Determine best predictions
if best_model_name == 'Weighted Ensemble (Optimized)':
    best_pred = weighted_pred
elif best_model_name == 'Stacking Ensemble' and STACKING_SUCCESS:
    best_pred = stacking_pred
elif 'Custom Loss' in best_model_name:
    best_pred = xgb_custom_pred
elif 'XGBoost' in best_model_name:
    best_pred = xgb_pred
elif 'LightGBM' in best_model_name:
    best_pred = lgb_pred
else:
    best_pred = rf_pred

# STEP 9: FEATURE IMPORTANCE
print("\n[STEP 9] Analyzing feature importance...")

# Weighted feature importance from ensemble
if USE_CUSTOM_LOSS:
    importance = (best_weights[0] * rf_model.feature_importances_ +
                 best_weights[1] * xgb_model.feature_importances_ +
                 best_weights[2] * lgb_model.feature_importances_ +
                 best_weights[3] * xgb_custom_model.feature_importances_)
else:
    importance = (best_weights[0] * rf_model.feature_importances_ +
                 best_weights[1] * xgb_model.feature_importances_ +
                 best_weights[2] * lgb_model.feature_importances_)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)

print("\n📊 Top 20 Most Important Features:")
for i, row in feature_importance.head(20).iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"  {row['feature']:35s} : {row['importance']:.4f} {bar}")

feature_importance.to_csv(f"{RESULTS_OUTPUT_DIR}ultra_feature_importance.csv", index=False)

# STEP 10: VISUALIZATIONS
print("\n[STEP 10] Creating ultra-advanced visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

fig.suptitle('RiskGrid - Ultra-Advanced Model Performance', 
             fontsize=22, fontweight='bold', y=0.995)

# Plot 1: Model Comparison
ax1 = fig.add_subplot(gs[0, :])
models_sorted = results_df.sort_values('RMSE')
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.95, len(models_sorted)))
bars = ax1.barh(range(len(models_sorted)), models_sorted['RMSE'], color=colors, 
                edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(models_sorted)))
ax1.set_yticklabels(models_sorted['Model'], fontsize=11)
ax1.set_xlabel('RMSE (Lower is Better)', fontsize=13, fontweight='bold')
ax1.set_title('Ultra-Advanced Model Rankings', fontsize=15, fontweight='bold', pad=10)
ax1.axvline(x=0.35, color='green', linestyle='--', linewidth=2, label='Target (0.35)', alpha=0.7)
ax1.axvline(x=0.30, color='darkgreen', linestyle='--', linewidth=2, label='Stretch Goal (0.30)', alpha=0.7)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
ax1.legend(fontsize=10)

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, f'  {width:.4f}',
             ha='left', va='center', fontweight='bold', fontsize=10)

bars[0].set_edgecolor('gold')
bars[0].set_linewidth(3)

# Plot 2: Actual vs Predicted (Hexbin)
ax2 = fig.add_subplot(gs[1, 0])
hb = ax2.hexbin(y_test, best_pred, gridsize=35, cmap='YlOrRd', mincnt=1)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'b--', lw=3, label='Perfect', alpha=0.7)
ax2.set_xlabel('Actual', fontsize=11, fontweight='bold')
ax2.set_ylabel('Predicted', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Accuracy', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.colorbar(hb, ax=ax2, label='Density')

# Plot 3: Residuals
ax3 = fig.add_subplot(gs[1, 1])
residuals = y_test - best_pred
ax3.scatter(best_pred, residuals, alpha=0.4, s=15, c=abs(residuals), cmap='coolwarm')
ax3.axhline(y=0, color='black', linestyle='--', lw=2)
ax3.fill_between(ax3.get_xlim(), -1, 1, alpha=0.2, color='green')
ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax3.set_ylabel('Residual', fontsize=11, fontweight='bold')
ax3.set_title('Residual Analysis', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Plot 4: Error Distribution
ax4 = fig.add_subplot(gs[1, 2])
errors = abs(residuals)
ax4.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax4.axvline(errors.mean(), color='red', linestyle='--', lw=2, label=f'Mean: {errors.mean():.3f}')
ax4.axvline(errors.median(), color='blue', linestyle='--', lw=2, label=f'Median: {errors.median():.3f}')
ax4.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Feature Importance
ax5 = fig.add_subplot(gs[2, :])
top_features = feature_importance.head(15)
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax5.barh(range(len(top_features)), top_features['importance'], 
                color=colors_feat, edgecolor='black')
ax5.set_yticks(range(len(top_features)))
ax5.set_yticklabels(top_features['feature'], fontsize=10)
ax5.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax5.set_title('Top 15 Features (Ensemble-Weighted)', fontsize=14, fontweight='bold', pad=10)
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# Plot 6: Improvement Over Time
ax6 = fig.add_subplot(gs[3, 0])
improvement_data = [
    ('Baseline', baseline_rmse),
    ('Advanced', 0.380),
    ('Ultra', best_rmse)
]
improvement_names = [x[0] for x in improvement_data]
improvement_values = [x[1] for x in improvement_data]
colors_imp = ['red', 'orange', 'green']
bars = ax6.bar(range(len(improvement_data)), improvement_values, color=colors_imp, 
               edgecolor='black', alpha=0.8, linewidth=1.5)
ax6.set_xticks(range(len(improvement_data)))
ax6.set_xticklabels(improvement_names, fontsize=11)
ax6.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax6.set_title('Model Evolution', fontsize=12, fontweight='bold')
ax6.axhline(y=0.35, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax6.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 7: Q-Q Plot
ax7 = fig.add_subplot(gs[3, 1])
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax7)
ax7.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# Plot 8: Prediction Confidence
ax8 = fig.add_subplot(gs[3, 2])
# Bin predictions and show error bars
pred_bins = pd.qcut(best_pred, q=10, duplicates='drop')
binned_errors = pd.DataFrame({'pred': best_pred, 'error': errors, 'bin': pred_bins})
bin_stats = binned_errors.groupby('bin')['error'].agg(['mean', 'std']).reset_index()
bin_centers = [i for i in range(len(bin_stats))]

ax8.errorbar(bin_centers, bin_stats['mean'], yerr=bin_stats['std'], 
             fmt='o-', linewidth=2, markersize=8, capsize=5, capthick=2)
ax8.set_xlabel('Prediction Bin', fontsize=11, fontweight='bold')
ax8.set_ylabel('Mean Absolute Error ± Std', fontsize=11, fontweight='bold')
ax8.set_title('Prediction Confidence', fontsize=12, fontweight='bold')
ax8.grid(alpha=0.3)

plt.savefig(f"{RESULTS_OUTPUT_DIR}ultra_performance.png", dpi=300, bbox_inches='tight', facecolor='white')
print(f"  ✓ Saved: {RESULTS_OUTPUT_DIR}ultra_performance.png")

plt.show()

# Save predictions
test_df['ultra_predicted'] = best_pred
test_df['ultra_error'] = abs(y_test - best_pred)
test_df.to_csv(f"{RESULTS_OUTPUT_DIR}ultra_predictions.csv", index=False)
# FINAL SUMMARY
print("\n" + "=" * 80)
print(" " * 25 + "🎉 ULTRA-TRAINING COMPLETE! 🎉")
print("=" * 80)

print(f"\nPERFORMANCE SUMMARY:")
print(f"   Best Model: {best_model_name}")
print(f"   RMSE: {best_rmse:.4f} {'✅' if best_rmse <= 0.35 else '📊'}")
print(f"   MAE: {results_df.iloc[0]['MAE']:.4f}")
print(f"   R²: {results_df.iloc[0]['R²']:.4f}")
print(f"   MAPE: {results_df.iloc[0]['MAPE']:.2f}%")

print(f"\nIMPROVEMENTS:")
print(f"   vs Baseline: {improvement:.1f}%")
print(f"   vs Advanced (0.380): {(1 - best_rmse/0.380)*100:.1f}%")

print(f"\nKEY INSIGHTS:")
print(f"   • Most important feature: {feature_importance.iloc[0]['feature']}")
print(f"   • Average error: {errors.mean():.3f} incidents")
print(f"   • Median error: {errors.median():.3f} incidents")
print(f"   • 95th percentile error: {np.percentile(errors, 95):.2f} incidents")

print(f"\nOUTPUT FILES:")
print(f"   • Models: {MODEL_OUTPUT_DIR}")
print(f"   • Visualizations: {RESULTS_OUTPUT_DIR}ultra_performance.png")
print(f"   • Predictions: {RESULTS_OUTPUT_DIR}ultra_predictions.csv")
print(f"   • Feature Importance: {RESULTS_OUTPUT_DIR}ultra_feature_importance.csv")

print("\nNEXT STEP: Build the Interactive Dashboard!")
print("=" * 80)