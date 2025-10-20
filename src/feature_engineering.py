"""
RiskGrid - Phase 3: Feature Engineering
Creates predictive features from grid-time aggregated data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


GRID_DATA_PATH = "A:/Work/RiskGrid/data/processed/grid_aggregated.csv"
CLEANED_DATA_PATH = "A:/Work/RiskGrid/data/processed/cleaned_data.csv"
FEATURE_OUTPUT_PATH = "A:/Work/RiskGrid/data/features/feature_matrix.csv"

# Feature engineering parameters
HISTORICAL_WINDOWS = [1, 3, 7, 14]  # Look back 1, 3, 7, 14 time periods
SPATIAL_NEIGHBOR_RADIUS = 1
print("=" * 80)
print(" " * 25 + "RISKGRID FEATURE ENGINEERING")
print("=" * 80)
print("\n[STEP 1] Loading processed data...")
df = pd.read_csv(GRID_DATA_PATH)
df['time_window'] = pd.to_datetime(df['time_window'])
df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)

print(f" Loaded {len(df):,} grid-time observations")
print(f" Unique cells: {df['cell_id'].nunique():,}")
print(f" Time range: {df['time_window'].min()} to {df['time_window'].max()}")


def create_temporal_features(df):
    """Create time-based features"""
    print("\n[STEP 2] Creating temporal features...")
    
    # Basic temporal features (already exist, but ensure they're present)
    df['hour'] = df['hour'].fillna(12)  # Fill any missing
    df['day_of_week'] = df['day_of_week'].fillna(0).astype(int)
    df['is_weekend'] = df['is_weekend'].fillna(0).astype(int)
    
    # Additional temporal features
    df['month'] = df['time_window'].dt.month
    df['day_of_month'] = df['time_window'].dt.day
    df['week_of_year'] = df['time_window'].dt.isocalendar().week
    
    # Cyclical encoding for time (important for ML models)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    print(f"  ✓ Created temporal features")
    
    return df
# HISTORICAL/LAG FEATURES
def create_lag_features(df, windows=HISTORICAL_WINDOWS):
    """Create historical features - incidents in past time periods"""
    print("\n[STEP 3] Creating lag features...")
    
    # Sort by cell and time
    df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
    
    for window in windows:
        print(f"  • Processing {window}-period lag...")
        
        # Lag feature: incidents N periods ago
        df[f'lag_{window}'] = df.groupby('cell_id')['incident_count'].shift(window)
        
        # Rolling mean: average incidents over past N periods
        df[f'rolling_mean_{window}'] = (
            df.groupby('cell_id')['incident_count']
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean().shift(1))
        )
        
        # Rolling max: maximum incidents in past N periods
        df[f'rolling_max_{window}'] = (
            df.groupby('cell_id')['incident_count']
            .transform(lambda x: x.rolling(window=window, min_periods=1).max().shift(1))
        )
        
        # Rolling std: variability in past N periods
        df[f'rolling_std_{window}'] = (
            df.groupby('cell_id')['incident_count']
            .transform(lambda x: x.rolling(window=window, min_periods=1).std().shift(1))
        )
    
    # Fill NaN values with 0 for early periods
    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]
    df[lag_cols] = df[lag_cols].fillna(0)
    
    print(f"  ✓ Created {len(lag_cols)} lag features")
    
    return df
# SPATIAL FEATURES
def create_spatial_features(df, radius=SPATIAL_NEIGHBOR_RADIUS):
    """Create features based on neighboring cells"""
    print("\n[STEP 4] Creating spatial neighborhood features...")
    
    # For each time window, calculate neighbor activity
    spatial_features = []
    
    for time_window in df['time_window'].unique():
        df_time = df[df['time_window'] == time_window].copy()
        
        # For each cell, find neighbors
        df_time['neighbor_sum'] = 0
        df_time['neighbor_mean'] = 0
        df_time['neighbor_max'] = 0
        
        for idx, row in df_time.iterrows():
            # Find cells within radius
            neighbors = df_time[
                (abs(df_time['grid_lat_idx'] - row['grid_lat_idx']) <= radius) &
                (abs(df_time['grid_lon_idx'] - row['grid_lon_idx']) <= radius) &
                (df_time['cell_id'] != row['cell_id'])  # Exclude self
            ]
            
            if len(neighbors) > 0:
                df_time.at[idx, 'neighbor_sum'] = neighbors['incident_count'].sum()
                df_time.at[idx, 'neighbor_mean'] = neighbors['incident_count'].mean()
                df_time.at[idx, 'neighbor_max'] = neighbors['incident_count'].max()
        
        spatial_features.append(df_time[['cell_id', 'time_window', 'neighbor_sum', 'neighbor_mean', 'neighbor_max']])
    
    # Combine all time windows
    spatial_df = pd.concat(spatial_features, ignore_index=True)
    
    # Merge back to main dataframe
    df = df.merge(spatial_df, on=['cell_id', 'time_window'], how='left')
    df[['neighbor_sum', 'neighbor_mean', 'neighbor_max']] = df[['neighbor_sum', 'neighbor_mean', 'neighbor_max']].fillna(0)
    
    print(f"  ✓ Created spatial neighborhood features")
    
    return df
# CELL-SPECIFIC STATISTICS
def create_cell_statistics(df):
    """Create per-cell aggregate statistics"""
    print("\n[STEP 5] Creating cell-level statistics...")
    
    # Calculate overall statistics per cell
    cell_stats = df.groupby('cell_id').agg({
        'incident_count': ['mean', 'std', 'max', 'sum']
    }).reset_index()
    
    cell_stats.columns = ['cell_id', 'cell_avg_incidents', 'cell_std_incidents', 
                          'cell_max_incidents', 'cell_total_incidents']
    
    # Fill NaN std with 0
    cell_stats['cell_std_incidents'] = cell_stats['cell_std_incidents'].fillna(0)
    
    # Merge back to main dataframe
    df = df.merge(cell_stats, on='cell_id', how='left')
    
    print(f"  ✓ Created cell-level statistics")
    
    return df
# TIME-SLOT SPECIFIC FEATURES
def create_timeslot_features(df):
    """Create features based on day of week and hour patterns"""
    print("\n[STEP 6] Creating time-slot specific features...")
    
    # Average incidents for this day of week
    dow_stats = df.groupby('day_of_week')['incident_count'].mean().to_dict()
    df['dow_avg_incidents'] = df['day_of_week'].map(dow_stats)
    
    # Average incidents for this hour
    hour_stats = df.groupby('hour')['incident_count'].mean().to_dict()
    df['hour_avg_incidents'] = df['hour'].map(hour_stats)
    
    # Average incidents for weekend vs weekday
    weekend_stats = df.groupby('is_weekend')['incident_count'].mean().to_dict()
    df['weekend_avg_incidents'] = df['is_weekend'].map(weekend_stats)
    
    print(f"  ✓ Created time-slot features")
    
    return df
# TREND FEATURES
def create_trend_features(df):
    """Create trend indicators"""
    print("\n[STEP 7] Creating trend features...")
    
    df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
    
    # Recent trend: difference between recent and older history
    df['trend_short'] = df['rolling_mean_3'] - df['rolling_mean_7']
    df['trend_long'] = df['rolling_mean_7'] - df['rolling_mean_14']
    
    # Momentum: rate of change
    df['momentum'] = df.groupby('cell_id')['incident_count'].diff(1)
    
    # Fill NaN with 0
    df[['trend_short', 'trend_long', 'momentum']] = df[['trend_short', 'trend_long', 'momentum']].fillna(0)
    
    print(f"  ✓ Created trend features")
    
    return df
# CREATE TARGET VARIABLE

def create_target(df):
    """Create target variable - incidents in NEXT time period"""
    print("\n[STEP 8] Creating target variable...")
    
    df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
    # Target: incident count in the next time period
    df['target'] = df.groupby('cell_id')['incident_count'].shift(-1)
    # For the last time period of each cell, target is NaN (no future data)
    # We'll remove these rows later for training
    # Also create binary target: will there be ANY incidents?
    df['target_binary'] = (df['target'] > 0).astype(int)
    
    print(f"  ✓ Created target variables")
    print(f"  • Regression target: 'target' (incident count)")
    print(f"  • Classification target: 'target_binary' (any incident yes/no)")
    
    return df
# MAIN PIPELINE
def run_feature_engineering():
    """Run complete feature engineering pipeline"""
    
    print("\nStarting feature engineering pipeline...\n")
    
    # Load data
    df = pd.read_csv(GRID_DATA_PATH)
    df['time_window'] = pd.to_datetime(df['time_window'])
    df = df.sort_values(['cell_id', 'time_window']).reset_index(drop=True)
    
    print(f"Initial data: {len(df):,} observations")
    
    # Create features
    df = create_temporal_features(df)
    df = create_lag_features(df)
    df = create_spatial_features(df)
    df = create_cell_statistics(df)
    df = create_timeslot_features(df)
    df = create_trend_features(df)
    df = create_target(df)
    
    # Remove rows without target (last time period for each cell)
    initial_rows = len(df)
    df = df.dropna(subset=['target'])
    removed = initial_rows - len(df)
    
    print(f"\n[STEP 9] Data preparation...")
    print(f"  • Removed {removed:,} rows without target (future periods)")
    print(f"  • Final dataset: {len(df):,} observations")
    
    # Save feature matrix
    import os
    os.makedirs("data/features", exist_ok=True)
    df.to_csv(FEATURE_OUTPUT_PATH, index=False)
    
    print(f"\n  ✓ Saved feature matrix: {FEATURE_OUTPUT_PATH}")
    
    # Print feature summary
    feature_cols = [col for col in df.columns if col not in 
                   ['cell_id', 'time_window', 'cell_center_lat', 'cell_center_lon', 
                    'grid_lat_idx', 'grid_lon_idx', 'time_slot', 'incident_count', 'target', 'target_binary']]
    
    print(f"\n" + "=" * 80)
    print("✓ FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    
    print(f"\nFeature Summary:")
    print(f"  • Total observations: {len(df):,}")
    print(f"  • Total features: {len(feature_cols)}")
    print(f"  • Target variable: 'target' (continuous) and 'target_binary' (classification)")
    
    print(f"\nFeature Categories:")
    temporal = [c for c in feature_cols if any(x in c for x in ['hour', 'day', 'month', 'week', 'weekend'])]
    lag = [c for c in feature_cols if any(x in c for x in ['lag', 'rolling'])]
    spatial = [c for c in feature_cols if 'neighbor' in c]
    cell = [c for c in feature_cols if 'cell_' in c]
    trend = [c for c in feature_cols if any(x in c for x in ['trend', 'momentum'])]
    
    print(f"  • Temporal features: {len(temporal)}")
    print(f"  • Historical/Lag features: {len(lag)}")
    print(f"  • Spatial features: {len(spatial)}")
    print(f"  • Cell statistics: {len(cell)}")
    print(f"  • Trend features: {len(trend)}")
    
    # Show sample
    print(f"\nSample features:")
    print(df[feature_cols[:10]].head(3))
    
    print(f"\n Target distribution:")
    print(f"  • Mean incidents: {df['target'].mean():.2f}")
    print(f"  • Median incidents: {df['target'].median():.0f}")
    print(f"  • Max incidents: {df['target'].max():.0f}")
    print(f"  • % with incidents: {(df['target'] > 0).mean() * 100:.1f}%")
    
    print("\n🚀 Ready for Phase 4: Model Training!")
    
    return df

# RUN PIPELINE
if __name__ == "__main__":
    df_features = run_feature_engineering()