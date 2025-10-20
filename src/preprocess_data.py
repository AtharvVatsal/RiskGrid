"""
RiskGrid - Phase 2: Data Preprocessing & Grid Generation
Converts raw crime data into grid-based spatial-temporal features
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from tqdm import tqdm
# Data paths
RAW_DATA_PATH = r"A:\Work\RiskGrid\data\raw\chicago_crimes.csv"
PROCESSED_OUTPUT = "A:/Work/RiskGrid/data/processed/cleaned_data.csv"
GRID_OUTPUT = "A:/Work/RiskGrid/data/processed/grid_aggregated.csv"

# Column names from diagnostic
CONFIG = {
    'date_column': 'Date',
    'latitude_column': 'Latitude',
    'longitude_column': 'Longitude',
    'crime_type_column': 'Primary Type',
}

# Grid parameters
GRID_SIZE_METERS = 500  # Each cell is 500m x 500m
CELL_SIZE_DEGREES = GRID_SIZE_METERS / 111000  # Approximate: 1 degree ≈ 111km

# Temporal parameters
TIME_WINDOW_HOURS = 6  # Aggregate into 6-hour windows

# Geographic bounds (Chicago area - will auto-detect from data)
# These will be set automatically based on your data

print("=" * 80)
print(" " * 25 + "RISKGRID PREPROCESSING PIPELINE")
print("=" * 80)
# STEP 1: LOAD AND CLEAN DATA
def load_and_clean_data(filepath, sample_size=None):
    """Load raw data and perform initial cleaning"""
    print(f"\n[STEP 1] Loading data from: {filepath}")
    
    # Load data
    df = pd.read_csv(filepath, nrows=sample_size, low_memory=False)
    print(f"  ✓ Loaded {len(df):,} records")
    
    # Extract key columns
    date_col = CONFIG['date_column']
    lat_col = CONFIG['latitude_column']
    lon_col = CONFIG['longitude_column']
    crime_col = CONFIG['crime_type_column']
    
    # Keep only necessary columns
    required_cols = [date_col, lat_col, lon_col, crime_col]
    
    # Add optional useful columns if they exist
    optional_cols = ['Arrest', 'Domestic', 'District', 'Beat', 'Description']
    for col in optional_cols:
        if col in df.columns:
            required_cols.append(col)
    
    df = df[required_cols].copy()
    
    print(f"\n[STEP 2] Cleaning data...")
    
    # Parse dates
    print(f"  • Parsing dates...")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Remove records with missing critical data
    initial_count = len(df)
    df = df.dropna(subset=[date_col, lat_col, lon_col, crime_col])
    removed = initial_count - len(df)
    print(f"  • Removed {removed:,} records with missing critical data")
    
    # Remove outlier coordinates (outside Chicago bounds)
    print(f"  • Filtering coordinates...")
    # Chicago bounds: roughly 41.6 to 42.1 lat, -87.95 to -87.5 lon
    df = df[
        (df[lat_col] >= 41.6) & (df[lat_col] <= 42.1) &
        (df[lon_col] >= -88.0) & (df[lon_col] <= -87.5)
    ]
    print(f"  • Retained {len(df):,} records within Chicago area")
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    print(f"  ✓ Final cleaned dataset: {len(df):,} records")
    print(f"  ✓ Date range: {df[date_col].min()} to {df[date_col].max()}")
    
    return df
# STEP 2: GRID GENERATION
def create_spatial_grid(df, cell_size_degrees=CELL_SIZE_DEGREES):
    """Convert lat/lon coordinates to grid cell IDs"""
    print(f"\n[STEP 3] Creating spatial grid...")
    
    lat_col = CONFIG['latitude_column']
    lon_col = CONFIG['longitude_column']
    
    # Determine grid bounds
    lat_min = df[lat_col].min()
    lat_max = df[lat_col].max()
    lon_min = df[lon_col].min()
    lon_max = df[lon_col].max()
    
    print(f"  • Geographic bounds:")
    print(f"    Latitude : {lat_min:.4f} to {lat_max:.4f}")
    print(f"    Longitude: {lon_min:.4f} to {lon_max:.4f}")
    
    # Calculate grid dimensions
    n_lat_cells = int(np.ceil((lat_max - lat_min) / cell_size_degrees))
    n_lon_cells = int(np.ceil((lon_max - lon_min) / cell_size_degrees))
    
    print(f"  • Grid size: {GRID_SIZE_METERS}m x {GRID_SIZE_METERS}m per cell")
    print(f"  • Grid dimensions: {n_lat_cells} x {n_lon_cells} = {n_lat_cells * n_lon_cells:,} total cells")
    
    # Assign each incident to a grid cell
    df['grid_lat_idx'] = ((df[lat_col] - lat_min) / cell_size_degrees).astype(int)
    df['grid_lon_idx'] = ((df[lon_col] - lon_min) / cell_size_degrees).astype(int)
    
    # Create unique cell ID
    df['cell_id'] = df['grid_lat_idx'].astype(str) + '_' + df['grid_lon_idx'].astype(str)
    
    # Calculate cell center coordinates (for visualization later)
    df['cell_center_lat'] = lat_min + (df['grid_lat_idx'] + 0.5) * cell_size_degrees
    df['cell_center_lon'] = lon_min + (df['grid_lon_idx'] + 0.5) * cell_size_degrees
    
    unique_cells = df['cell_id'].nunique()
    print(f"  ✓ Mapped {len(df):,} incidents to {unique_cells:,} unique grid cells")
    
    # Store grid metadata
    grid_metadata = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'lon_min': lon_min,
        'lon_max': lon_max,
        'cell_size_degrees': cell_size_degrees,
        'cell_size_meters': GRID_SIZE_METERS,
        'n_lat_cells': n_lat_cells,
        'n_lon_cells': n_lon_cells,
        'total_cells': n_lat_cells * n_lon_cells
    }
    
    return df, grid_metadata
# STEP 3: TEMPORAL AGGREGATION
def create_time_windows(df, window_hours=TIME_WINDOW_HOURS):
    """Discretize time into fixed windows"""
    print(f"\n[STEP 4] Creating temporal windows...")
    
    date_col = CONFIG['date_column']
    
    # Round datetime to nearest time window
    df['timestamp'] = df[date_col]
    df['time_window'] = df['timestamp'].dt.floor(f'{window_hours}H')
    
    # Extract temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create time slot categories
    df['time_slot'] = pd.cut(
        df['hour'], 
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        include_lowest=True
    )
    
    n_windows = df['time_window'].nunique()
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    
    print(f"  • Time window size: {window_hours} hours")
    print(f"  • Total time windows: {n_windows:,}")
    print(f"  • Date range: {date_range} days")
    print(f"  ✓ Temporal features created")
    
    return df
# STEP 4: AGGREGATE TO GRID-TIME CELLS
def aggregate_to_grid_time(df):
    """Aggregate incidents by grid cell and time window"""
    print(f"\n[STEP 5] Aggregating to grid-time cells...")
    
    crime_col = CONFIG['crime_type_column']
    
    # Group by cell_id and time_window
    agg_df = df.groupby(['cell_id', 'time_window']).agg({
        'grid_lat_idx': 'first',
        'grid_lon_idx': 'first',
        'cell_center_lat': 'first',
        'cell_center_lon': 'first',
        crime_col: 'count',  # Count incidents
        'hour': 'mean',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'time_slot': lambda x: x.mode()[0] if len(x) > 0 else None
    }).reset_index()
    
    # Rename incident count column
    agg_df.rename(columns={crime_col: 'incident_count'}, inplace=True)
    
    # Add additional features
    agg_df = agg_df.sort_values('time_window').reset_index(drop=True)
    
    print(f"  ✓ Created {len(agg_df):,} grid-time observations")
    print(f"  ✓ Average incidents per cell-time: {agg_df['incident_count'].mean():.2f}")
    print(f"  ✓ Max incidents in a cell-time: {agg_df['incident_count'].max()}")
    
    return agg_df
# STEP 5: SAVE PROCESSED DATA
def save_processed_data(df_cleaned, df_aggregated, grid_metadata):
    """Save processed data to disk"""
    print(f"\n[STEP 6] Saving processed data...")
    
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Save cleaned incident-level data
    df_cleaned.to_csv(PROCESSED_OUTPUT, index=False)
    print(f"  ✓ Saved cleaned data: {PROCESSED_OUTPUT}")
    
    # Save aggregated grid-time data
    df_aggregated.to_csv(GRID_OUTPUT, index=False)
    print(f"  ✓ Saved grid aggregated data: {GRID_OUTPUT}")
    
    # Save grid metadata
    metadata_path = "data/processed/grid_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("RISKGRID METADATA\n")
        f.write("=" * 50 + "\n\n")
        for key, value in grid_metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ Saved grid metadata: {metadata_path}")
# MAIN PIPELINE
def run_preprocessing_pipeline(sample_size=None):
    """Run the complete preprocessing pipeline"""
    
    print("\nStarting RiskGrid preprocessing pipeline...")
    print(f"Sample size: {sample_size if sample_size else 'Full dataset'}\n")
    
    # Step 1: Load and clean
    df_cleaned = load_and_clean_data(RAW_DATA_PATH, sample_size=sample_size)
    
    # Step 2: Create spatial grid
    df_cleaned, grid_metadata = create_spatial_grid(df_cleaned)
    
    # Step 3: Create temporal windows
    df_cleaned = create_time_windows(df_cleaned)
    
    # Step 4: Aggregate to grid-time cells
    df_aggregated = aggregate_to_grid_time(df_cleaned)
    
    # Step 5: Save everything
    save_processed_data(df_cleaned, df_aggregated, grid_metadata)
    
    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 80)
    
    print("\nSummary:")
    print(f"  • Original incidents: {len(df_cleaned):,}")
    print(f"  • Unique grid cells: {df_cleaned['cell_id'].nunique():,}")
    print(f"  • Time windows: {df_cleaned['time_window'].nunique():,}")
    print(f"  • Grid-time observations: {len(df_aggregated):,}")
    
    print("\nOutput files:")
    print(f"  • {PROCESSED_OUTPUT}")
    print(f"  • {GRID_OUTPUT}")
    print(f"  • data/processed/grid_metadata.txt")
    
    print("\n🚀 Next step: Feature engineering & model training")
    
    return df_cleaned, df_aggregated, grid_metadata
# RUN THE PIPELINE
if __name__ == "__main__":
    # Process full dataset (set sample_size=100000 for testing with smaller data)
    df_cleaned, df_aggregated, grid_metadata = run_preprocessing_pipeline(sample_size=None)
    
    print("\n" + "=" * 80)
    print("Ready for Phase 3: Feature Engineering!")
    print("=" * 80)