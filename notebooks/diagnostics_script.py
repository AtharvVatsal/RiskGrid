"""
RiskGrid - Fixed Column Detection Script
"""

import pandas as pd
import numpy as np

DATA_PATH = "data/raw/chicago_crimes.csv"
SAMPLE_SIZE = 50000

print("=" * 80)
print(" " * 25 + "RISKGRID DATA DIAGNOSTIC (FIXED)")
print("=" * 80)

print("\n[STEP 1] Loading data and showing ALL columns...\n")

df = pd.read_csv(DATA_PATH, nrows=SAMPLE_SIZE, low_memory=False)
print(f"✓ Loaded {len(df):,} rows\n")

print("=" * 80)
print("ALL COLUMN NAMES IN YOUR CSV:")
print("=" * 80)
for i, col in enumerate(df.columns, 1):
    sample_value = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
    dtype = df[col].dtype
    print(f"{i:3d}. {col:40s} | Type: {str(dtype):15s} | Sample: {str(sample_value)[:30]}")

print("\n\n" + "=" * 80)
print("FIRST 3 ROWS OF DATA:")
print("=" * 80)
print(df.head(3).to_string())

print("\n\n" + "=" * 80)
print("LOOKING FOR KEY COLUMNS...")
print("=" * 80)

date_col = None
lat_col = None
lon_col = None
crime_type_col = None

for col in df.columns:
    col_lower = col.lower()
    if 'date' in col_lower and 'update' not in col_lower:
        date_col = col
        break

for col in df.columns:
    col_lower = col.lower()
    if df[col].dtype in ['float64', 'float32', 'int64']:
        if 'lat' in col_lower or col_lower == 'y coordinate':
            sample_vals = df[col].dropna().head(100)
            if len(sample_vals) > 0:
                avg_val = sample_vals.mean()
                if 41 <= avg_val <= 42:
                    lat_col = col
                    break

for col in df.columns:
    col_lower = col.lower()
    if df[col].dtype in ['float64', 'float32', 'int64']:
        if 'lon' in col_lower or 'long' in col_lower or col_lower == 'x coordinate':
            sample_vals = df[col].dropna().head(100)
            if len(sample_vals) > 0:
                avg_val = sample_vals.mean()
                if -88 <= avg_val <= -87:
                    lon_col = col
                    break
for col in df.columns:
    if df[col].dtype == 'object':
        col_lower = col.lower()
        if 'primary type' in col_lower or 'crime type' in col_lower or col == 'Primary Type':
            crime_type_col = col
            break

print("\n DETECTED COLUMNS:")
print(f"  Date/Time : {date_col if date_col else '❌ NOT FOUND'}")
print(f"  Latitude  : {lat_col if lat_col else '❌ NOT FOUND'}")
print(f"  Longitude : {lon_col if lon_col else '❌ NOT FOUND'}")
print(f"  Crime Type: {crime_type_col if crime_type_col else '❌ NOT FOUND'}")

print("\n\n" + "=" * 80)
print("DATA QUALITY ANALYSIS:")
print("=" * 80)

if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    valid_dates = df[date_col].dropna()
    print(f"\n📅 DATE RANGE:")
    print(f"   Earliest: {valid_dates.min()}")
    print(f"   Latest  : {valid_dates.max()}")
    print(f"   Span    : {(valid_dates.max() - valid_dates.min()).days} days")

if lat_col and lon_col:
    valid_coords = df[[lat_col, lon_col]].dropna()
    print(f"\nCOORDINATES:")
    print(f"   Valid records: {len(valid_coords):,} / {len(df):,} ({len(valid_coords)/len(df)*100:.1f}%)")
    if len(valid_coords) > 0:
        print(f"   Lat range : {valid_coords[lat_col].min():.4f} to {valid_coords[lat_col].max():.4f}")
        print(f"   Lon range : {valid_coords[lon_col].min():.4f} to {valid_coords[lon_col].max():.4f}")

if crime_type_col:
    print(f"\nCRIME TYPES:")
    print(f"   Unique types: {df[crime_type_col].nunique()}")
    print(f"\n   Top 5:")
    for i, (crime, count) in enumerate(df[crime_type_col].value_counts().head(5).items(), 1):
        pct = count / len(df) * 100
        print(f"   {i}. {crime:30s} {count:6,} ({pct:5.1f}%)")

print("\n\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

print("\nCONFIGURATION FOR PHASE 2:\n")
print("CONFIG = {")
print(f"    'date_column': '{date_col}',")
print(f"    'latitude_column': '{lat_col}',")
print(f"    'longitude_column': '{lon_col}',")
print(f"    'crime_type_column': '{crime_type_col}',")
print("}")
print("\n\nVALIDATION:")
issues = []
if not date_col:
    issues.append("Date column not found")
if not lat_col:
    issues.append("Latitude column not found")
if not lon_col:
    issues.append("Longitude column not found")
if not crime_type_col:
    issues.append("Crime type column not found")

if issues:
    print("\n".join(issues))
    print("\nPlease review the column list above and manually identify:")
    print("   - Which column has the DATE/TIME?")
    print("   - Which column has LATITUDE (should be ~41.x for Chicago)?")
    print("   - Which column has LONGITUDE (should be ~-87.x for Chicago)?")
    print("   - Which column has CRIME TYPE/DESCRIPTION?")
else:
    print("All columns detected successfully!")