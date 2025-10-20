"""
RiskGrid - Quick Grid Visualization
Visualize the spatial grid and incident distribution
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

GRID_DATA_PATH = "A:/Work/RiskGrid/data/processed/grid_aggregated.csv"
CLEANED_DATA_PATH = "A:/Work/RiskGrid/data/processed/cleaned_data.csv"

print("=" * 80)
print(" " * 25 + "RISKGRID GRID VISUALIZATION")
print("=" * 80)
# LOAD DATA
print("\n[1] Loading processed data...")
df_grid = pd.read_csv(GRID_DATA_PATH)
df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

print(f"  ✓ Loaded {len(df_grid):,} grid-time observations")
print(f"  ✓ Loaded {len(df_cleaned):,} cleaned incidents")
# CREATE VISUALIZATIONS
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main title
fig.suptitle('RiskGrid - Spatial-Temporal Analysis', 
             fontsize=20, fontweight='bold', y=0.98)
# PLOT 1: Incident Heatmap (All Time)
ax1 = fig.add_subplot(gs[0:2, 0:2])

# Aggregate total incidents per cell
cell_totals = df_cleaned.groupby(['grid_lon_idx', 'grid_lat_idx']).size().reset_index(name='count')

# Create pivot table for heatmap
heatmap_data = cell_totals.pivot(index='grid_lat_idx', columns='grid_lon_idx', values='count')
heatmap_data = heatmap_data.fillna(0)

# Plot
im1 = ax1.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', origin='lower')
ax1.set_title('Crime Density Heatmap (All Time)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('Longitude Grid Index', fontsize=11)
ax1.set_ylabel('Latitude Grid Index', fontsize=11)

# Add colorbar
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Incident Count', fontsize=10)

# Add grid
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# PLOT 2: Top 10 Hotspot Cells
ax2 = fig.add_subplot(gs[0, 2])

top_cells = cell_totals.nlargest(10, 'count')
bars = ax2.barh(range(len(top_cells)), top_cells['count'].values, color='crimson', alpha=0.7)
ax2.set_yticks(range(len(top_cells)))
ax2.set_yticklabels([f"Cell {row['grid_lon_idx']},{row['grid_lat_idx']}" 
                      for _, row in top_cells.iterrows()], fontsize=9)
ax2.set_xlabel('Incident Count', fontsize=10)
ax2.set_title('Top 10 Hotspot Cells', fontsize=12, fontweight='bold')
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)
# PLOT 3: Grid Statistics
ax3 = fig.add_subplot(gs[1, 2])
ax3.axis('off')

unique_cells = df_cleaned['cell_id'].nunique()
total_incidents = len(df_cleaned)
avg_per_cell = total_incidents / unique_cells
max_in_cell = cell_totals['count'].max()

# Count cells by incident density
low_density = len(cell_totals[cell_totals['count'] <= 10])
medium_density = len(cell_totals[(cell_totals['count'] > 10) & (cell_totals['count'] <= 50)])
high_density = len(cell_totals[cell_totals['count'] > 50])

stats_text = f"""
GRID STATISTICS
{'=' * 35}

Total Grid Cells: {unique_cells:,}
Total Incidents: {total_incidents:,}

Avg Incidents/Cell: {avg_per_cell:.1f}
Max Incidents/Cell: {max_in_cell}

DENSITY DISTRIBUTION
{'─' * 35}
Low (<= 10):   {low_density:4d} cells
Medium (11-50): {medium_density:4d} cells
High (> 50):    {high_density:4d} cells
"""

ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
# PLOT 4: Temporal Distribution
ax4 = fig.add_subplot(gs[2, 0])

df_cleaned['time_window'] = pd.to_datetime(df_cleaned['time_window'])
daily_counts = df_cleaned.groupby(df_cleaned['time_window'].dt.date).size()

ax4.plot(daily_counts.index, daily_counts.values, linewidth=2, color='steelblue', marker='o', markersize=4)
ax4.fill_between(daily_counts.index, daily_counts.values, alpha=0.3, color='steelblue')
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Incident Count', fontsize=10)
ax4.set_title('Daily Incident Trends', fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)
ax4.tick_params(axis='x', rotation=45)
# PLOT 5: Hour of Day Distribution
ax5 = fig.add_subplot(gs[2, 1])

hourly = df_cleaned['hour'].value_counts().sort_index()
bars = ax5.bar(hourly.index, hourly.values, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
ax5.set_xlabel('Hour of Day', fontsize=10)
ax5.set_ylabel('Incident Count', fontsize=10)
ax5.set_title('Hourly Distribution', fontsize=12, fontweight='bold')
ax5.set_xticks(range(0, 24, 2))
ax5.grid(axis='y', alpha=0.3)

# Highlight peak hour
peak_hour = hourly.idxmax()
bars[peak_hour].set_color('crimson')
bars[peak_hour].set_alpha(0.9)
# PLOT 6: Grid Time Coverage
ax6 = fig.add_subplot(gs[2, 2])

# Count incidents per grid-time cell
coverage = df_grid['incident_count'].value_counts().sort_index()

ax6.bar(coverage.index, coverage.values, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
ax6.set_xlabel('Incidents per Grid-Time Cell', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('Grid-Time Cell Distribution', fontsize=12, fontweight='bold')
ax6.set_xlim(0, min(20, coverage.index.max()))  # Limit x-axis for readability
ax6.grid(axis='y', alpha=0.3)
# SAVE AND DISPLAY
output_path = "outputs/grid_visualization.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Visualization saved: {output_path}")

plt.show()

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nKey Insights:")
print(f"  • Most active cell has {max_in_cell} incidents")
print(f"  • Peak hour for crimes: {peak_hour}:00")
print(f"  • {high_density} cells are high-density hotspots (>50 incidents)")