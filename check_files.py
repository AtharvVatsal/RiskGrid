"""
RiskGrid - File Checker
Diagnose file location issues for the dashboard
"""

import os
import pandas as pd

print("=" * 80)
print(" " * 25 + "RISKGRID FILE CHECKER")
print("=" * 80)

print(f"\n📂 Current directory: {os.getcwd()}")

# Check for required files
required_files = [
    "outputs/test_predictions.csv",
    "outputs/feature_importance.csv",
    "outputs/ultra/ultra_predictions.csv",
    "outputs/ultra/ultra_feature_importance.csv",
    "models/best_model.pkl",
    "models/ultra/xgb_ultra.pkl"
]

print("\n🔍 Checking for required files...\n")

found_files = []
missing_files = []

for filepath in required_files:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / 1024  # KB
        print(f"✅ FOUND: {filepath} ({size:.1f} KB)")
        found_files.append(filepath)
    else:
        print(f"❌ MISSING: {filepath}")
        missing_files.append(filepath)

print("\n" + "=" * 80)
print(f"Summary: {len(found_files)} found, {len(missing_files)} missing")
print("=" * 80)

# Check directory structure
print("\n📁 Directory structure:")

def show_tree(path, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return
    
    if not os.path.exists(path):
        return
    
    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            
            if os.path.isdir(item_path):
                print(f"{prefix}{'└── ' if is_last else '├── '}{item}/")
                new_prefix = prefix + ("    " if is_last else "│   ")
                show_tree(item_path, new_prefix, max_depth, current_depth + 1)
            else:
                size = os.path.getsize(item_path) / 1024
                print(f"{prefix}{'└── ' if is_last else '├── '}{item} ({size:.1f} KB)")
    except PermissionError:
        pass

show_tree("outputs")
show_tree("models")

# If prediction file exists, check its structure
print("\n" + "=" * 80)

prediction_files = [f for f in found_files if 'predictions.csv' in f]

if prediction_files:
    print(f"\n📊 Analyzing prediction file: {prediction_files[0]}")
    try:
        df = pd.read_csv(prediction_files[0], nrows=5)
        print(f"\n✅ Successfully loaded!")
        print(f"   Rows: {len(df):,}")
        print(f"\n   Columns ({len(df.columns)}):")
        for col in df.columns:
            dtype = df[col].dtype
            sample = df[col].iloc[0] if len(df) > 0 else 'N/A'
            print(f"     • {col:30s} ({dtype}) - Sample: {sample}")
    except Exception as e:
        print(f"\n❌ Error reading file: {e}")
else:
    print("\n❌ No prediction files found!")

# Recommendations
print("\n" + "=" * 80)
print("💡 RECOMMENDATIONS:")
print("=" * 80)

if not found_files:
    print("""
⚠️  No output files found!

ACTION REQUIRED:
1. Run the training script first:
   python src/train_model.py
   
   OR
   
   python src/train_model_ultra.py

2. Wait for training to complete (5-30 minutes)

3. Verify files are created in outputs/ folder

4. Then run the dashboard:
   streamlit run dashboard.py
""")
elif missing_files:
    print(f"""
⚠️  Some files missing: {len(missing_files)}

But found: {len(found_files)} files ✅

The dashboard should work with the files you have.

Try running the dashboard from the RiskGrid root folder:
   cd {os.getcwd()}
   streamlit run dashboard.py

If it still doesn't work, try running from one directory up:
   cd ..
   streamlit run dashboard.py
""")
else:
    print("""
✅ All files found! Dashboard should work perfectly.

Run the dashboard with:
   streamlit run dashboard.py

If you still get errors, make sure you're running from:
   """ + os.getcwd())

print("\n" + "=" * 80)
print("✓ Diagnostic complete!")
print("=" * 80)