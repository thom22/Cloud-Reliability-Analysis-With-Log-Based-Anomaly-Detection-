"""
Save Required Files for DeepLog Deployment
Run this AFTER training your DeepLog model to save the preprocessing artifacts
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import ast

# Step 1: Create and Save Event-to-ID Mapping

print("Creating event-to-id mapping...")

# Load your event traces to get all unique events
df = pd.read_csv("Event_traces.csv")

# Extract all unique events from Features column
all_events = set()
for features_str in df["Features"]:
    if pd.notna(features_str):
        try:
            event_list = (
                ast.literal_eval(features_str)
                if isinstance(features_str, str)
                else features_str
            )
            all_events.update(event_list)
        except:
            pass

# Create mapping (0 reserved for padding)
event_to_id = {event: idx + 1 for idx, event in enumerate(sorted(all_events))}

# Add padding
event_to_id["<PAD>"] = 0

print(f"Found {len(all_events)} unique events")
print(f"Event mapping sample: {dict(list(event_to_id.items())[:5])}")

# Save
joblib.dump(event_to_id, "event_to_id.pkl")
print("✓ Saved event_to_id.pkl")

# Step 2: Create and Save Count Features Scaler

print("\nCreating count features scaler...")

# Load the occurrence matrix
occurrence_df = pd.read_csv("Event_occurrence_matrix.csv")

# Get event columns (E1-E29)
event_columns = [
    col for col in occurrence_df.columns if col.startswith("E") and col[1:].isdigit()
]
print(f"Found {len(event_columns)} event count features: {event_columns[:5]}...")

# Extract count features
X_counts = occurrence_df[event_columns].values

# Fit scaler
scaler = StandardScaler()
scaler.fit(X_counts)

print(f"Scaler fit on {X_counts.shape[0]} samples")
print(f"Feature means (first 5): {scaler.mean_[:5]}")
print(f"Feature stds (first 5): {scaler.scale_[:5]}")

# Save
joblib.dump(scaler, "count_scaler.pkl")
print("✓ Saved count_scaler.pkl")

# Step 3: Save Feature Names for Reference

print("\nSaving feature names...")
joblib.dump(event_columns, "feature_names.pkl")
print("✓ Saved feature_names.pkl")

# Step 4: Verification

print("\n" + "=" * 70)
print("VERIFICATION - Testing if files can be loaded")
print("=" * 70)

try:
    loaded_event_to_id = joblib.load("event_to_id.pkl")
    loaded_scaler = joblib.load("count_scaler.pkl")
    loaded_features = joblib.load("feature_names.pkl")

    print(f"✓ event_to_id.pkl loaded successfully ({len(loaded_event_to_id)} events)")
    print(f"✓ count_scaler.pkl loaded successfully")
    print(f"✓ feature_names.pkl loaded successfully ({len(loaded_features)} features)")

    print(
        "\nAll files saved successfully! You can now use them in your Streamlit dashboard."
    )

except Exception as e:
    print(f"✗ Error loading files: {e}")

# Step 5: Create deployment checklist

print("\n" + "=" * 70)
print("DEPLOYMENT CHECKLIST")
print("=" * 70)
print(
    """
For your Streamlit dashboard to work, you need these files:

Required Model Files:
  [✓] deeplog_best_model.h5          (your trained model)
  [✓] event_to_id.pkl                (event encoding mapping)
  [✓] count_scaler.pkl               (feature scaler)
  [✓] feature_names.pkl              (optional, for reference)

Required Data Files:
  [ ] combined_dataset.csv           (your data with all features)
  [ ] HDFS.log_templates.csv         (event descriptions)

All these files should be in the same directory as your dashboard script.
"""
)

print("\nNext steps:")
print("1. Make sure you have deeplog_best_model.h5 from your training")
print("2. Copy all the above files to your dashboard directory")
print("3. Run the updated dashboard script")
