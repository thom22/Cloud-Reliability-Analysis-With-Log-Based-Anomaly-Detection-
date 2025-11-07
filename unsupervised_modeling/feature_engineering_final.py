"""
Feature Engineering Module for HDFS Log Anomaly Detection
Handles both sequential (deep learning) and traditional ML features
Final Version - Complete Implementation
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import ast


class HDFSFeatureEngineering:
    """
    Complete feature extraction for HDFS log anomaly detection
    Supports both traditional ML and deep learning approaches
    """

    def __init__(self, max_sequence_length=100):
        """
        Initialize feature engineering

        Args:
            max_sequence_length: Maximum sequence length for padding (default: 100)
        """
        self.max_sequence_length = max_sequence_length
        self.event_to_id = {}
        self.vocab_size = 0
        self.feature_names = []

    def prepare_sequences_for_deep_learning(self, event_traces_df):
        """
        Prepare sequential features for Autoencoder/VAE/LSTM models

        Args:
            event_traces_df: DataFrame with Features and Label columns

        Returns:
            X: Padded sequences (numpy array)
            y: Binary labels (numpy array)
        """
        print("\n" + "=" * 70)
        print("PREPARING SEQUENTIAL FEATURES (Deep Learning)")
        print("=" * 70)

        sequences = []
        labels = []
        all_events = set()

        # Step 1: Build vocabulary and extract sequences
        print("\nStep 1: Building event vocabulary...")
        for idx, row in event_traces_df.iterrows():
            # Parse Features column
            seq = self._parse_sequence(row["Features"])

            # Collect all events
            all_events.update(seq)
            sequences.append(seq)

            # Extract label (Anomaly=1, Success=0)
            label = self._parse_label(row["Label"])
            labels.append(label)

        # Create event-to-ID mapping (0 reserved for padding)
        self.event_to_id = {
            event: idx + 1 for idx, event in enumerate(sorted(all_events))
        }
        self.vocab_size = len(self.event_to_id) + 1  # +1 for padding

        print(f"✓ Vocabulary size: {self.vocab_size} (including padding)")
        print(f"✓ Unique events: {len(all_events)}")

        # Step 2: Convert sequences to integer IDs
        print("\nStep 2: Converting sequences to integer IDs...")
        encoded_sequences = []
        for seq in sequences:
            encoded_seq = [self.event_to_id.get(event, 0) for event in seq]
            encoded_sequences.append(encoded_seq)

        # Step 3: Pad sequences to same length
        print(f"\nStep 3: Padding sequences to length {self.max_sequence_length}...")
        X = pad_sequences(
            encoded_sequences,
            maxlen=self.max_sequence_length,
            padding="post",
            truncating="post",
            value=0,
        )

        y = np.array(labels)

        print(f"\n✓ Final shape: X={X.shape}, y={y.shape}")
        print(f"✓ Anomaly rate: {y.sum() / len(y) * 100:.2f}%")

        return X, y

    def prepare_features_for_traditional_ml(self, event_matrix_df, event_traces_df):
        """
        Prepare count-based + engineered features for Isolation Forest

        Args:
            event_matrix_df: DataFrame with event occurrence counts
            event_traces_df: DataFrame with Features, TimeInterval, Latency

        Returns:
            X: Feature matrix (numpy array)
            y: Binary labels (numpy array)
            feature_names: List of feature names
        """
        print("\n" + "=" * 70)
        print("PREPARING TRADITIONAL ML FEATURES")
        print("=" * 70)

        # Step 1: Event count features (from matrix)
        print("\nStep 1: Extracting event count features...")
        event_cols = [col for col in event_matrix_df.columns if col.startswith("E")]
        count_features = event_matrix_df[event_cols].values
        print(f"✓ Event count features: {len(event_cols)} features")

        # Step 2: Sequence statistics
        print("\nStep 2: Computing sequence statistics...")
        seq_stats = self._extract_sequence_stats(event_traces_df)
        print(f"✓ Sequence statistics: {seq_stats.shape[1]} features")

        # Step 3: Temporal features
        print("\nStep 3: Extracting temporal features...")
        temporal_features = self._extract_temporal_features(event_traces_df)
        print(f"✓ Temporal features: {temporal_features.shape[1]} features")

        # Step 4: Error-specific features
        print("\nStep 4: Computing error features...")
        error_features = self._extract_error_features(event_matrix_df)
        print(f"✓ Error features: {error_features.shape[1]} features")

        # Combine all features
        X = np.hstack([count_features, seq_stats, temporal_features, error_features])

        # Feature names
        self.feature_names = (
            event_cols
            + ["seq_length", "unique_events", "event_diversity"]
            + ["time_interval", "latency"]
            + ["error_count", "error_ratio", "has_errors"]
        )

        # Labels
        y = event_traces_df["Label"].apply(self._parse_label).values

        print(f"\n✓ Total features: {X.shape[1]}")
        print(f"✓ Total samples: {X.shape[0]}")
        print(f"✓ Anomaly rate: {y.sum() / len(y) * 100:.2f}%")

        return X, y, self.feature_names

    # ========== Helper Methods ==========

    def _parse_sequence(self, features):
        """Parse Features column (handles list, string, etc.)"""
        if isinstance(features, list):
            return features
        elif isinstance(features, str):
            try:
                return ast.literal_eval(features)
            except (ValueError, SyntaxError):
                return [item.strip() for item in features.strip("[]").split(",")]
        else:
            return []

    def _parse_label(self, label):
        """Convert label to binary (1=Anomaly, 0=Success)"""
        if isinstance(label, str):
            return 1 if label.lower() in ["anomaly", "fail"] else 0
        return int(label)

    def _extract_sequence_stats(self, df):
        """Extract sequence-level statistics"""
        stats = []
        for _, row in df.iterrows():
            seq = self._parse_sequence(row["Features"])

            seq_len = len(seq)
            unique = len(set(seq))
            diversity = unique / seq_len if seq_len > 0 else 0

            stats.append([seq_len, unique, diversity])

        return np.array(stats)

    def _extract_temporal_features(self, df):
        """Extract temporal features (handle lists or single values)"""
        temporal_cols = ["TimeInterval", "Latency"]
        temporal_features = []

        for _, row in df.iterrows():
            temp_vals = []
            for col in temporal_cols:
                val = row[col]

                # Handle different formats
                if isinstance(val, list):
                    temp_vals.append(np.mean(val) if len(val) > 0 else 0)
                elif isinstance(val, str) and val.startswith("["):
                    try:
                        val_list = ast.literal_eval(val)
                        temp_vals.append(np.mean(val_list) if len(val_list) > 0 else 0)
                    except:
                        temp_vals.append(0)
                else:
                    try:
                        temp_vals.append(float(val) if pd.notna(val) else 0)
                    except:
                        temp_vals.append(0)

            temporal_features.append(temp_vals)

        return np.array(temporal_features)

    def _extract_error_features(self, matrix_df):
        """Extract error-related features"""
        error_events = ["E4", "E7", "E10", "E12"]
        event_cols = [col for col in matrix_df.columns if col.startswith("E")]

        # Total error count
        error_count = matrix_df[error_events].sum(axis=1).values.reshape(-1, 1)

        # Total events
        total_events = matrix_df[event_cols].sum(axis=1).values.reshape(-1, 1)

        # Error ratio
        error_ratio = error_count / (total_events + 1)  # +1 to avoid division by zero

        # Has errors (binary)
        has_errors = (error_count > 0).astype(int)

        return np.hstack([error_count, error_ratio, has_errors])

    def get_vocab_size(self):
        """Return vocabulary size"""
        return self.vocab_size

    def get_feature_names(self):
        """Return feature names for traditional ML"""
        return self.feature_names

    def get_event_mapping(self):
        """Return event to ID mapping"""
        return self.event_to_id


if __name__ == "__main__":
    print("=" * 70)
    print("HDFS FEATURE ENGINEERING MODULE - FINAL VERSION")
    print("=" * 70)
    print("\nThis module provides comprehensive feature extraction for:")
    print("  1. Deep Learning Models (Autoencoder, VAE, LSTM)")
    print("  2. Traditional ML Models (Isolation Forest, Random Forest)")
    print("\nUsage:")
    print("  fe = HDFSFeatureEngineering(max_sequence_length=100)")
    print("  X_seq, y_seq = fe.prepare_sequences_for_deep_learning(traces_df)")
    print(
        "  X_trad, y_trad, names = fe.prepare_features_for_traditional_ml(matrix_df, traces_df)"
    )
    print("=" * 70)
