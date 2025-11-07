"""
Autoencoder Model for HDFS Log Anomaly Detection
Deep learning-based unsupervised anomaly detection
Final Version - Complete Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


class AutoencoderDetector:
    """
    Autoencoder for unsupervised anomaly detection
    Detects anomalies based on reconstruction error
    """

    def __init__(self, input_dim, encoding_dim=32):
        """
        Initialize Autoencoder

        Args:
            input_dim: Dimension of input features
            encoding_dim: Dimension of encoded representation (default: 32)
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.history = None
        self.threshold = None

    def build_model(self, hidden_layers=[64, 32]):
        """
        Build Autoencoder architecture

        Args:
            hidden_layers: List of hidden layer sizes (default: [64, 32])

        Returns:
            model: Compiled Keras model
        """
        print("\n" + "=" * 70)
        print("BUILDING AUTOENCODER MODEL")
        print("=" * 70)

        # Encoder
        encoder_input = layers.Input(shape=(self.input_dim,), name="input")
        x = encoder_input

        for i, units in enumerate(hidden_layers):
            x = layers.Dense(units, activation="relu", name=f"encoder_dense_{i+1}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation="relu", name="bottleneck")(
            x
        )

        # Decoder (mirror of encoder)
        x = encoded
        for i, units in enumerate(reversed(hidden_layers)):
            x = layers.Dense(units, activation="relu", name=f"decoder_dense_{i+1}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)

        # Output
        decoded = layers.Dense(self.input_dim, activation="sigmoid", name="output")(x)

        # Create model
        self.model = models.Model(encoder_input, decoded, name="autoencoder")

        # Compile
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        print("\n--- Model Architecture ---")
        self.model.summary()

        return self.model

    def train(
        self,
        X_train,
        X_val,
        epochs=100,
        batch_size=32,
        patience=15,
        save_path="best_autoencoder.h5",
    ):
        """
        Train Autoencoder on normal data

        Args:
            X_train: Training features (should be mostly normal)
            X_val: Validation features
            epochs: Maximum number of epochs (default: 100)
            batch_size: Batch size (default: 32)
            patience: Early stopping patience (default: 15)
            save_path: Path to save best model

        Returns:
            history: Training history
        """
        print("\n" + "=" * 70)
        print("TRAINING AUTOENCODER")
        print("=" * 70)

        print(f"\nTraining samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Input dimension: {X_train.shape[1]}")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
            callbacks.ModelCheckpoint(
                save_path, monitor="val_loss", save_best_only=True, verbose=1
            ),
        ]

        # Train (input = output for autoencoder)
        print(f"\nTraining for up to {epochs} epochs...")
        self.history = self.model.fit(
            X_train,
            X_train,  # Train to reconstruct input
            validation_data=(X_val, X_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1,
        )

        print("\n✓ Training complete!")
        print(f"✓ Best model saved to: {save_path}")

        return self.history

    def compute_threshold(self, X_train, percentile=95):
        """
        Compute anomaly threshold based on training data reconstruction error

        Args:
            X_train: Training features
            percentile: Percentile for threshold (default: 95)

        Returns:
            threshold: Reconstruction error threshold
        """
        print("\n" + "=" * 70)
        print("COMPUTING ANOMALY THRESHOLD")
        print("=" * 70)

        # Compute reconstruction errors on training data
        X_reconstructed = self.model.predict(X_train, verbose=0)
        reconstruction_errors = np.mean(np.square(X_train - X_reconstructed), axis=1)

        # Set threshold at percentile
        self.threshold = np.percentile(reconstruction_errors, percentile)

        print(f"\nReconstruction error statistics (training data):")
        print(f"  Mean: {np.mean(reconstruction_errors):.6f}")
        print(f"  Std:  {np.std(reconstruction_errors):.6f}")
        print(f"  Min:  {np.min(reconstruction_errors):.6f}")
        print(f"  Max:  {np.max(reconstruction_errors):.6f}")
        print(f"\n✓ Threshold set at {percentile}th percentile: {self.threshold:.6f}")

        return self.threshold

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error

        Args:
            X: Features to predict

        Returns:
            predictions: Binary predictions (1=Anomaly, 0=Normal)
            errors: Reconstruction errors
        """
        if self.model is None:
            raise ValueError("Model must be built and trained before prediction!")

        if self.threshold is None:
            raise ValueError(
                "Threshold must be computed before prediction! Call compute_threshold() first."
            )

        # Compute reconstruction errors
        X_reconstructed = self.model.predict(X, verbose=0)
        errors = np.mean(np.square(X - X_reconstructed), axis=1)

        # Predict anomalies (error > threshold)
        y_pred = (errors > self.threshold).astype(int)

        return y_pred, errors

    def evaluate(self, X_test, y_test, save_path="autoencoder_results.png"):
        """
        Comprehensive evaluation with visualizations

        Args:
            X_test: Test features
            y_test: True labels
            save_path: Path to save visualization

        Returns:
            results: Dictionary with predictions and metrics
        """
        print("\n" + "=" * 70)
        print("EVALUATING AUTOENCODER")
        print("=" * 70)

        # Predictions
        y_pred, errors = self.predict(X_test)

        # Metrics
        print("\n--- Classification Report ---")
        print(
            classification_report(
                y_test, y_pred, target_names=["Normal", "Anomaly"], digits=4
            )
        )

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("\n--- Confusion Matrix ---")
        print(f"True Negatives:  {tn:6d}  |  False Positives: {fp:6d}")
        print(f"False Negatives: {fn:6d}  |  True Positives:  {tp:6d}")

        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ROC-AUC
        try:
            auc = roc_auc_score(y_test, errors)
        except:
            auc = 0.0

        print("\n--- Key Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")
        print(f"Threshold: {self.threshold:.6f}")

        # Create visualization
        self._plot_results(y_test, y_pred, errors, cm, save_path)

        return {
            "y_pred": y_pred,
            "errors": errors,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
            "threshold": self.threshold,
        }

    def plot_training_history(self, save_path="autoencoder_training.png"):
        """
        Plot training history

        Args:
            save_path: Path to save figure
        """
        if self.history is None:
            print("No training history available.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        axes[0].plot(self.history.history["loss"], label="Train", linewidth=2)
        axes[0].plot(self.history.history["val_loss"], label="Validation", linewidth=2)
        axes[0].set_title("Loss", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (MSE)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(self.history.history["mae"], label="Train", linewidth=2)
        axes[1].plot(self.history.history["val_mae"], label="Validation", linewidth=2)
        axes[1].set_title("Mean Absolute Error", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Training history saved: {save_path}")
        plt.show()

    def _plot_results(self, y_test, y_pred, errors, cm, save_path):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax1,
        )
        ax1.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        ax1.set_ylabel("True Label")
        ax1.set_xlabel("Predicted Label")

        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        try:
            fpr, tpr, _ = roc_curve(y_test, errors)
            auc = roc_auc_score(y_test, errors)

            ax2.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
            ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except:
            ax2.text(0.5, 0.5, "ROC unavailable", ha="center", va="center", fontsize=12)

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, errors)
            ax3.plot(recall_vals, precision_vals, linewidth=2)
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
            ax3.grid(True, alpha=0.3)
        except:
            ax3.text(
                0.5, 0.5, "PR curve unavailable", ha="center", va="center", fontsize=12
            )

        # 4. Reconstruction Error Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        normal_errors = errors[y_test == 0]
        anomaly_errors = errors[y_test == 1]

        ax4.hist(normal_errors, bins=50, alpha=0.7, label="Normal", color="green")
        ax4.hist(anomaly_errors, bins=50, alpha=0.7, label="Anomaly", color="red")
        ax4.axvline(
            self.threshold,
            color="black",
            linestyle="--",
            linewidth=2,
            label="Threshold",
        )
        ax4.set_xlabel("Reconstruction Error")
        ax4.set_ylabel("Frequency")
        ax4.set_title(
            "Error Distribution by True Label", fontsize=14, fontweight="bold"
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Performance Metrics
        ax5 = fig.add_subplot(gs[1, 1])
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        metrics = ["Precision", "Recall", "F1-Score"]
        values = [precision, recall, f1]
        colors = ["#3498db", "#e74c3c", "#2ecc71"]

        bars = ax5.bar(metrics, values, color=colors, alpha=0.7)
        ax5.set_ylim([0, 1])
        ax5.set_ylabel("Score")
        ax5.set_title("Performance Metrics", fontsize=14, fontweight="bold")
        ax5.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 6. Prediction Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        pred_counts = [np.sum(y_pred == 0), np.sum(y_pred == 1)]
        true_counts = [np.sum(y_test == 0), np.sum(y_test == 1)]

        x = np.arange(2)
        width = 0.35

        ax6.bar(
            x - width / 2, true_counts, width, label="True", alpha=0.7, color="blue"
        )
        ax6.bar(
            x + width / 2, pred_counts, width, label="Predicted", alpha=0.7, color="red"
        )
        ax6.set_xticks(x)
        ax6.set_xticklabels(["Normal", "Anomaly"])
        ax6.set_ylabel("Count")
        ax6.set_title("True vs Predicted Distribution", fontsize=14, fontweight="bold")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Autoencoder - Comprehensive Evaluation",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Results visualization saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("AUTOENCODER DETECTOR - FINAL VERSION")
    print("=" * 70)
    print("\nDeep learning-based unsupervised anomaly detection")
    print("\nUsage:")
    print("  detector = AutoencoderDetector(input_dim=100, encoding_dim=32)")
    print("  detector.build_model(hidden_layers=[64, 32])")
    print("  detector.train(X_train, X_val, epochs=100)")
    print("  detector.compute_threshold(X_train, percentile=95)")
    print("  results = detector.evaluate(X_test, y_test)")
    print("=" * 70)
