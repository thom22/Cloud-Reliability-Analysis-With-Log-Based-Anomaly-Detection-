"""
Isolation Forest Model for HDFS Log Anomaly Detection
Unsupervised baseline with comprehensive evaluation
Final Version - Complete Implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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


class IsolationForestDetector:
    """
    Isolation Forest for unsupervised anomaly detection
    With comprehensive evaluation and visualization
    """

    def __init__(self, contamination=0.03, n_estimators=100, random_state=42):
        """
        Initialize Isolation Forest

        Args:
            contamination: Expected proportion of outliers (default: 0.03 for 3%)
            n_estimators: Number of trees (default: 100)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples="auto",
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def train(self, X_train):
        """
        Train Isolation Forest (unsupervised)

        Args:
            X_train: Training features
        """
        print("\n" + "=" * 70)
        print("TRAINING ISOLATION FOREST (Unsupervised)")
        print("=" * 70)
        print(f"\nContamination: {self.contamination}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features: {X_train.shape[1]}")

        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        print("Training Isolation Forest...")
        self.model.fit(X_train_scaled)
        self.is_fitted = True

        print("✓ Training complete!")

    def predict(self, X):
        """
        Predict anomalies

        Args:
            X: Features to predict

        Returns:
            predictions: Binary predictions (1=Anomaly, 0=Normal)
            scores: Anomaly scores (lower = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction!")

        X_scaled = self.scaler.transform(X)

        # Predict (-1 for anomaly, 1 for normal)
        y_pred_if = self.model.predict(X_scaled)
        y_pred = (y_pred_if == -1).astype(int)  # Convert to 0/1

        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)

        return y_pred, scores

    def evaluate(self, X_test, y_test, save_path="isolation_forest_results.png"):
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
        print("EVALUATING ISOLATION FOREST")
        print("=" * 70)

        # Predictions
        y_pred, scores = self.predict(X_test)

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

        # ROC-AUC (using scores)
        # Convert scores: lower scores = more anomalous, so negate for ROC
        try:
            auc = roc_auc_score(y_test, -scores)  # Negate scores
        except:
            auc = 0.0

        print("\n--- Key Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {auc:.4f}")

        # Create visualization
        self._plot_results(y_test, y_pred, scores, cm, save_path)

        return {
            "y_pred": y_pred,
            "scores": scores,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "confusion_matrix": cm,
        }

    def _plot_results(self, y_test, y_pred, scores, cm, save_path):
        """
        Create comprehensive visualization

        Args:
            y_test: True labels
            y_pred: Predicted labels
            scores: Anomaly scores
            cm: Confusion matrix
            save_path: Path to save figure
        """
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
            fpr, tpr, _ = roc_curve(y_test, -scores)  # Negate scores
            auc = roc_auc_score(y_test, -scores)

            ax2.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc:.4f})")
            ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve", fontsize=14, fontweight="bold")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except:
            ax2.text(
                0.5,
                0.5,
                "ROC curve unavailable",
                ha="center",
                va="center",
                fontsize=12,
            )

        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[0, 2])
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, -scores)
            ax3.plot(recall_vals, precision_vals, linewidth=2)
            ax3.set_xlabel("Recall")
            ax3.set_ylabel("Precision")
            ax3.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
            ax3.grid(True, alpha=0.3)
        except:
            ax3.text(
                0.5,
                0.5,
                "PR curve unavailable",
                ha="center",
                va="center",
                fontsize=12,
            )

        # 4. Score Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]

        ax4.hist(normal_scores, bins=50, alpha=0.7, label="Normal", color="green")
        ax4.hist(anomaly_scores, bins=50, alpha=0.7, label="Anomaly", color="red")
        ax4.set_xlabel("Anomaly Score")
        ax4.set_ylabel("Frequency")
        ax4.set_title(
            "Score Distribution by True Label", fontsize=14, fontweight="bold"
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Performance Metrics Bar Chart
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

        # Add value labels on bars
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
            "Isolation Forest - Comprehensive Evaluation",
            fontsize=16,
            fontweight="bold",
            y=0.995,
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Results visualization saved: {save_path}")
        plt.show()


if __name__ == "__main__":
    print("=" * 70)
    print("ISOLATION FOREST DETECTOR - FINAL VERSION")
    print("=" * 70)
    print("\nUnsupervised anomaly detection with comprehensive evaluation")
    print("\nUsage:")
    print("  detector = IsolationForestDetector(contamination=0.03)")
    print("  detector.train(X_train)")
    print("  results = detector.evaluate(X_test, y_test)")
    print("=" * 70)
