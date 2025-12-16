# DEEPLOG LSTM - FINAL PROPOSED MODEL
# Training on Combined Dataset (Event Sequences + Count Features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import warnings

warnings.filterwarnings("ignore")

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))

# 1. LOAD DATA

# Load datasets
event_traces = pd.read_csv("Event_traces.csv")
event_matrix = pd.read_csv("Event_occurrence_matrix.csv")

print(f"Event Traces shape: {event_traces.shape}")
print(f"Event Matrix shape: {event_matrix.shape}")

# 2. PREPARE SEQUENTIAL FEATURES (Event Sequences)


# Parse event sequences from Features column
def parse_sequence(seq_str):
    if isinstance(seq_str, str):
        # Remove brackets and split
        seq = seq_str.strip("[]").replace("'", "").split(", ")
        return [s.strip() for s in seq if s.strip()]
    return []


event_traces["EventSequence"] = event_traces["Features"].apply(parse_sequence)

# Create event to ID mapping (E1=1, E2=2, ..., E29=29, padding=0)
all_events = set()
for seq in event_traces["EventSequence"]:
    all_events.update(seq)
all_events = sorted(list(all_events))
event_to_id = {event: idx + 1 for idx, event in enumerate(all_events)}
event_to_id["PAD"] = 0

print(f"\nEvent vocabulary size: {len(all_events)}")
print(f"Event to ID mapping (first 10): {dict(list(event_to_id.items())[:10])}")


# Convert sequences to integer IDs
def encode_sequence(seq):
    return [event_to_id.get(event, 0) for event in seq]


event_traces["EncodedSequence"] = event_traces["EventSequence"].apply(encode_sequence)

# Analyze sequence lengths
seq_lengths = event_traces["EncodedSequence"].apply(len)
print(f"\nSequence length statistics:")
print(f"Min: {seq_lengths.min()}")
print(f"Max: {seq_lengths.max()}")
print(f"Mean: {seq_lengths.mean():.2f}")
print(f"Median: {seq_lengths.median():.2f}")
print(f"90th percentile: {seq_lengths.quantile(0.90):.2f}")

# Set max sequence length
MAX_SEQ_LENGTH = 100
print(f"\nUsing MAX_SEQ_LENGTH = {MAX_SEQ_LENGTH}")

# Pad sequences
X_sequences = pad_sequences(
    event_traces["EncodedSequence"].tolist(),
    maxlen=MAX_SEQ_LENGTH,
    padding="post",
    truncating="post",
    value=0,
)

print(f"Padded sequences shape: {X_sequences.shape}")

# 3. PREPARE COUNT FEATURES (E1-E29)

# Get count features from event matrix
event_columns = [col for col in event_matrix.columns if col.startswith("E")]
X_counts = event_matrix[event_columns].values

print(f"\nCount features shape: {X_counts.shape}")
print(f"Number of event features: {len(event_columns)}")

# Scale count features
scaler = StandardScaler()
X_counts_scaled = scaler.fit_transform(X_counts)

# 4. PREPARE LABELS

# Get labels from event matrix (both datasets have same order)
y = event_matrix["Label"].map({"Success": 0, "Fail": 1}).values

print(f"\nLabel distribution:")
print(f"Normal (0): {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.2f}%)")
print(f"Anomaly (1): {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.2f}%)")

# 5. TRAIN-TEST SPLIT

# Split sequences
X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_sequences, y, test_size=0.2, random_state=42, stratify=y
)

# Split count features
X_count_train, X_count_test = train_test_split(
    X_counts_scaled, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set:")
print(f"  Sequences: {X_seq_train.shape}")
print(f"  Counts: {X_count_train.shape}")
print(f"  Labels: {y_train.shape}")
print(f"\nTest set:")
print(f"  Sequences: {X_seq_test.shape}")
print(f"  Counts: {X_count_test.shape}")
print(f"  Labels: {y_test.shape}")

# 6. BUILD DEEPLOG LSTM MODEL

# Model architecture
VOCAB_SIZE = len(event_to_id)
EMBEDDING_DIM = 64
LSTM_UNITS = 128
DROPOUT_RATE = 0.3

print(f"\nBuilding DeepLog LSTM model...")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Embedding dimension: {EMBEDDING_DIM}")
print(f"LSTM units: {LSTM_UNITS}")

# Sequential input branch
sequence_input = keras.Input(shape=(MAX_SEQ_LENGTH,), name="sequence_input")
embedded = Embedding(
    input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, mask_zero=True, name="embedding"
)(sequence_input)

# Bidirectional LSTM layers
lstm1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(embedded)
dropout1 = Dropout(DROPOUT_RATE)(lstm1)
lstm2 = Bidirectional(LSTM(LSTM_UNITS // 2))(dropout1)
dropout2 = Dropout(DROPOUT_RATE)(lstm2)

# Count features input branch
count_input = keras.Input(shape=(len(event_columns),), name="count_input")
count_dense = Dense(32, activation="relu")(count_input)
count_dropout = Dropout(DROPOUT_RATE)(count_dense)

# Concatenate both branches
concatenated = keras.layers.concatenate([dropout2, count_dropout])

# Final dense layers
dense1 = Dense(64, activation="relu")(concatenated)
dropout3 = Dropout(DROPOUT_RATE)(dense1)
output = Dense(1, activation="sigmoid", name="output")(dropout3)

# Create model
model = keras.Model(
    inputs=[sequence_input, count_input], outputs=output, name="DeepLog_LSTM"
)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
)

model.summary()

# 7. TRAIN MODEL

# Callbacks
callbacks = [
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    ModelCheckpoint(
        "deeplog_best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
    ),
]

# Class weights for imbalanced data
class_weight = {0: 1.0, 1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])}
print(f"\nClass weights: {class_weight}")

# Train
print("\nTraining DeepLog LSTM model...")
history = model.fit(
    [X_seq_train, X_count_train],
    y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=256,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# 8. EVALUATE MODEL

print("\nEvaluating on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(
    [X_seq_test, X_count_test], y_test, verbose=0
)

print(f"\nTest Results:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# Predictions
y_pred_proba = model.predict([X_seq_test, X_count_test], verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\nDetailed Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 9. VISUALIZATION

# Create comprehensive evaluation plot
fig = plt.figure(figsize=(20, 12))
fig.suptitle(
    "DeepLog LSTM - Comprehensive Evaluation", fontsize=20, fontweight="bold", y=0.98
)

colors = {
    "normal": "#2ecc71",
    "anomaly": "#e74c3c",
    "confusion_cmap": "RdYlGn_r",
    "roc": "#3498db",
    "pr": "#9b59b6",
    "bar1": "#1abc9c",
    "bar2": "#f39c12",
    "bar3": "#e67e22",
}

# 1. Training History - Loss
ax1 = plt.subplot(3, 3, 1)
ax1.plot(history.history["loss"], label="Train Loss", linewidth=2, color="#3498db")
ax1.plot(history.history["val_loss"], label="Val Loss", linewidth=2, color="#e74c3c")
ax1.set_xlabel("Epoch", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Training History - Accuracy
ax2 = plt.subplot(3, 3, 2)
ax2.plot(history.history["accuracy"], label="Train Acc", linewidth=2, color="#2ecc71")
ax2.plot(history.history["val_accuracy"], label="Val Acc", linewidth=2, color="#f39c12")
ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Accuracy", fontsize=12)
ax2.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix
ax3 = plt.subplot(3, 3, 3)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap=colors["confusion_cmap"],
    cbar_kws={"label": ""},
    ax=ax3,
    annot_kws={"size": 14},
)
ax3.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
ax3.set_ylabel("True Label", fontsize=12)
ax3.set_xlabel("Predicted Label", fontsize=12)
ax3.set_xticklabels(["Normal", "Anomaly"])
ax3.set_yticklabels(["Normal", "Anomaly"])

# 4. ROC Curve
ax4 = plt.subplot(3, 3, 4)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
ax4.plot(
    fpr, tpr, color=colors["roc"], lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})"
)
ax4.plot([0, 1], [0, 1], color="black", lw=1.5, linestyle="--", label="Random")
ax4.set_xlim([0.0, 1.0])
ax4.set_ylim([0.0, 1.05])
ax4.set_xlabel("False Positive Rate", fontsize=12)
ax4.set_ylabel("True Positive Rate", fontsize=12)
ax4.set_title("ROC Curve", fontsize=14, fontweight="bold")
ax4.legend(loc="lower right", fontsize=10)
ax4.grid(True, alpha=0.3)

# 5. Precision-Recall Curve
ax5 = plt.subplot(3, 3, 5)
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
ax5.plot(recall_curve, precision_curve, color=colors["pr"], lw=2.5)
ax5.set_xlim([0.0, 1.0])
ax5.set_ylim([0.0, 1.05])
ax5.set_xlabel("Recall", fontsize=12)
ax5.set_ylabel("Precision", fontsize=12)
ax5.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
ax5.grid(True, alpha=0.3)

# 6. Score Distribution
ax6 = plt.subplot(3, 3, 6)
normal_scores = y_pred_proba[y_test == 0]
anomaly_scores = y_pred_proba[y_test == 1]
bins = np.linspace(0, 1, 50)
ax6.hist(
    normal_scores,
    bins=bins,
    alpha=0.7,
    label="Normal",
    color=colors["normal"],
    edgecolor="black",
    linewidth=0.5,
)
ax6.hist(
    anomaly_scores,
    bins=bins,
    alpha=0.7,
    label="Anomaly",
    color=colors["anomaly"],
    edgecolor="black",
    linewidth=0.5,
)
ax6.set_xlabel("Anomaly Score", fontsize=12)
ax6.set_ylabel("Frequency", fontsize=12)
ax6.set_title("Score Distribution by True Label", fontsize=14, fontweight="bold")
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis="y")

# 7. Performance Metrics
ax7 = plt.subplot(3, 3, 7)
metrics = [precision, recall, f1]
metric_names = ["Precision", "Recall", "F1-Score"]
bars = ax7.bar(
    metric_names,
    metrics,
    color=[colors["bar1"], colors["bar2"], colors["bar3"]],
    edgecolor="black",
    linewidth=1.5,
    alpha=0.8,
)
ax7.set_ylim([0, 1])
ax7.set_ylabel("Score", fontsize=12)
ax7.set_title("Performance Metrics", fontsize=14, fontweight="bold")
ax7.grid(True, alpha=0.3, axis="y")
for bar, metric in zip(bars, metrics):
    height = bar.get_height()
    ax7.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{metric:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# 8. True vs Predicted Distribution
ax8 = plt.subplot(3, 3, 8)
true_counts = [sum(y_test == 0), sum(y_test == 1)]
pred_counts = [sum(y_pred == 0), sum(y_pred == 1)]
x = np.arange(2)
width = 0.35
bars1 = ax8.bar(
    x - width / 2,
    true_counts,
    width,
    label="True",
    color="#3498db",
    edgecolor="black",
    linewidth=1.5,
    alpha=0.8,
)
bars2 = ax8.bar(
    x + width / 2,
    pred_counts,
    width,
    label="Predicted",
    color="#e74c3c",
    edgecolor="black",
    linewidth=1.5,
    alpha=0.8,
)
ax8.set_ylabel("Count", fontsize=12)
ax8.set_title("True vs Predicted Distribution", fontsize=14, fontweight="bold")
ax8.set_xticks(x)
ax8.set_xticklabels(["Normal", "Anomaly"])
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3, axis="y")

# 9. Precision-Recall Trade-off
ax9 = plt.subplot(3, 3, 9)
precision_vals = history.history.get("precision", [])
recall_vals = history.history.get("recall", [])
if precision_vals and recall_vals:
    epochs = range(1, len(precision_vals) + 1)
    ax9.plot(epochs, precision_vals, label="Precision", linewidth=2, color="#1abc9c")
    ax9.plot(epochs, recall_vals, label="Recall", linewidth=2, color="#f39c12")
    ax9.set_xlabel("Epoch", fontsize=12)
    ax9.set_ylabel("Score", fontsize=12)
    ax9.set_title("Precision & Recall During Training", fontsize=14, fontweight="bold")
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("deeplog_evaluation.png", dpi=300, bbox_inches="tight")
plt.show()

# 10. CLASSIFICATION REPORT

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

# 11. SAVE MODEL

model.save("deeplog_final_model.h5")
print("\nModel saved as 'deeplog_final_model.h5'")
print("Best model checkpoint saved as 'deeplog_best_model.h5'")
