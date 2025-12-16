"""
Cloud Reliability Metrics Module
Calculates MTTD, MTTR, and System Availability
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import ast


def convert_to_numeric(arr):
    """
    Convert array values to numeric (handles lists, strings, single values)

    Args:
        arr: Array-like input

    Returns:
        numeric_arr: Numpy array of numeric values
    """
    numeric_arr = []
    for val in arr:
        if isinstance(val, list):
            numeric_arr.append(np.mean(val) if len(val) > 0 else 0)
        elif isinstance(val, str) and val.startswith("["):
            try:
                val_list = ast.literal_eval(val)
                numeric_arr.append(np.mean(val_list) if len(val_list) > 0 else 0)
            except:
                numeric_arr.append(0)
        else:
            try:
                numeric_arr.append(float(val) if pd.notna(val) else 0)
            except:
                numeric_arr.append(0)
    return np.array(numeric_arr)


def calculate_mttd(y_test, y_pred, time_intervals):
    """
    Calculate Mean Time to Detect (MTTD)

    Args:
        y_test: True labels
        y_pred: Predicted labels
        time_intervals: Time taken to process each block

    Returns:
        mttd_metrics: Dictionary with MTTD statistics
    """
    # Convert to numeric if needed
    time_intervals = convert_to_numeric(time_intervals)

    # Get detected anomalies (true positives)
    detected_idx = np.where((y_test == 1) & (y_pred == 1))[0]

    if len(detected_idx) > 0:
        detection_times = time_intervals[detected_idx]
        mttd_mean = np.mean(detection_times)
        mttd_median = np.median(detection_times)
        mttd_p95 = np.percentile(detection_times, 95)
        mttd_min = np.min(detection_times)
        mttd_max = np.max(detection_times)
    else:
        mttd_mean = mttd_median = mttd_p95 = mttd_min = mttd_max = 0

    return {
        "mean": mttd_mean,
        "median": mttd_median,
        "p95": mttd_p95,
        "min": mttd_min,
        "max": mttd_max,
        "detected_count": len(detected_idx),
        "total_anomalies": (y_test == 1).sum(),
    }


def calculate_mttr(y_test, y_pred, latencies):
    """
    Calculate Mean Time to Resolve (MTTR)

    Args:
        y_test: True labels
        y_pred: Predicted labels
        latencies: Latency/resolution time for each block

    Returns:
        mttr_metrics: Dictionary with MTTR statistics
    """
    # Convert to numeric if needed
    latencies = convert_to_numeric(latencies)

    # Get detected anomalies (true positives)
    detected_idx = np.where((y_test == 1) & (y_pred == 1))[0]

    if len(detected_idx) > 0:
        resolution_times = latencies[detected_idx]
        mttr_mean = np.mean(resolution_times)
        mttr_median = np.median(resolution_times)
        mttr_p95 = np.percentile(resolution_times, 95)
        mttr_min = np.min(resolution_times)
        mttr_max = np.max(resolution_times)
    else:
        mttr_mean = mttr_median = mttr_p95 = mttr_min = mttr_max = 0

    return {
        "mean": mttr_mean,
        "median": mttr_median,
        "p95": mttr_p95,
        "min": mttr_min,
        "max": mttr_max,
    }


def calculate_availability(y_test, y_pred):
    """
    Calculate system availability metrics

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        availability_metrics: Dictionary with availability statistics
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Detection rate (recall)
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

    # False alarm rate
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    # System availability (considering undetected failures as downtime)
    availability = 1 - (fn / len(y_test))

    # False discovery rate
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0

    return {
        "detection_rate": detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "availability": availability,
        "false_discovery_rate": fdr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def calculate_reliability_metrics(y_test, y_pred, time_intervals, latencies):
    """
    Calculate all cloud reliability metrics: MTTD, MTTR, Availability

    Args:
        y_test: True labels
        y_pred: Predicted labels
        time_intervals: Time to process each block
        latencies: Latency for each block

    Returns:
        metrics: Dictionary with all reliability metrics
    """
    print("\n" + "=" * 70)
    print("CLOUD RELIABILITY METRICS")
    print("=" * 70)

    # MTTD
    mttd = calculate_mttd(y_test, y_pred, time_intervals)
    print("\n--- Mean Time to Detect (MTTD) ---")
    print(f"Mean:            {mttd['mean']:.2f} time units")
    print(f"Median:          {mttd['median']:.2f} time units")
    print(f"95th percentile: {mttd['p95']:.2f} time units")
    print(f"Range:           [{mttd['min']:.2f}, {mttd['max']:.2f}]")
    print(
        f"Detected:        {mttd['detected_count']} / {mttd['total_anomalies']} anomalies"
    )

    # MTTR
    mttr = calculate_mttr(y_test, y_pred, latencies)
    print("\n--- Mean Time to Resolve (MTTR) ---")
    print(f"Mean:            {mttr['mean']:.2f} time units")
    print(f"Median:          {mttr['median']:.2f} time units")
    print(f"95th percentile: {mttr['p95']:.2f} time units")
    print(f"Range:           [{mttr['min']:.2f}, {mttr['max']:.2f}]")

    # MTBF
    mtbf = mttd["mean"] + mttr["mean"]
    print(f"\n--- Mean Time Between Failures (MTBF) ---")
    print(f"MTBF: {mtbf:.2f} time units (MTTD + MTTR)")

    # Availability
    availability = calculate_availability(y_test, y_pred)
    print(f"\n--- System Health & Availability ---")
    print(
        f"Detection Rate:      {availability['detection_rate']:.4f} ({availability['detection_rate']*100:.2f}%)"
    )
    print(
        f"False Alarm Rate:    {availability['false_alarm_rate']:.4f} ({availability['false_alarm_rate']*100:.2f}%)"
    )
    print(
        f"System Availability: {availability['availability']:.4f} ({availability['availability']*100:.2f}%)"
    )
    print(f"False Discovery Rate: {availability['false_discovery_rate']:.4f}")

    print(f"\n--- Operational Impact ---")
    print(f"True Positives (Caught):     {availability['true_positives']:6d}")
    print(f"False Negatives (Missed):    {availability['false_negatives']:6d}")
    print(f"False Positives (Alarms):    {availability['false_positives']:6d}")
    print(f"True Negatives (Correct):    {availability['true_negatives']:6d}")

    return {"mttd": mttd, "mttr": mttr, "mtbf": mtbf, "availability": availability}


def print_reliability_summary(metrics):
    """
    Print a concise summary of reliability metrics

    Args:
        metrics: Dictionary from calculate_reliability_metrics
    """
    print("\n" + "=" * 70)
    print("RELIABILITY METRICS SUMMARY")
    print("=" * 70)

    print(f"\n🎯 Detection Performance:")
    print(f"   Detection Rate: {metrics['availability']['detection_rate']*100:.2f}%")
    print(f"   Missed Anomalies: {metrics['availability']['false_negatives']}")

    print(f"\n⏱️  Time Metrics:")
    print(f"   MTTD (Mean):  {metrics['mttd']['mean']:.2f} time units")
    print(f"   MTTR (Mean):  {metrics['mttr']['mean']:.2f} time units")
    print(f"   MTBF:         {metrics['mtbf']:.2f} time units")

    print(f"\n📊 System Availability:")
    print(f"   Availability: {metrics['availability']['availability']*100:.2f}%")
    print(f"   False Alarms: {metrics['availability']['false_alarm_rate']*100:.2f}%")


if __name__ == "__main__":
    print("Reliability Metrics Module Loaded!")
    print("\nUsage:")
    print(
        "  metrics = calculate_reliability_metrics(y_test, y_pred, time_intervals, latencies)"
    )
    print("  print_reliability_summary(metrics)")
