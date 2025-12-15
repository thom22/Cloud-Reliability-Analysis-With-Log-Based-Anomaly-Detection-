"""
Cloud Reliability Monitor - HDFS Log Anomaly Detection Dashboard
================================================================
A comprehensive Streamlit dashboard for showcasing both supervised 
and unsupervised ML model results with real-time log simulation.

Run with: streamlit run streamlit_anomaly_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
from collections import deque
import json

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Cloud Reliability Monitor - HDFS Anomaly Detection",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING (GCP-Inspired Theme)
# ============================================================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --gcp-blue: #4285f4;
        --gcp-green: #34a853;
        --gcp-yellow: #fbbc04;
        --gcp-red: #ea4335;
        --bg-primary: #0e1117;
        --bg-secondary: #1a1d24;
        --text-primary: #e6eaf0;
        --text-secondary: #9aa0a6;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1d24 0%, #0e1117 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #2d3748;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4285f4, #34a853, #fbbc04, #ea4335);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #9aa0a6;
        font-size: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #4285f4;
        box-shadow: 0 6px 20px rgba(66, 133, 244, 0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e6eaf0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    .metric-change {
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    
    .positive { color: #34a853; }
    .negative { color: #ea4335; }
    .warning { color: #fbbc04; }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-healthy {
        background: rgba(52, 168, 83, 0.15);
        color: #34a853;
        border: 1px solid rgba(52, 168, 83, 0.3);
    }
    
    .status-warning {
        background: rgba(251, 188, 4, 0.15);
        color: #fbbc04;
        border: 1px solid rgba(251, 188, 4, 0.3);
    }
    
    .status-critical {
        background: rgba(234, 67, 53, 0.15);
        color: #ea4335;
        border: 1px solid rgba(234, 67, 53, 0.3);
    }
    
    /* Log stream styling */
    .log-container {
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        max-height: 350px;
        overflow-y: auto;
    }
    
    .log-entry {
        padding: 4px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    
    .log-timestamp { color: #608b4e; }
    .log-info { color: #4fc1ff; }
    .log-warning { color: #fbbc04; }
    .log-error { color: #ea4335; }
    .log-anomaly { color: #ff00ff; font-weight: bold; }
    .log-blockid { color: #dcdcaa; }
    
    /* Model comparison cards */
    .model-card {
        background: linear-gradient(135deg, #1a1d24 0%, #252a34 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .model-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6eaf0;
        margin-bottom: 1rem;
    }
    
    /* Anomaly alert */
    .anomaly-alert {
        background: linear-gradient(135deg, rgba(234, 67, 53, 0.15) 0%, rgba(234, 67, 53, 0.08) 100%);
        border: 1px solid rgba(234, 67, 53, 0.4);
        border-left: 4px solid #ea4335;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    .alert-time {
        font-size: 0.8rem;
        color: #5f6368;
        font-weight: 500;
    }
    
    .alert-block {
        font-weight: 700;
        color: #c5221f;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 1.1rem;
        margin: 0.3rem 0;
    }
    
    .alert-type {
        color: #ea4335;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e6eaf0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2d3748;
    }
    
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1d24;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4285f4;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a9bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'log_history' not in st.session_state:
    st.session_state.log_history = deque(maxlen=50)
    
if 'anomaly_history' not in st.session_state:
    st.session_state.anomaly_history = deque(maxlen=10)
    
if 'total_blocks' not in st.session_state:
    st.session_state.total_blocks = 547832
    
if 'anomaly_count' not in st.session_state:
    st.session_state.anomaly_count = 23
    
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = {
        'timestamps': [],
        'normal': [],
        'anomaly': []
    }

if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = True

# Real-time analytics data
if 'realtime_event_counts' not in st.session_state:
    st.session_state.realtime_event_counts = {
        'timestamps': [],
        'E3': [], 'E5': [], 'E7': [], 'E20': [], 'E22': []
    }

if 'realtime_model_scores' not in st.session_state:
    st.session_state.realtime_model_scores = {
        'timestamps': [],
        'isolation_forest': [],
        'autoencoder': [],
        'xgboost': []
    }

if 'realtime_detections' not in st.session_state:
    st.session_state.realtime_detections = {
        'timestamps': [],
        'true_positives': [],
        'false_positives': [],
        'precision': [],
        'recall': []
    }

if 'analytics_streaming' not in st.session_state:
    st.session_state.analytics_streaming = False

if 'cumulative_stats' not in st.session_state:
    st.session_state.cumulative_stats = {
        'total_processed': 575061,
        'total_anomalies': 16838,
        'true_positives': 15234,
        'false_positives': 1245,
        'false_negatives': 1604
    }

# ============================================================================
# DATA AND MODEL DEFINITIONS
# ============================================================================

# Event templates from HDFS logs
EVENT_TEMPLATES = {
    'E1': 'Adding an already existing block',
    'E2': 'Verification succeeded for block',
    'E3': 'Served block to destination',
    'E4': 'Got exception while serving block',
    'E5': 'Receiving block src dest',
    'E6': 'Received block of size',
    'E7': 'writeBlock received exception',
    'E8': 'PacketResponder Interrupted',
    'E9': 'Received block from source',
    'E10': 'PacketResponder Exception',
    'E11': 'PacketResponder terminating',
    'E12': 'Exception writing block to mirror',
    'E13': 'Receiving empty packet for block',
    'E14': 'Exception in receiveBlock',
    'E15': 'Changing block file offset',
    'E16': 'Transmitted block successfully',
    'E17': 'Failed to transfer block',
    'E18': 'Starting thread to transfer block',
    'E19': 'Reopen Block',
    'E20': 'Unexpected error deleting block',
    'E21': 'Deleting block file',
    'E22': 'BLOCK* NameSystem allocateBlock',
    'E23': 'BLOCK* delete added to invalidSet',
    'E24': 'BLOCK* Removing from neededReplications',
    'E25': 'BLOCK* ask to replicate',
    'E26': 'BLOCK* blockMap updated',
    'E27': 'BLOCK* Redundant addStoredBlock',
    'E28': 'BLOCK* addStoredBlock no file',
    'E29': 'PendingReplicationMonitor timeout'
}

# Log message templates
LOG_TEMPLATES = [
    {'level': 'INFO', 'message': 'Receiving block {blockId} src: /10.0.1.{s1}:50010 dest: /10.0.1.{s2}:50010', 'event': 'E5'},
    {'level': 'INFO', 'message': 'Served block {blockId} to /10.0.1.{s1}', 'event': 'E3'},
    {'level': 'INFO', 'message': 'PacketResponder for block {blockId} terminating', 'event': 'E11'},
    {'level': 'INFO', 'message': 'BLOCK* NameSystem.allocateBlock: {blockId}', 'event': 'E22'},
    {'level': 'INFO', 'message': 'Transmitted block {blockId} successfully', 'event': 'E16'},
    {'level': 'INFO', 'message': 'Verification succeeded for {blockId}', 'event': 'E2'},
    {'level': 'WARNING', 'message': 'Verification failed for {blockId}', 'event': 'E4'},
    {'level': 'ERROR', 'message': 'Exception writing block {blockId} to mirror', 'event': 'E12'},
]

ANOMALY_TEMPLATES = [
    {'level': 'ANOMALY', 'message': 'DETECTED: Unusual sequence E7→E14→E10 for block {blockId}', 'pattern': 'Critical Failure Pattern'},
    {'level': 'ANOMALY', 'message': 'DETECTED: High latency pattern for block {blockId}', 'pattern': 'High Latency'},
    {'level': 'ANOMALY', 'message': 'DETECTED: Missing replication for block {blockId}', 'pattern': 'Missing Replication'},
    {'level': 'ANOMALY', 'message': 'DETECTED: Unexpected termination for block {blockId}', 'pattern': 'Unexpected Termination'},
]

# Model performance data (from actual notebook results)
SUPERVISED_MODELS = {
    'Random Forest': {
        'accuracy': 0.9892,
        'precision': 0.9456,
        'recall': 0.9234,
        'f1': 0.9344,
        'roc_auc': 0.9876,
        'description': 'Ensemble of decision trees with balanced class weights',
        'color': '#4285f4'
    },
    'XGBoost': {
        'accuracy': 0.9951,
        'precision': 0.9418,
        'recall': 0.8913,
        'f1': 0.9159,
        'roc_auc': 0.9982,
        'description': 'Rigorously tested gradient boosting (5-fold CV validated)',
        'color': '#34a853',
        'is_best': True,
        'confusion_matrix': {
            'tn': 22272, 'fp': 38,
            'fn': 75, 'tp': 615
        },
        'analysis': {
            'cv_f1_mean': 0.9105,
            'cv_f1_std': 0.0115,
            'stability': 'High (std=0.0028 across splits)',
            'strengths': [
                'Low false positive rate (38/23,000 = 0.17%)',
                'Strong precision (94.2%) - few false alarms',
                'Excellent ROC-AUC (0.998) - great ranking',
                'Stable across different data splits'
            ],
            'limitations': [
                'Misses 75 anomalies (10.9% false negative rate)',
                'Recall at 89.1% - room for improvement',
                'Trade-off: fewer false alarms but some missed detections'
            ],
            'recommendation': 'Best for production where false alarms are costly'
        }
    },
    'LightGBM': {
        'accuracy': 0.9889,
        'precision': 0.9412,
        'recall': 0.9289,
        'f1': 0.9350,
        'roc_auc': 0.9889,
        'description': 'Leaf-wise gradient boosting for efficiency',
        'color': '#fbbc04'
    },
    'Gradient Boosting': {
        'accuracy': 0.9867,
        'precision': 0.9345,
        'recall': 0.9156,
        'f1': 0.9250,
        'roc_auc': 0.9845,
        'description': 'Sklearn gradient boosting classifier',
        'color': '#ea4335'
    },
    'Neural Network (MLP)': {
        'accuracy': 0.9834,
        'precision': 0.9234,
        'recall': 0.9089,
        'f1': 0.9161,
        'roc_auc': 0.9812,
        'description': '3-layer neural network (128-64-32)',
        'color': '#9c27b0'
    },
    'Logistic Regression': {
        'accuracy': 0.9756,
        'precision': 0.8934,
        'recall': 0.8756,
        'f1': 0.8844,
        'roc_auc': 0.9678,
        'description': 'Baseline linear model with class balancing',
        'color': '#00bcd4'
    }
}

UNSUPERVISED_MODELS = {
    'Isolation Forest': {
        'precision': 0.5887,
        'recall': 0.5924,
        'f1': 0.5906,
        'roc_auc': 0.9630,
        'contamination': 0.03,
        'description': 'Tree-based isolation of anomalies',
        'color': '#ff5722',
        'confusion_matrix': {
            'tn': 110278, 'fp': 1386,
            'fn': 1365, 'tp': 1984
        }
    },
    'Autoencoder': {
        'precision': 0.6234,
        'recall': 0.5812,
        'f1': 0.6015,
        'roc_auc': 0.9542,
        'threshold_percentile': 95,
        'description': 'Deep learning reconstruction-based detection',
        'color': '#e91e63',
        'confusion_matrix': {
            'tn': 108956, 'fp': 2708,
            'fn': 1401, 'tp': 1948
        }
    }
}

# Feature importance data
FEATURE_IMPORTANCE = {
    'E20': 0.156, 'E7': 0.134, 'E6': 0.098, 'E25': 0.087,
    'E18': 0.076, 'E16': 0.065, 'E14': 0.054, 'E13': 0.048,
    'E28': 0.042, 'E27': 0.038, 'E10': 0.035, 'E12': 0.032,
    'E22': 0.028, 'E5': 0.025, 'E3': 0.022
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_block_id():
    """Generate a realistic HDFS block ID"""
    return f"blk_-{random.randint(1000000000, 9999999999)}{random.randint(100, 999)}"

def get_timestamp():
    """Get current timestamp formatted"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def generate_log_entry(is_anomaly=False):
    """Generate a simulated log entry"""
    block_id = generate_block_id()
    timestamp = get_timestamp()
    
    if is_anomaly:
        template = random.choice(ANOMALY_TEMPLATES)
        level = 'ANOMALY'
        message = template['message'].format(blockId=block_id)
        pattern = template['pattern']
    else:
        template = random.choice(LOG_TEMPLATES)
        level = template['level']
        message = template['message'].format(
            blockId=block_id,
            s1=random.randint(1, 254),
            s2=random.randint(1, 254)
        )
        pattern = None
    
    return {
        'timestamp': timestamp,
        'level': level,
        'message': message,
        'block_id': block_id,
        'pattern': pattern
    }

def render_metric_card(value, label, change=None, change_type='positive'):
    """Render a styled metric card"""
    change_html = ""
    if change:
        arrow = "↑" if change_type in ['positive', 'negative-up'] else "↓"
        color_class = change_type.split('-')[0] if '-' in change_type else change_type
        change_html = f'<div class="metric-change {color_class}">{arrow} {change}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {change_html}
    </div>
    """

def render_status_indicator(status, label):
    """Render a status indicator badge"""
    status_class = f"status-{status}"
    dot = "●"
    return f'<span class="status-indicator {status_class}">{dot} {label}</span>'

def render_log_entry(entry):
    """Render a formatted log entry"""
    level_class = f"log-{entry['level'].lower()}"
    return f"""
    <div class="log-entry">
        <span class="log-timestamp">{entry['timestamp']}</span>
        <span class="{level_class}">[{entry['level']:8s}]</span>
        <span>{entry['message']}</span>
    </div>
    """

def render_anomaly_alert(entry):
    """Render an anomaly alert card"""
    return f"""
    <div class="anomaly-alert">
        <div class="alert-time">{entry['timestamp']}</div>
        <div class="alert-block">{entry['block_id']}</div>
        <div class="alert-type">{entry['pattern']}</div>
    </div>
    """

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_model_comparison_chart():
    """Create interactive model comparison bar chart"""
    models = list(SUPERVISED_MODELS.keys())
    metrics = ['precision', 'recall', 'f1', 'roc_auc']
    
    fig = go.Figure()
    
    colors = ['#4285f4', '#34a853', '#fbbc04', '#ea4335']
    
    for i, metric in enumerate(metrics):
        values = [SUPERVISED_MODELS[m][metric] for m in models]
        fig.add_trace(go.Bar(
            name=metric.upper().replace('_', '-'),
            x=models,
            y=values,
            marker_color=colors[i],
            text=[f'{v:.3f}' for v in values],
            textposition='outside',
            textfont=dict(size=10)
        ))
    
    fig.update_layout(
        title=dict(text='Supervised Model Performance Comparison', font=dict(size=16)),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(tickangle=-30, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(range=[0, 1.1], gridcolor='rgba(255,255,255,0.1)'),
        height=450
    )
    
    return fig

def create_unsupervised_comparison_chart():
    """Create unsupervised model comparison"""
    fig = go.Figure()
    
    for model_name, model_data in UNSUPERVISED_MODELS.items():
        fig.add_trace(go.Scatterpolar(
            r=[model_data['precision'], model_data['recall'], 
               model_data['f1'], model_data['roc_auc']],
            theta=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            fill='toself',
            name=model_name,
            line=dict(color=model_data['color'])
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.2)'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        title=dict(text='Unsupervised Model Comparison', font=dict(size=16)),
        height=400
    )
    
    return fig

def create_roc_curves():
    """Create ROC curves for all models"""
    fig = go.Figure()
    
    # Generate synthetic ROC curve data based on AUC values
    np.random.seed(42)
    
    all_models = {**SUPERVISED_MODELS, **UNSUPERVISED_MODELS}
    
    for model_name, model_data in all_models.items():
        auc = model_data['roc_auc']
        
        # Generate plausible ROC curve points
        n_points = 100
        fpr = np.linspace(0, 1, n_points)
        
        # Generate TPR based on AUC using a beta distribution-like curve
        tpr = np.power(fpr, (1 - auc) / auc)
        tpr = np.clip(tpr + np.random.normal(0, 0.02, n_points), 0, 1)
        tpr = np.sort(tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={auc:.3f})',
            line=dict(color=model_data['color'], width=2)
        ))
    
    # Add diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random (AUC=0.500)',
        line=dict(color='gray', dash='dash', width=1)
    ))
    
    fig.update_layout(
        title=dict(text='ROC Curves - All Models', font=dict(size=16)),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        legend=dict(x=1.05, y=0.5),
        height=450
    )
    
    return fig

def create_confusion_matrix_heatmap(model_name, cm_data):
    """Create confusion matrix heatmap"""
    cm = np.array([[cm_data['tn'], cm_data['fp']], 
                   [cm_data['fn'], cm_data['tp']]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Anomaly'],
        y=['True Normal', 'True Anomaly'],
        text=[[f'{cm[i][j]:,}' for j in range(2)] for i in range(2)],
        texttemplate='%{text}',
        textfont={"size": 16},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title=dict(text=f'{model_name} - Confusion Matrix', font=dict(size=14)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        height=300
    )
    
    return fig

def create_feature_importance_chart():
    """Create feature importance bar chart"""
    sorted_features = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features[:15]]
    importance = [f[1] for f in sorted_features[:15]]
    
    # Create color gradient
    colors = [f'rgba(66, 133, 244, {0.4 + (i * 0.04)})' for i in range(len(features))][::-1]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text='Feature Importance (Random Forest)', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Importance'),
        yaxis=dict(categoryorder='total ascending'),
        height=450
    )
    
    return fig

def create_event_distribution_chart():
    """Create event type distribution chart"""
    events = ['E3', 'E5', 'E22', 'E26', 'E16', 'E11', 'E2', 'E9']
    normal_counts = [203, 156, 89, 76, 65, 58, 52, 48]
    anomaly_counts = [45, 32, 28, 21, 18, 15, 12, 10]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Normal',
        x=events,
        y=normal_counts,
        marker_color='#34a853'
    ))
    
    fig.add_trace(go.Bar(
        name='Anomaly',
        x=events,
        y=anomaly_counts,
        marker_color='#ea4335'
    ))
    
    fig.update_layout(
        title=dict(text='Event Distribution by Class', font=dict(size=16)),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Event Type'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Count'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=350
    )
    
    return fig

def create_realtime_chart():
    """Create real-time anomaly detection chart"""
    if len(st.session_state.chart_data['timestamps']) < 2:
        # Initialize with some data
        now = datetime.now()
        for i in range(30):
            ts = now - timedelta(seconds=(30-i)*2)
            st.session_state.chart_data['timestamps'].append(ts)
            st.session_state.chart_data['normal'].append(random.randint(150, 200))
            st.session_state.chart_data['anomaly'].append(random.randint(0, 5) + (3 if random.random() < 0.2 else 0))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=st.session_state.chart_data['timestamps'],
        y=st.session_state.chart_data['normal'],
        mode='lines',
        name='Normal Events',
        line=dict(color='#34a853', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 168, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=st.session_state.chart_data['timestamps'],
        y=st.session_state.chart_data['anomaly'],
        mode='lines+markers',
        name='Anomalies',
        line=dict(color='#ea4335', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=dict(text='Real-time Anomaly Detection', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Time'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Event Count'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=300
    )
    
    return fig

def create_precision_recall_chart():
    """Create precision-recall curves"""
    fig = go.Figure()
    
    np.random.seed(42)
    
    all_models = {**SUPERVISED_MODELS, **UNSUPERVISED_MODELS}
    
    for model_name, model_data in all_models.items():
        prec = model_data['precision']
        rec = model_data['recall']
        
        # Generate plausible PR curve
        n_points = 50
        recall = np.linspace(0, 1, n_points)
        
        # Approximate precision curve
        precision = prec * np.exp(-((recall - rec) ** 2) / (2 * 0.3 ** 2))
        precision = np.clip(precision + np.random.normal(0, 0.02, n_points), 0, 1)
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=model_name,
            line=dict(color=model_data['color'], width=2)
        ))
    
    fig.update_layout(
        title=dict(text='Precision-Recall Curves', font=dict(size=16)),
        xaxis_title='Recall',
        yaxis_title='Precision',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 1]),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', range=[0, 1]),
        legend=dict(x=1.05, y=0.5),
        height=400
    )
    
    return fig

# ============================================================================
# REAL-TIME ANALYTICS FUNCTIONS
# ============================================================================

def update_realtime_analytics_data():
    """Update real-time analytics data with simulated values"""
    now = datetime.now()
    
    # Update event counts
    st.session_state.realtime_event_counts['timestamps'].append(now)
    st.session_state.realtime_event_counts['E3'].append(random.randint(180, 220))
    st.session_state.realtime_event_counts['E5'].append(random.randint(140, 170))
    st.session_state.realtime_event_counts['E7'].append(random.randint(5, 25))
    st.session_state.realtime_event_counts['E20'].append(random.randint(2, 15))
    st.session_state.realtime_event_counts['E22'].append(random.randint(70, 100))
    
    # Keep last 60 points
    for key in st.session_state.realtime_event_counts:
        if len(st.session_state.realtime_event_counts[key]) > 60:
            st.session_state.realtime_event_counts[key] = st.session_state.realtime_event_counts[key][-60:]
    
    # Update model scores
    st.session_state.realtime_model_scores['timestamps'].append(now)
    st.session_state.realtime_model_scores['isolation_forest'].append(
        0.59 + random.uniform(-0.03, 0.03)
    )
    st.session_state.realtime_model_scores['autoencoder'].append(
        0.60 + random.uniform(-0.03, 0.03)
    )
    st.session_state.realtime_model_scores['xgboost'].append(
        0.94 + random.uniform(-0.02, 0.02)
    )
    
    # Keep last 60 points
    for key in st.session_state.realtime_model_scores:
        if len(st.session_state.realtime_model_scores[key]) > 60:
            st.session_state.realtime_model_scores[key] = st.session_state.realtime_model_scores[key][-60:]
    
    # Update detection metrics
    st.session_state.realtime_detections['timestamps'].append(now)
    
    # Simulate detection events
    new_tp = random.randint(8, 15)
    new_fp = random.randint(0, 3)
    
    st.session_state.realtime_detections['true_positives'].append(new_tp)
    st.session_state.realtime_detections['false_positives'].append(new_fp)
    
    # Calculate running precision/recall
    total_tp = sum(st.session_state.realtime_detections['true_positives'][-20:])
    total_fp = sum(st.session_state.realtime_detections['false_positives'][-20:])
    total_fn = int(total_tp * 0.07)  # ~7% false negative rate
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    st.session_state.realtime_detections['precision'].append(precision)
    st.session_state.realtime_detections['recall'].append(recall)
    
    # Keep last 60 points
    for key in st.session_state.realtime_detections:
        if len(st.session_state.realtime_detections[key]) > 60:
            st.session_state.realtime_detections[key] = st.session_state.realtime_detections[key][-60:]
    
    # Update cumulative stats
    st.session_state.cumulative_stats['total_processed'] += random.randint(10, 50)
    if random.random() < 0.1:  # 10% chance of anomaly
        st.session_state.cumulative_stats['total_anomalies'] += 1
        if random.random() < 0.93:  # 93% detection rate
            st.session_state.cumulative_stats['true_positives'] += 1
        else:
            st.session_state.cumulative_stats['false_negatives'] += 1
    if random.random() < 0.02:  # 2% false positive rate
        st.session_state.cumulative_stats['false_positives'] += 1

def create_realtime_event_chart():
    """Create real-time event frequency chart"""
    data = st.session_state.realtime_event_counts
    
    if len(data['timestamps']) < 2:
        # Initialize with some data
        now = datetime.now()
        for i in range(30):
            ts = now - timedelta(seconds=(30-i)*2)
            data['timestamps'].append(ts)
            data['E3'].append(random.randint(180, 220))
            data['E5'].append(random.randint(140, 170))
            data['E7'].append(random.randint(5, 25))
            data['E20'].append(random.randint(2, 15))
            data['E22'].append(random.randint(70, 100))
    
    fig = go.Figure()
    
    colors = {'E3': '#34a853', 'E5': '#4285f4', 'E7': '#ea4335', 'E20': '#ff5722', 'E22': '#9c27b0'}
    
    for event in ['E3', 'E5', 'E22', 'E7', 'E20']:
        fig.add_trace(go.Scatter(
            x=data['timestamps'],
            y=data[event],
            mode='lines',
            name=f'{event}: {EVENT_TEMPLATES.get(event, "")[:30]}...',
            line=dict(color=colors[event], width=2)
        ))
    
    fig.update_layout(
        title=dict(text='📊 Real-time Event Frequency', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Time'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Events/sec'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=350
    )
    
    return fig

def create_realtime_model_performance_chart():
    """Create real-time model F1-score tracking chart"""
    data = st.session_state.realtime_model_scores
    
    if len(data['timestamps']) < 2:
        now = datetime.now()
        for i in range(30):
            ts = now - timedelta(seconds=(30-i)*2)
            data['timestamps'].append(ts)
            data['isolation_forest'].append(0.59 + random.uniform(-0.03, 0.03))
            data['autoencoder'].append(0.60 + random.uniform(-0.03, 0.03))
            data['xgboost'].append(0.94 + random.uniform(-0.02, 0.02))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['xgboost'],
        mode='lines',
        name='XGBoost (Supervised)',
        line=dict(color='#34a853', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 168, 83, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['autoencoder'],
        mode='lines',
        name='Autoencoder (Unsupervised)',
        line=dict(color='#e91e63', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['isolation_forest'],
        mode='lines',
        name='Isolation Forest (Unsupervised)',
        line=dict(color='#ff5722', width=2)
    ))
    
    fig.update_layout(
        title=dict(text='🤖 Real-time Model F1-Score Tracking', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Time'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='F1-Score', range=[0, 1]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        height=350
    )
    
    return fig

def create_realtime_detection_chart():
    """Create real-time detection metrics chart"""
    data = st.session_state.realtime_detections
    
    if len(data['timestamps']) < 2:
        now = datetime.now()
        for i in range(30):
            ts = now - timedelta(seconds=(30-i)*2)
            data['timestamps'].append(ts)
            tp = random.randint(8, 15)
            fp = random.randint(0, 3)
            data['true_positives'].append(tp)
            data['false_positives'].append(fp)
            data['precision'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
            data['recall'].append(tp / (tp + int(tp * 0.07)) if tp > 0 else 0)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Detection Counts', 'Precision & Recall'),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Detection counts
    fig.add_trace(go.Bar(
        x=data['timestamps'][-20:],
        y=data['true_positives'][-20:],
        name='True Positives',
        marker_color='#34a853'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=data['timestamps'][-20:],
        y=data['false_positives'][-20:],
        name='False Positives',
        marker_color='#ea4335'
    ), row=1, col=1)
    
    # Precision/Recall lines
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['precision'],
        mode='lines',
        name='Precision',
        line=dict(color='#4285f4', width=2)
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=data['timestamps'],
        y=data['recall'],
        mode='lines',
        name='Recall',
        line=dict(color='#fbbc04', width=2)
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(text='📈 Real-time Detection Metrics', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        barmode='stack',
        legend=dict(orientation='h', yanchor='bottom', y=1.05),
        height=450
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    return fig

def create_realtime_feature_importance_chart():
    """Create animated feature importance with real-time variation"""
    # Add small random variations to simulate real-time updates
    base_importance = FEATURE_IMPORTANCE.copy()
    
    for key in base_importance:
        variation = random.uniform(-0.01, 0.01)
        base_importance[key] = max(0, base_importance[key] + variation)
    
    sorted_features = sorted(base_importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features[:10]]
    importance = [f[1] for f in sorted_features[:10]]
    
    # Color based on whether value increased or decreased
    colors = []
    for f in features:
        if base_importance[f] > FEATURE_IMPORTANCE.get(f, 0):
            colors.append('#34a853')  # Green for increase
        elif base_importance[f] < FEATURE_IMPORTANCE.get(f, 0):
            colors.append('#ea4335')  # Red for decrease
        else:
            colors.append('#4285f4')  # Blue for stable
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f'{v:.3f}' for v in importance],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(text=f'🎯 Live Feature Importance (Updated: {datetime.now().strftime("%H:%M:%S")})', font=dict(size=16)),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6eaf0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', title='Importance'),
        yaxis=dict(categoryorder='total ascending'),
        height=400
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ========== HEADER ==========
    st.markdown("""
    <div class="main-header">
        <div class="main-title">☁️ Cloud Reliability Monitor</div>
        <div class="subtitle">HDFS Log Anomaly Detection Dashboard | Real-time ML Model Monitoring</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        # Navigation
        page = st.radio(
            "📊 Select View",
            ["🏠 Overview", "📝 Log Stream", "🤖 Model Analysis", "📈 Deep Analytics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System Status
        st.markdown("### 🔗 System Status")
        st.markdown(render_status_indicator('healthy', 'Pub/Sub: Connected'), unsafe_allow_html=True)
        st.markdown(render_status_indicator('healthy', 'BigQuery: Active'), unsafe_allow_html=True)
        st.markdown(render_status_indicator('warning', 'ML Model: Processing'), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Streaming controls
        st.markdown("### 🔄 Stream Controls")
        streaming = st.toggle("Enable Real-time Stream", value=st.session_state.is_streaming)
        st.session_state.is_streaming = streaming
        
        refresh_rate = st.slider("Refresh Rate (sec)", 1, 10, 2)
        anomaly_rate = st.slider("Anomaly Rate (%)", 1, 30, 10)
        
        st.markdown("---")
        
        # Model selection
        st.markdown("### 🎯 Model Selection")
        selected_supervised = st.multiselect(
            "Supervised Models",
            list(SUPERVISED_MODELS.keys()),
            default=['XGBoost', 'Random Forest']
        )
        
        selected_unsupervised = st.multiselect(
            "Unsupervised Models",
            list(UNSUPERVISED_MODELS.keys()),
            default=['Isolation Forest', 'Autoencoder']
        )
        
        st.markdown("---")
        
        # Dataset info
        st.markdown("### 📁 Dataset Info")
        st.markdown("""
        **Source:** HDFS LogHub Dataset  
        **Total Samples:** 575,061  
        **Anomaly Rate:** 2.93%  
        **Features:** 37 engineered  
        """)
    
    # ========== MAIN CONTENT ==========
    
    if page == "🏠 Overview":
        render_overview_page(anomaly_rate)
        
    elif page == "📝 Log Stream":
        render_log_stream_page(anomaly_rate, refresh_rate)
        
    elif page == "🤖 Model Analysis":
        render_model_analysis_page(selected_supervised, selected_unsupervised)
        
    elif page == "📈 Deep Analytics":
        render_deep_analytics_page()

def render_overview_page(anomaly_rate):
    """Render the overview dashboard page"""
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(render_metric_card(
            f"{st.session_state.total_blocks:,}",
            "Total Blocks Processed",
            "12% from last hour",
            "positive"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(render_metric_card(
            str(st.session_state.anomaly_count),
            "Anomalies Detected",
            "35% from baseline",
            "negative"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card(
            "1.2s",
            "Mean Time to Detect",
            "15% improvement",
            "positive"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card(
            "98.7%",
            "Final Model Accuracy",
            "positive"
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(create_realtime_chart(), use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">⚠️ Recent Anomalies</div>', unsafe_allow_html=True)
        
        # Generate some initial anomalies if empty
        if len(st.session_state.anomaly_history) == 0:
            for _ in range(5):
                entry = generate_log_entry(is_anomaly=True)
                st.session_state.anomaly_history.append(entry)
        
        for entry in list(st.session_state.anomaly_history)[:5]:
            st.markdown(render_anomaly_alert(entry), unsafe_allow_html=True)
    
    # Model comparison quick view
    st.markdown('<div class="section-header">🤖 Model Performance Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_model_comparison_chart(), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_unsupervised_comparison_chart(), use_container_width=True)

def render_log_stream_page(anomaly_rate, refresh_rate):
    """Render the live log stream page"""
    
    st.markdown('<div class="section-header">📝 Real-time HDFS Log Stream</div>', unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("▶️ Start Stream" if not st.session_state.is_streaming else "⏸️ Pause Stream"):
            st.session_state.is_streaming = not st.session_state.is_streaming
    
    with col2:
        if st.button("🗑️ Clear Logs"):
            st.session_state.log_history.clear()
    
    with col3:
        log_filter = st.selectbox("Filter", ["All", "INFO", "WARNING", "ERROR", "ANOMALY"])
    
    with col4:
        st.markdown(f"**Logs:** {len(st.session_state.log_history)} | **Anomaly Rate:** {anomaly_rate}%")
    
    # Log stream container
    log_container = st.empty()
    metrics_container = st.empty()
    chart_container = st.empty()
    
    # Simulation loop
    if st.session_state.is_streaming:
        # Generate new log entries
        is_anomaly = random.random() < (anomaly_rate / 100)
        new_entry = generate_log_entry(is_anomaly)
        st.session_state.log_history.appendleft(new_entry)
        
        if is_anomaly:
            st.session_state.anomaly_history.appendleft(new_entry)
            st.session_state.anomaly_count += 1
        
        st.session_state.total_blocks += 1
        
        # Update chart data
        now = datetime.now()
        st.session_state.chart_data['timestamps'].append(now)
        st.session_state.chart_data['normal'].append(random.randint(150, 200))
        st.session_state.chart_data['anomaly'].append(
            random.randint(0, 5) + (5 if is_anomaly else 0)
        )
        
        # Keep last 60 points
        for key in st.session_state.chart_data:
            if len(st.session_state.chart_data[key]) > 60:
                st.session_state.chart_data[key] = st.session_state.chart_data[key][-60:]
    
    # Display logs
    filtered_logs = st.session_state.log_history
    if log_filter != "All":
        filtered_logs = [l for l in filtered_logs if l['level'] == log_filter]
    
    log_html = '<div class="log-container">'
    for entry in list(filtered_logs)[:30]:
        log_html += render_log_entry(entry)
    log_html += '</div>'
    
    log_container.markdown(log_html, unsafe_allow_html=True)
    
    # Metrics
    with metrics_container.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Blocks", f"{st.session_state.total_blocks:,}", "+12")
        with col2:
            st.metric("Anomalies", st.session_state.anomaly_count, "+1" if is_anomaly else None)
        with col3:
            st.metric("Log Entries", len(st.session_state.log_history))
        with col4:
            st.metric("Processing Rate", "1,247/min")
    
    # Real-time chart
    chart_container.plotly_chart(create_realtime_chart(), use_container_width=True)
    
    # Auto-refresh
    if st.session_state.is_streaming:
        time.sleep(refresh_rate)
        st.rerun()

def render_model_analysis_page(selected_supervised, selected_unsupervised):
    """Render the model analysis page"""
    
    st.markdown('<div class="section-header">🤖 ML Model Analysis</div>', unsafe_allow_html=True)
    
    # Tabs for different model categories
    tab1, tab2, tab3 = st.tabs(["📊 Supervised Models", "🔍 Unsupervised Models", "⚖️ Model Comparison"])
    
    with tab1:
        st.markdown("### Supervised Learning Models")
        st.markdown("*Trained with labeled anomaly data for high-accuracy classification*")
        
        # Model cards
        for model_name in selected_supervised:
            if model_name in SUPERVISED_MODELS:
                model = SUPERVISED_MODELS[model_name]
                
                # Add "Best Model" badge for XGBoost
                title = f"**{model_name}**"
                if model.get('is_best', False):
                    title = f"**🏆 {model_name} (Best Model)**"
                
                with st.expander(f"{title} | F1: {model['f1']:.4f} | AUC: {model['roc_auc']:.4f}", expanded=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Description:** {model['description']}")
                        st.markdown("**Performance Metrics:**")
                        
                        metrics_df = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                            'Value': [model['accuracy'], model['precision'], model['recall'], 
                                     model['f1'], model['roc_auc']]
                        })
                        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                        
                        # Show confusion matrix if available
                        if 'confusion_matrix' in model:
                            st.markdown("---")
                            st.markdown("**Confusion Matrix:**")
                    
                    with col2:
                        # Create mini gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=model['f1'] * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "F1 Score", 'font': {'size': 16, 'color': '#e6eaf0'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickcolor': '#e6eaf0'},
                                'bar': {'color': model['color']},
                                'bgcolor': '#1a1d24',
                                'bordercolor': '#2d3748',
                                'steps': [
                                    {'range': [0, 50], 'color': 'rgba(234,67,53,0.3)'},
                                    {'range': [50, 80], 'color': 'rgba(251,188,4,0.3)'},
                                    {'range': [80, 100], 'color': 'rgba(52,168,83,0.3)'}
                                ]
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            font={'color': '#e6eaf0'},
                            height=250
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display confusion matrix in full width if available
                    if 'confusion_matrix' in model:
                        st.plotly_chart(
                            create_confusion_matrix_heatmap(model_name, model['confusion_matrix']),
                            use_container_width=True
                        )
                    
                    # Display analysis if available
                    if 'analysis' in model:
                        st.markdown("---")
                        st.markdown("### Model Analysis")
                        
                        analysis = model['analysis']
                        
                        # Validation info
                        if 'cv_f1_mean' in analysis:
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Cross-Validation F1", f"{analysis['cv_f1_mean']:.4f}")
                            with col_b:
                                st.metric("Stability", analysis['stability'])
                        
                        # Strengths
                        st.markdown("**✅ Strengths:**")
                        for strength in analysis['strengths']:
                            st.markdown(f"- {strength}")
                        
                        # Limitations
                        st.markdown("**⚠️ Limitations:**")
                        for limitation in analysis['limitations']:
                            st.markdown(f"- {limitation}")
                        
                        # Recommendation
                        st.info(f"**💡 Recommendation:** {analysis['recommendation']}")
    
    with tab2:
        st.markdown("### Unsupervised Learning Models")
        st.markdown("*Detect anomalies without labeled training data*")
        
        col1, col2 = st.columns(2)
        
        for i, model_name in enumerate(selected_unsupervised):
            if model_name in UNSUPERVISED_MODELS:
                model = UNSUPERVISED_MODELS[model_name]
                
                with (col1 if i % 2 == 0 else col2):
                    st.markdown(f"#### {model_name}")
                    st.markdown(f"*{model['description']}*")
                    
                    # Display metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Precision", f"{model['precision']:.4f}")
                        st.metric("Recall", f"{model['recall']:.4f}")
                    with col_b:
                        st.metric("F1-Score", f"{model['f1']:.4f}")
                        st.metric("ROC-AUC", f"{model['roc_auc']:.4f}")
                    
                    # Confusion matrix
                    st.plotly_chart(
                        create_confusion_matrix_heatmap(model_name, model['confusion_matrix']),
                        use_container_width=True
                    )
    
    with tab3:
        st.markdown("### All Models Comparison")
        
        # Combined metrics table
        all_models_data = []
        
        for name, data in SUPERVISED_MODELS.items():
            all_models_data.append({
                'Model': name,
                'Type': 'Supervised',
                'Accuracy': f"{data.get('accuracy', 0):.4f}",
                'Precision': data['precision'],
                'Recall': data['recall'],
                'F1-Score': data['f1'],
                'ROC-AUC': data['roc_auc']
            })
        
        for name, data in UNSUPERVISED_MODELS.items():
            all_models_data.append({
                'Model': name,
                'Type': 'Unsupervised',
                'Accuracy': '—',
                'Precision': data['precision'],
                'Recall': data['recall'],
                'F1-Score': data['f1'],
                'ROC-AUC': data['roc_auc']
            })
        
        comparison_df = pd.DataFrame(all_models_data)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        st.dataframe(
            comparison_df.style.background_gradient(subset=['F1-Score', 'ROC-AUC'], cmap='Greens'),
            hide_index=True,
            use_container_width=True
        )
        
        # ROC curves
        st.plotly_chart(create_roc_curves(), use_container_width=True)

def render_deep_analytics_page():
    """Render the deep analytics page with real-time data"""
    
    st.markdown('<div class="section-header">📈 Deep Analytics - Real-time Monitoring</div>', unsafe_allow_html=True)
    
    # Real-time controls at the top
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 1, 1, 1])
    
    with ctrl_col1:
        analytics_stream = st.toggle("🔴 Enable Real-time Updates", value=st.session_state.analytics_streaming, key='analytics_toggle')
        st.session_state.analytics_streaming = analytics_stream
    
    with ctrl_col2:
        analytics_refresh = st.slider("Refresh Rate (sec)", 1, 10, 3, key='analytics_refresh')
    
    with ctrl_col3:
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
    
    with ctrl_col4:
        stats = st.session_state.cumulative_stats
        current_precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
        st.metric("Live Precision", f"{current_precision:.2%}", f"+{random.uniform(0, 0.5):.2f}%")
    
    # Update data if streaming is enabled
    if st.session_state.analytics_streaming:
        update_realtime_analytics_data()
    
    # Real-time metrics row
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    stats = st.session_state.cumulative_stats
    
    with m_col1:
        st.metric(
            "📊 Total Processed", 
            f"{stats['total_processed']:,}",
            f"+{random.randint(10, 50)}" if st.session_state.analytics_streaming else None
        )
    
    with m_col2:
        st.metric(
            "⚠️ Anomalies Found", 
            f"{stats['total_anomalies']:,}",
            f"+{random.randint(0, 3)}" if st.session_state.analytics_streaming else None
        )
    
    with m_col3:
        detection_rate = stats['true_positives'] / stats['total_anomalies'] * 100 if stats['total_anomalies'] > 0 else 0
        st.metric("✅ Detection Rate", f"{detection_rate:.1f}%")
    
    with m_col4:
        fp_rate = stats['false_positives'] / stats['total_processed'] * 100 if stats['total_processed'] > 0 else 0
        st.metric("❌ False Positive Rate", f"{fp_rate:.2f}%")
    
    # Tabs with real-time content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Live Feature Analysis", 
        "📊 Real-time Events", 
        "🤖 Model Performance",
        "📉 Detection Metrics",
        "🔬 Insights & Config"
    ])
    
    with tab1:
        st.markdown("### Real-time Feature Importance Analysis")
        st.markdown("*Live tracking of which log events are contributing most to anomaly detection*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Use real-time feature importance chart
            st.plotly_chart(create_realtime_feature_importance_chart(), use_container_width=True)
        
        with col2:
            st.markdown("#### 🔥 Live Top Anomaly Indicators")
            
            # Add variation to importance values for real-time effect
            live_importance = {k: v + random.uniform(-0.01, 0.01) for k, v in FEATURE_IMPORTANCE.items()}
            sorted_live = sorted(live_importance.items(), key=lambda x: x[1], reverse=True)
            
            for event, importance in sorted_live[:8]:
                template = EVENT_TEMPLATES.get(event, 'Unknown event')
                trend = "↑" if random.random() > 0.5 else "↓"
                trend_color = "#34a853" if trend == "↑" else "#ea4335"
                st.markdown(f"""
                <div style="padding: 10px; margin: 5px 0; background: rgba(66,133,244,{min(importance, 0.3)}); 
                            border-radius: 6px; border-left: 4px solid #4285f4;">
                    <strong>{event}</strong>: {importance:.3f} <span style="color: {trend_color}">{trend}</span><br>
                    <small style="color: #9aa0a6;">{template[:40]}...</small>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Real-time Event Stream Analysis")
        st.markdown("*Live monitoring of HDFS event frequencies across different event types*")
        
        # Real-time event chart
        st.plotly_chart(create_realtime_event_chart(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Event Statistics (Live)")
            
            # Calculate live stats
            event_data = st.session_state.realtime_event_counts
            if len(event_data['timestamps']) > 0:
                live_stats = {
                    'Metric': ['Events/sec (E3)', 'Events/sec (E5)', 'Anomaly Events (E7)', 
                              'Critical Events (E20)', 'Block Allocations (E22)', 'Stream Duration'],
                    'Value': [
                        f"{np.mean(event_data['E3'][-10:]):.1f}" if len(event_data['E3']) > 0 else "0",
                        f"{np.mean(event_data['E5'][-10:]):.1f}" if len(event_data['E5']) > 0 else "0",
                        f"{np.mean(event_data['E7'][-10:]):.1f}" if len(event_data['E7']) > 0 else "0",
                        f"{np.mean(event_data['E20'][-10:]):.1f}" if len(event_data['E20']) > 0 else "0",
                        f"{np.mean(event_data['E22'][-10:]):.1f}" if len(event_data['E22']) > 0 else "0",
                        f"{len(event_data['timestamps']) * 2}s"
                    ]
                }
                st.dataframe(pd.DataFrame(live_stats), hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Key Findings (Real-time)")
            
            # Dynamic insights based on current data
            if len(event_data['E7']) > 5:
                avg_e7 = np.mean(event_data['E7'][-10:])
                avg_e20 = np.mean(event_data['E20'][-10:])
                
                if avg_e7 > 15:
                    st.error(f"🚨 High E7 (writeBlock exception) rate: {avg_e7:.1f}/sec - Investigate immediately!")
                elif avg_e7 > 10:
                    st.warning(f"⚠️ Elevated E7 rate: {avg_e7:.1f}/sec - Monitor closely")
                else:
                    st.success(f"✅ E7 rate normal: {avg_e7:.1f}/sec")
                
                if avg_e20 > 8:
                    st.error(f"🚨 High E20 (unexpected deletion) rate: {avg_e20:.1f}/sec")
                else:
                    st.info(f"ℹ️ E20 rate: {avg_e20:.1f}/sec")
            else:
                st.info("📊 Collecting real-time data... Enable streaming for live updates.")
    
    with tab3:
        st.markdown("### Real-time Model Performance Tracking")
        st.markdown("*Live F1-score comparison between supervised and unsupervised models*")
        
        # Real-time model performance chart
        st.plotly_chart(create_realtime_model_performance_chart(), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        model_data = st.session_state.realtime_model_scores
        
        with col1:
            if len(model_data['xgboost']) > 0:
                current_xgb = model_data['xgboost'][-1]
                avg_xgb = np.mean(model_data['xgboost'][-20:])
                st.metric("XGBoost F1", f"{current_xgb:.4f}", f"{(current_xgb - avg_xgb)*100:+.2f}%")
            else:
                st.metric("XGBoost F1", "0.9416")
        
        with col2:
            if len(model_data['isolation_forest']) > 0:
                current_if = model_data['isolation_forest'][-1]
                avg_if = np.mean(model_data['isolation_forest'][-20:])
                st.metric("Isolation Forest F1", f"{current_if:.4f}", f"{(current_if - avg_if)*100:+.2f}%")
            else:
                st.metric("Isolation Forest F1", "0.5906")
        
        with col3:
            if len(model_data['autoencoder']) > 0:
                current_ae = model_data['autoencoder'][-1]
                avg_ae = np.mean(model_data['autoencoder'][-20:])
                st.metric("Autoencoder F1", f"{current_ae:.4f}", f"{(current_ae - avg_ae)*100:+.2f}%")
            else:
                st.metric("Autoencoder F1", "0.6015")
        
        st.markdown("---")
        st.plotly_chart(create_roc_curves(), use_container_width=True)
    
    with tab4:
        st.markdown("### Real-time Detection Metrics")
        st.markdown("*Live tracking of true positives, false positives, precision, and recall*")
        
        # Real-time detection chart
        st.plotly_chart(create_realtime_detection_chart(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Detection Summary (Rolling 20 samples)")
            
            det_data = st.session_state.realtime_detections
            if len(det_data['true_positives']) > 0:
                total_tp = sum(det_data['true_positives'][-20:])
                total_fp = sum(det_data['false_positives'][-20:])
                avg_prec = np.mean(det_data['precision'][-20:]) if len(det_data['precision']) > 0 else 0
                avg_rec = np.mean(det_data['recall'][-20:]) if len(det_data['recall']) > 0 else 0
                
                det_stats = {
                    'Metric': ['True Positives', 'False Positives', 'Avg Precision', 'Avg Recall', 'F1-Score'],
                    'Value': [
                        f"{total_tp:,}",
                        f"{total_fp:,}",
                        f"{avg_prec:.4f}",
                        f"{avg_rec:.4f}",
                        f"{2 * avg_prec * avg_rec / (avg_prec + avg_rec):.4f}" if (avg_prec + avg_rec) > 0 else "0.0000"
                    ]
                }
                st.dataframe(pd.DataFrame(det_stats), hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Alert Status")
            
            if len(det_data['precision']) > 5:
                current_prec = det_data['precision'][-1]
                current_rec = det_data['recall'][-1]
                
                if current_prec < 0.85:
                    st.error("🚨 Precision below threshold (85%) - High false positive rate!")
                if current_rec < 0.90:
                    st.warning("⚠️ Recall below threshold (90%) - Missing some anomalies")
                if current_prec >= 0.85 and current_rec >= 0.90:
                    st.success("✅ All detection metrics within acceptable range")
            
            st.markdown("---")
            st.markdown("**Thresholds:**")
            st.markdown("- Precision target: ≥ 85%")
            st.markdown("- Recall target: ≥ 90%")
            st.markdown("- F1-Score target: ≥ 87%")
    
    with tab5:
        st.markdown("### Model Insights & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Supervised vs Unsupervised")
            
            comparison_insight = """
            | Aspect | Supervised | Unsupervised |
            |--------|------------|--------------|
            | **F1-Score** | 0.88 - 0.94 | 0.59 - 0.60 |
            | **ROC-AUC** | 0.97 - 0.99 | 0.95 - 0.96 |
            | **Labeled Data** | Required | Not Required |
            | **Novel Anomalies** | May Miss | Can Detect |
            | **Training Time** | Longer | Faster |
            """
            st.markdown(comparison_insight)
            
            st.success("""
            **✅ Recommendation:**  
            Use **XGBoost** as primary detector for production.
            Deploy **Isolation Forest** as secondary for detecting novel anomalies.
            """)
        
        with col2:
            st.markdown("#### 📈 Production Deployment Strategy")
            
            st.markdown("""
            **Hybrid Architecture:**
            
            1. **Primary Detection (XGBoost)**
               - High precision for known patterns
               - Low false positive rate
               - Fast inference (~1.2ms/sample)
            
            2. **Secondary Detection (Isolation Forest)**
               - Catches novel anomalies
               - No retraining needed
               - Unsupervised adaptation
            
            3. **Alert Prioritization**
               - Both agree → Critical Alert
               - Only supervised → High Alert
               - Only unsupervised → Medium Alert
            """)
        
        st.markdown("---")
        
        st.markdown("#### 🔧 Hyperparameter Configuration")
        
        params_col1, params_col2, params_col3 = st.columns(3)
        
        with params_col1:
            st.markdown("""
            **XGBoost**
            ```
            n_estimators: 100
            max_depth: 6
            learning_rate: 0.1
            scale_pos_weight: 33.2
            eval_metric: 'logloss'
            ```
            """)
        
        with params_col2:
            st.markdown("""
            **Isolation Forest**
            ```
            contamination: 0.03
            n_estimators: 100
            max_samples: 'auto'
            random_state: 42
            ```
            """)
        
        with params_col3:
            st.markdown("""
            **Autoencoder**
            ```
            encoding_dim: 32
            hidden_layers: [64, 32]
            threshold_percentile: 95
            epochs: 100
            ```
            """)
    
    # Auto-refresh for real-time updates
    if st.session_state.analytics_streaming:
        time.sleep(analytics_refresh)
        st.rerun()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()

