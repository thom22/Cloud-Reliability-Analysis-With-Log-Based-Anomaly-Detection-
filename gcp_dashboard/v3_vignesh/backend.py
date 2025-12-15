"""
HDFS Anomaly Detection Live Demo - Backend Server
Real-time log streaming, ML inference, and WebSocket communication
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from collections import deque
import threading

# ==========================================
# Data Models
# ==========================================

class LogEvent(BaseModel):
    block_id: str
    timestamp: datetime
    events: Dict[str, int]
    latency: float
    sequence: List[str]
    
class AnomalyDetection(BaseModel):
    block_id: str
    timestamp: datetime
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    event_pattern: str
    
class SystemMetrics(BaseModel):
    total_blocks: int
    anomalies_detected: int
    anomaly_rate: float
    avg_latency: float
    blocks_per_minute: float
    model_accuracy: float

# ==========================================
# ML Model Manager
# ==========================================

class AnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = [f'E{i}' for i in range(1, 30)]
        self.event_templates = self._load_event_templates()
        self.threshold = 0.5
        
        # Performance metrics
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
        
    def _load_event_templates(self):
        """Load event templates for interpretation"""
        templates = {
            'E1': 'Adding an already existing block',
            'E2': 'Verification succeeded',
            'E3': 'Served block',
            'E4': 'Got exception while serving',
            'E5': 'Receiving block',
            'E6': 'Received block',
            'E7': 'writeBlock received exception',
            'E8': 'PacketResponder Interrupted',
            'E9': 'Received block from',
            'E10': 'PacketResponder Exception',
            'E11': 'PacketResponder terminating',
            'E12': 'Exception writing block',
            'E13': 'Receiving empty packet',
            'E14': 'Exception in receiveBlock',
            'E15': 'Changing block file offset',
            'E16': 'Transmitted block',
            'E17': 'Failed to transfer',
            'E18': 'Starting thread to transfer',
            'E19': 'Reopen Block',
            'E20': 'Unexpected error deleting block',
            'E21': 'Deleting block',
            'E22': 'NameSystem allocateBlock',
            'E23': 'NameSystem delete',
            'E24': 'Removing block from neededReplications',
            'E25': 'ask to replicate',
            'E26': 'addStoredBlock: blockMap updated',
            'E27': 'Redundant addStoredBlock',
            'E28': 'addStoredBlock: does not belong',
            'E29': 'PendingReplicationMonitor timeout'
        }
        return templates
    
    def train_model(self, X_train, y_train):
        """Train the anomaly detection model"""
        print("Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        self.model.fit(X_scaled, y_train)
        
        print("Model trained successfully!")
        return self
    
    def predict(self, features: Dict[str, int]) -> tuple:
        """Predict if a log block is anomalous"""
        if self.model is None:
            # Return random prediction if model not trained
            is_anomaly = random.random() > 0.97  # ~3% anomaly rate
            confidence = random.uniform(0.6, 0.95) if is_anomaly else random.uniform(0.05, 0.4)
            return is_anomaly, confidence
        
        # Prepare features
        X = pd.DataFrame([features])[self.feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        is_anomaly = bool(prediction == 1)
        confidence = float(probability[1] if is_anomaly else probability[0])
        
        return is_anomaly, confidence
    
    def get_anomaly_pattern(self, features: Dict[str, int]) -> str:
        """Identify the pattern of anomaly"""
        top_events = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if not top_events:
            return "No significant events"
        
        patterns = []
        for event, count in top_events:
            if event in self.event_templates and count > 0:
                patterns.append(f"{event}({count})")
        
        return " → ".join(patterns) if patterns else "Unknown pattern"

# ==========================================
# Log Stream Simulator
# ==========================================

class HDFSLogSimulator:
    def __init__(self, dataset_path: Optional[str] = None):
        self.current_index = 0
        self.streaming = False
        self.speed_multiplier = 1.0
        
        # Load real HDFS data if available
        self.real_data = None
        self.load_dataset(dataset_path)
        
        # Event generation probabilities
        self.normal_event_probs = self._get_normal_probabilities()
        self.anomaly_event_probs = self._get_anomaly_probabilities()
        
    def load_dataset(self, path: Optional[str]):
        """Load real HDFS dataset if available"""
        try:
            if path and os.path.exists(path):
                self.real_data = pd.read_csv(path)
                print(f"Loaded {len(self.real_data)} blocks from dataset")
        except Exception as e:
            print(f"Could not load dataset: {e}")
            self.real_data = None
    
    def _get_normal_probabilities(self):
        """Event probabilities for normal blocks"""
        return {
            'E1': 0.01, 'E2': 0.15, 'E3': 0.60, 'E4': 0.02,
            'E5': 0.20, 'E6': 0.18, 'E7': 0.01, 'E8': 0.02,
            'E9': 0.15, 'E10': 0.01, 'E11': 0.08, 'E12': 0.01,
            'E13': 0.02, 'E14': 0.01, 'E15': 0.05, 'E16': 0.12,
            'E17': 0.01, 'E18': 0.10, 'E19': 0.03, 'E20': 0.01,
            'E21': 0.08, 'E22': 0.15, 'E23': 0.10, 'E24': 0.02,
            'E25': 0.08, 'E26': 0.15, 'E27': 0.02, 'E28': 0.01,
            'E29': 0.01
        }
    
    def _get_anomaly_probabilities(self):
        """Event probabilities for anomalous blocks"""
        return {
            'E1': 0.05, 'E2': 0.02, 'E3': 0.10, 'E4': 0.25,
            'E5': 0.08, 'E6': 0.05, 'E7': 0.20, 'E8': 0.15,
            'E9': 0.05, 'E10': 0.18, 'E11': 0.03, 'E12': 0.22,
            'E13': 0.10, 'E14': 0.15, 'E15': 0.02, 'E16': 0.03,
            'E17': 0.20, 'E18': 0.05, 'E19': 0.08, 'E20': 0.12,
            'E21': 0.02, 'E22': 0.03, 'E23': 0.02, 'E24': 0.10,
            'E25': 0.03, 'E26': 0.02, 'E27': 0.08, 'E28': 0.10,
            'E29': 0.15
        }
    
    def generate_log_event(self, force_anomaly: bool = False) -> LogEvent:
        """Generate a synthetic log event"""
        # Decide if this should be an anomaly
        is_anomaly = force_anomaly or (random.random() < 0.03)  # 3% anomaly rate
        
        # Select appropriate probabilities
        probs = self.anomaly_event_probs if is_anomaly else self.normal_event_probs
        
        # Generate events
        events = {}
        sequence = []
        
        # Number of events in this block
        num_events = random.randint(5, 50) if not is_anomaly else random.randint(10, 80)
        
        for _ in range(num_events):
            # Select event based on probabilities
            event = random.choices(
                list(probs.keys()),
                weights=list(probs.values()),
                k=1
            )[0]
            
            events[event] = events.get(event, 0) + 1
            sequence.append(event)
        
        # Generate latency (anomalies tend to have higher latency)
        if is_anomaly:
            latency = random.uniform(100, 5000)
        else:
            latency = random.uniform(10, 500)
        
        # Create log event
        return LogEvent(
            block_id=f"blk_{random.randint(1000000000, 9999999999)}",
            timestamp=datetime.now(),
            events=events,
            latency=latency,
            sequence=sequence[:20]  # Keep first 20 events
        )
    
    async def stream_logs(self, callback):
        """Stream log events continuously"""
        self.streaming = True
        
        while self.streaming:
            # Generate log event
            log_event = self.generate_log_event()
            
            # Call callback with event
            await callback(log_event)
            
            # Wait before next event (adjust speed)
            await asyncio.sleep(1.0 / self.speed_multiplier)
    
    def stop_streaming(self):
        """Stop the log stream"""
        self.streaming = False

# ==========================================
# WebSocket Connection Manager
# ==========================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

# ==========================================
# Main Application
# ==========================================

app = FastAPI(title="HDFS Anomaly Detection Live Demo")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
manager = ConnectionManager()
detector = AnomalyDetector()
simulator = HDFSLogSimulator()

# Metrics storage
metrics_store = {
    "total_blocks": 0,
    "anomalies_detected": 0,
    "recent_latencies": deque(maxlen=100),
    "blocks_per_minute": deque(maxlen=60),
    "anomaly_history": deque(maxlen=1000)
}

# ==========================================
# API Endpoints
# ==========================================

@app.get("/")
async def root():
    return {"message": "HDFS Anomaly Detection API", "status": "running"}

@app.get("/api/metrics")
async def get_metrics():
    """Get current system metrics"""
    anomaly_rate = (metrics_store["anomalies_detected"] / max(metrics_store["total_blocks"], 1)) * 100
    avg_latency = np.mean(metrics_store["recent_latencies"]) if metrics_store["recent_latencies"] else 0
    
    return SystemMetrics(
        total_blocks=metrics_store["total_blocks"],
        anomalies_detected=metrics_store["anomalies_detected"],
        anomaly_rate=anomaly_rate,
        avg_latency=avg_latency,
        blocks_per_minute=len(metrics_store["blocks_per_minute"]),
        model_accuracy=0.95  # Placeholder
    )

@app.get("/api/anomaly-history")
async def get_anomaly_history(limit: int = 100):
    """Get recent anomaly detections"""
    return list(metrics_store["anomaly_history"])[:limit]

@app.post("/api/control/start")
async def start_streaming():
    """Start log streaming"""
    if not simulator.streaming:
        asyncio.create_task(process_log_stream())
        return {"status": "started"}
    return {"status": "already running"}

@app.post("/api/control/stop")
async def stop_streaming():
    """Stop log streaming"""
    simulator.stop_streaming()
    return {"status": "stopped"}

@app.post("/api/control/speed/{multiplier}")
async def set_speed(multiplier: float):
    """Set streaming speed multiplier"""
    simulator.speed_multiplier = max(0.1, min(10.0, multiplier))
    return {"speed_multiplier": simulator.speed_multiplier}

@app.post("/api/train-model")
async def train_model():
    """Train the ML model with sample data"""
    # Generate training data
    X_train = []
    y_train = []
    
    for _ in range(1000):
        is_anomaly = random.random() < 0.1
        log_event = simulator.generate_log_event(force_anomaly=is_anomaly)
        
        features = [log_event.events.get(f'E{i}', 0) for i in range(1, 30)]
        X_train.append(features)
        y_train.append(1 if is_anomaly else 0)
    
    X_train = pd.DataFrame(X_train, columns=detector.feature_cols)
    detector.train_model(X_train, y_train)
    
    return {"status": "model trained", "samples": len(X_train)}

# ==========================================
# WebSocket Endpoint
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==========================================
# Processing Pipeline
# ==========================================

async def process_log_stream():
    """Main processing pipeline"""
    async def process_log(log_event: LogEvent):
        # Update metrics
        metrics_store["total_blocks"] += 1
        metrics_store["recent_latencies"].append(log_event.latency)
        metrics_store["blocks_per_minute"].append(datetime.now())
        
        # Prepare features for ML model
        features = {f'E{i}': 0 for i in range(1, 30)}
        features.update(log_event.events)
        
        # Run anomaly detection
        is_anomaly, confidence = detector.predict(features)
        
        # Create anomaly detection result
        detection = AnomalyDetection(
            block_id=log_event.block_id,
            timestamp=log_event.timestamp,
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_score=confidence if is_anomaly else 1 - confidence,
            event_pattern=detector.get_anomaly_pattern(log_event.events)
        )
        
        # Update anomaly metrics
        if is_anomaly:
            metrics_store["anomalies_detected"] += 1
            metrics_store["anomaly_history"].append(detection.dict())
        
        # Prepare WebSocket message
        message = {
            "type": "log_event",
            "data": {
                "log": log_event.dict(),
                "detection": detection.dict(),
                "metrics": {
                    "total_blocks": metrics_store["total_blocks"],
                    "anomalies_detected": metrics_store["anomalies_detected"],
                    "current_rate": len([t for t in metrics_store["blocks_per_minute"] 
                                        if t > datetime.now() - timedelta(minutes=1)])
                }
            }
        }
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps(message, default=str))
    
    # Start streaming
    await simulator.stream_logs(process_log)

# ==========================================
# Startup Events
# ==========================================

@app.on_event("startup")
async def startup_event():
    print("🚀 HDFS Anomaly Detection Server Starting...")
    print("📊 Initializing ML model...")
    
    # Try to load real dataset
    simulator.load_dataset("Event_occurrence_matrix.csv")
    
    # Train initial model with synthetic data
    await train_model()
    
    print("✅ Server ready!")
    print("🌐 Dashboard: http://localhost:8000/dashboard")
    print("📡 WebSocket: ws://localhost:8000/ws")

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML"""
    if os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    else:
        return HTMLResponse("<h1>Dashboard file not found. Please save dashboard.html</h1>")
