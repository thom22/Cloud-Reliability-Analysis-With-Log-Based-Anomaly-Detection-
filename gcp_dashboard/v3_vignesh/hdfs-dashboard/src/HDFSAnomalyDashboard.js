import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { AlertCircle, Activity, TrendingUp, Database, Zap, Shield, PlayCircle, PauseCircle, AlertTriangle, CheckCircle } from 'lucide-react';

const HDFSAnomalyDashboard = () => {
  // State management
  const [isStreaming, setIsStreaming] = useState(false);
  const [metrics, setMetrics] = useState({
    totalBlocks: 0,
    anomaliesDetected: 0,
    anomalyRate: 0,
    avgLatency: 0,
    blocksPerMin: 0,
    modelAccuracy: 95.2
  });
  
  const [logs, setLogs] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [timelineData, setTimelineData] = useState([]);
  const [eventDistribution, setEventDistribution] = useState([]);
  const [patternData, setPatternData] = useState([]);
  const [streamSpeed, setStreamSpeed] = useState(1);
  
  const wsRef = useRef(null);
  const logsRef = useRef([]);
  const anomaliesRef = useRef([]);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    wsRef.current = new WebSocket('ws://localhost:8000/ws');
    
    wsRef.current.onopen = () => {
      console.log('WebSocket connected');
      addNotification('Connected to server', 'success');
    };
    
    wsRef.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      processMessage(message);
    };
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      addNotification('Connection error', 'error');
    };
    
    wsRef.current.onclose = () => {
      console.log('WebSocket disconnected');
      setTimeout(connectWebSocket, 3000);
    };
  };

  const processMessage = (message) => {
    if (message.type === 'log_event') {
      const { log, detection, metrics: newMetrics } = message.data;
      
      // Update metrics
      setMetrics(prev => ({
        ...prev,
        totalBlocks: newMetrics.total_blocks,
        anomaliesDetected: newMetrics.anomalies_detected,
        anomalyRate: ((newMetrics.anomalies_detected / Math.max(newMetrics.total_blocks, 1)) * 100).toFixed(2),
        blocksPerMin: newMetrics.current_rate || 0
      }));
      
      // Add log entry
      const logEntry = {
        id: Date.now() + Math.random(),
        blockId: log.block_id,
        timestamp: new Date(log.timestamp).toLocaleTimeString(),
        eventCount: Object.values(log.events || {}).reduce((a, b) => a + b, 0),
        latency: log.latency,
        isAnomaly: detection.is_anomaly,
        confidence: detection.confidence,
        pattern: detection.event_pattern
      };
      
      logsRef.current = [logEntry, ...logsRef.current].slice(0, 50);
      setLogs([...logsRef.current]);
      
      // Add to anomalies if detected
      if (detection.is_anomaly) {
        const anomalyEntry = {
          ...logEntry,
          score: detection.anomaly_score
        };
        anomaliesRef.current = [anomalyEntry, ...anomaliesRef.current].slice(0, 10);
        setAnomalies([...anomaliesRef.current]);
      }
      
      // Update timeline
      setTimelineData(prev => {
        const newData = [...prev, {
          time: new Date(log.timestamp).toLocaleTimeString(),
          score: detection.anomaly_score,
          label: detection.is_anomaly ? 'Anomaly' : 'Normal'
        }].slice(-30);
        return newData;
      });
      
      // Update event distribution
      const events = log.events || {};
      setEventDistribution(prev => {
        const dist = { ...prev };
        Object.entries(events).forEach(([event, count]) => {
          dist[event] = (dist[event] || 0) + count;
        });
        return Object.entries(dist)
          .map(([name, value]) => ({ name, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 10);
      });
    }
  };

  const addNotification = (message, type) => {
    console.log(`[${type}] ${message}`);
  };

  const startStreaming = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/control/start', {
        method: 'POST'
      });
      if (response.ok) {
        setIsStreaming(true);
        addNotification('Streaming started', 'success');
      }
    } catch (error) {
      addNotification('Failed to start streaming', 'error');
    }
  };

  const stopStreaming = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/control/stop', {
        method: 'POST'
      });
      if (response.ok) {
        setIsStreaming(false);
        addNotification('Streaming stopped', 'info');
      }
    } catch (error) {
      addNotification('Failed to stop streaming', 'error');
    }
  };

  const changeSpeed = async (speed) => {
    try {
      const response = await fetch(`http://localhost:8000/api/control/speed/${speed}`, {
        method: 'POST'
      });
      if (response.ok) {
        setStreamSpeed(speed);
        addNotification(`Speed set to ${speed}x`, 'info');
      }
    } catch (error) {
      addNotification('Failed to change speed', 'error');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 mb-6 border border-white/20">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">HDFS Anomaly Detection</h1>
              <p className="text-gray-300">Real-time Log Analysis with ML-Powered Detection</p>
            </div>
            <div className="flex gap-4">
              <button
                onClick={isStreaming ? stopStreaming : startStreaming}
                className={`px-6 py-3 rounded-xl font-semibold flex items-center gap-2 transition-all transform hover:scale-105 ${
                  isStreaming 
                    ? 'bg-red-500 hover:bg-red-600 text-white' 
                    : 'bg-green-500 hover:bg-green-600 text-white'
                }`}
              >
                {isStreaming ? <PauseCircle size={20} /> : <PlayCircle size={20} />}
                {isStreaming ? 'Stop' : 'Start'} Streaming
              </button>
              <select
                value={streamSpeed}
                onChange={(e) => changeSpeed(e.target.value)}
                className="px-4 py-2 rounded-xl bg-white/20 text-white border border-white/30 backdrop-blur"
              >
                <option value="0.5">0.5x Speed</option>
                <option value="1">1x Speed</option>
                <option value="2">2x Speed</option>
                <option value="5">5x Speed</option>
              </select>
            </div>
          </div>
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Total Blocks"
            value={metrics.totalBlocks.toLocaleString()}
            icon={<Database className="text-blue-400" size={24} />}
            color="blue"
          />
          <MetricCard
            title="Anomalies"
            value={metrics.anomaliesDetected.toLocaleString()}
            icon={<AlertTriangle className="text-red-400" size={24} />}
            color="red"
          />
          <MetricCard
            title="Anomaly Rate"
            value={`${metrics.anomalyRate}%`}
            icon={<TrendingUp className="text-orange-400" size={24} />}
            color="orange"
          />
          <MetricCard
            title="Blocks/Min"
            value={metrics.blocksPerMin}
            icon={<Zap className="text-green-400" size={24} />}
            color="green"
          />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Live Log Stream */}
          <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <Activity size={20} />
              Live Log Stream
            </h2>
            <div className="h-[500px] overflow-y-auto space-y-2">
              {logs.map((log) => (
                <LogEntry key={log.id} log={log} />
              ))}
              {logs.length === 0 && (
                <div className="text-gray-400 text-center py-8">
                  No logs yet. Start streaming to see data.
                </div>
              )}
            </div>
          </div>

          {/* Charts */}
          <div className="space-y-6">
            {/* Anomaly Timeline */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4">Anomaly Timeline</h2>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="time" stroke="#888" fontSize={10} />
                  <YAxis stroke="#888" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: 'none' }}
                    labelStyle={{ color: '#888' }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="score" 
                    stroke="#ef4444" 
                    fill="url(#colorGradient)" 
                    strokeWidth={2}
                  />
                  <defs>
                    <linearGradient id="colorGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1}/>
                    </linearGradient>
                  </defs>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Event Distribution */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4">Event Distribution</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={eventDistribution}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="name" stroke="#888" fontSize={10} />
                  <YAxis stroke="#888" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: 'none' }}
                    labelStyle={{ color: '#888' }}
                  />
                  <Bar dataKey="value" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Anomalies & Model Info */}
          <div className="space-y-6">
            {/* Recent Anomalies */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <AlertCircle className="text-red-400" size={20} />
                Recent Anomalies
              </h2>
              <div className="space-y-2 max-h-[250px] overflow-y-auto">
                {anomalies.map((anomaly) => (
                  <AnomalyAlert key={anomaly.id} anomaly={anomaly} />
                ))}
                {anomalies.length === 0 && (
                  <div className="text-gray-400 text-center py-4">
                    No anomalies detected yet.
                  </div>
                )}
              </div>
            </div>

            {/* Model Performance */}
            <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
              <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <Shield className="text-green-400" size={20} />
                Model Performance
              </h2>
              <div className="space-y-4">
                <PerformanceBar label="Accuracy" value={95.2} color="green" />
                <PerformanceBar label="Precision" value={92.8} color="blue" />
                <PerformanceBar label="Recall" value={89.5} color="purple" />
                
                <div className="mt-4 p-3 bg-black/30 rounded-lg">
                  <p className="text-xs text-gray-400">Model: XGBoost</p>
                  <p className="text-xs text-gray-400">Features: 29 Event Types</p>
                  <p className="text-xs text-gray-400">Dataset: 575K HDFS Blocks</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Component for metric cards
const MetricCard = ({ title, value, icon, color }) => {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/20 border-blue-400/30',
    red: 'from-red-500/20 to-red-600/20 border-red-400/30',
    orange: 'from-orange-500/20 to-orange-600/20 border-orange-400/30',
    green: 'from-green-500/20 to-green-600/20 border-green-400/30',
  };

  return (
    <div className={`bg-gradient-to-br ${colorClasses[color]} backdrop-blur-lg rounded-xl p-4 border`}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-gray-300 text-sm">{title}</p>
          <p className="text-3xl font-bold text-white mt-1">{value}</p>
        </div>
        {icon}
      </div>
    </div>
  );
};

// Component for log entries
const LogEntry = ({ log }) => {
  return (
    <div className={`p-3 rounded-lg border ${
      log.isAnomaly 
        ? 'bg-red-500/20 border-red-400/40' 
        : 'bg-gray-500/10 border-gray-400/20'
    }`}>
      <div className="flex justify-between items-start">
        <div className="flex-1">
          <p className="text-xs text-gray-400">{log.timestamp}</p>
          <p className="text-xs font-mono text-gray-300 truncate">{log.blockId}</p>
          <p className="text-xs text-gray-400">
            Events: {log.eventCount} | Latency: {log.latency?.toFixed(0)}ms
          </p>
        </div>
        {log.isAnomaly ? (
          <span className="text-red-400 text-xs font-semibold flex items-center gap-1">
            <AlertTriangle size={12} />
            {(log.confidence * 100).toFixed(0)}%
          </span>
        ) : (
          <CheckCircle className="text-green-400" size={16} />
        )}
      </div>
    </div>
  );
};

// Component for anomaly alerts
const AnomalyAlert = ({ anomaly }) => {
  return (
    <div className="bg-red-500/20 border border-red-400/40 rounded-lg p-3">
      <div className="flex justify-between items-start">
        <div>
          <p className="text-xs font-semibold text-red-300">{anomaly.timestamp}</p>
          <p className="text-xs font-mono text-gray-300 truncate">
            {anomaly.blockId?.substring(0, 20)}...
          </p>
          <p className="text-xs text-gray-400 mt-1">Pattern: {anomaly.pattern}</p>
        </div>
        <span className="text-sm font-bold text-red-400">
          {(anomaly.confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
};

// Component for performance bars
const PerformanceBar = ({ label, value, color }) => {
  const colorClasses = {
    green: 'bg-green-500',
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="font-semibold text-white">{value}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div 
          className={`${colorClasses[color]} h-2 rounded-full transition-all duration-500`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
};

export default HDFSAnomalyDashboard;
