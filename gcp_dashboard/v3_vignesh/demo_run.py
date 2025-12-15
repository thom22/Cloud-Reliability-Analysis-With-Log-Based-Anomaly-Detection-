#!/usr/bin/env python3
"""
HDFS Anomaly Detection Demo - Startup Script
This script sets up and runs the complete demo
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required = ['fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 'xgboost']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed!")
    else:
        print("✅ All requirements satisfied")

def check_data_files():
    """Check if HDFS data files are present"""
    data_files = [
        "anomaly_label.csv",
        "Event_occurrence_matrix.csv",
        "Event_traces.csv",
        "HDFS.log_templates.csv"
    ]
    
    missing_files = []
    for file in data_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"⚠️  Missing data files: {', '.join(missing_files)}")
        print("The demo will work with synthetic data, but real data provides better results.")
        print("Place the HDFS CSV files in the same directory as this script.")
    else:
        print("✅ All data files found")

def save_files():
    """Save the backend and frontend files"""
    # Check if files already exist
    if not Path("backend.py").exists():
        print("📝 Creating backend.py...")
        # Note: In production, you would copy the backend.py content here
        print("   Please save the backend.py file from the artifact above")
    
    if not Path("dashboard.html").exists():
        print("📝 Creating dashboard.html...")
        # Note: In production, you would copy the dashboard.html content here
        print("   Please save the dashboard.html file from the artifact above")

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting HDFS Anomaly Detection Server...")
    print("-" * 50)
    print("📡 API Server: http://localhost:8000")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📈 API Docs: http://localhost:8000/docs")
    print("-" * 50)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8000/dashboard")
    
    # Start browser in background
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    try:
        subprocess.run([sys.executable, "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down server...")
        print("Thank you for using HDFS Anomaly Detection Demo!")

def main():
    """Main entry point"""
    print("=" * 50)
    print("HDFS ANOMALY DETECTION - LIVE DEMO")
    print("=" * 50)
    print()
    
    # Check environment
    print("🔍 Checking environment...")
    check_requirements()
    check_data_files()
    save_files()
    
    # Provide instructions if files are missing
    if not Path("backend.py").exists() or not Path("dashboard.html").exists():
        print("\n⚠️  Required files missing!")
        print("Please ensure you have:")
        print("  1. backend.py - The FastAPI server")
        print("  2. dashboard.html - The web dashboard")
        print("\nSave these files from the artifacts above and run this script again.")
        return
    
    # Create a simple route to serve the dashboard
    if Path("backend.py").exists():
        # Add dashboard route to backend if not present
        backend_content = Path("backend.py").read_text()
        if '@app.get("/dashboard")' not in backend_content:
            dashboard_route = '''
@app.get("/dashboard")
async def dashboard():
    """Serve the dashboard HTML"""
    if os.path.exists("dashboard.html"):
        return FileResponse("dashboard.html")
    else:
        return HTMLResponse("<h1>Dashboard file not found. Please save dashboard.html</h1>")
'''
            # Append the route to backend.py
            with open("backend.py", "a") as f:
                f.write("\n" + dashboard_route)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()