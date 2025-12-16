# Cloud-Reliability-Analysis-With-Log-Based-Anomaly-Detection-

 
This project focus on Cloud Reliability Analysis with Log-Based Anomaly Detection, to improve cloud system reliability by automatically detecting abnormal patterns in system logs using AI and machine learning techniques. The work builds upon the use of large-scale distributed system logs, specifically the HDFS dataset from LogHub, to identify anomalies that may indicate system faults or performance degradation. The project combines data preprocessing, feature engineering, and both supervised and unsupervised modeling approaches to detect unusual behaviors within system components. It also includes a prototype monitoring dashboard built  to visualize the analysis and prediction results. The end goal is to demonstrate how AI-based anomaly detection can enhance reliability monitoring and operational visibility in modern cloud environments.
<img width="1023" height="229" alt="image" src="https://github.com/user-attachments/assets/68a3110f-d77a-473d-be6c-391d2a0d58fc" />

## Live Dashboard

**Try the live dashboard:** [Log Anomaly Detection Dashboard](https://cloud-computing-dashboard-test.streamlit.app/)
<img width="1637" height="699" alt="image" src="https://github.com/user-attachments/assets/3db519de-5058-4bd0-a195-68f768c46e19" />


## Repository Structure

```
├── app1/                       # Published Streamlit dashboard (app1)
├── app2/                       # Alternative dashboard implementation (app2)
├── datasets/                   # HDFS datasets (not included - large files)
├── eda_notebook/              # Exploratory data analysis notebooks
├── unsupervised_modeling/     # Unsupervised models (Isolation Forest & Autoencoder)
├── modeling/                  # Supervised ML models (Random Forest, XGBoost, etc.)
├── deeplog_model/             # Enhanced DeepLog LSTM model
├── gcp_dashboard/             # React based dashboard attempts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

**Note**: Dataset files are not uploaded due to large file size. Download HDFS_v1 dataset from [LogHub](https://github.com/logpai/loghub) and place in `datasets/` folder.



## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Cloud-Reliability-Analysis-With-Log-Based-Anomaly-Detection.git
cd Cloud-Reliability-Analysis-With-Log-Based-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt
```


## Usage

### 1. Exploratory Data Analysis

```bash
cd eda_notebook
# Open and run Jupyter notebooks for data exploration
jupyter notebook
```

### 2. Unsupervised Models

Train baseline unsupervised models for anomaly detection:

```bash
cd unsupervised_modeling

# Isolation Forest
python isolation_forest_final.py

# Autoencoder
python autoencoder_final.py

```

### 3. Supervised Models

Train traditional supervised ML models:

```bash
cd modeling
# Run modeling notebooks or scripts
# Examples:  Gradient Boosting,Random Forest, XGBoost,
jupyter notebook
```

### 4. Enhanced DeepLog LSTM Model

The enhanced DeepLog LSTM model:

```bash
cd deeplog_model

# Train the model
python deeplog_model.py

# This will generate:
# - deeplog_best_model.h5 (trained model)
# - deeplog_final_model.h5 (final checkpoint)
# - deeplog_evaluation.png (performance visualization)

# Create deployment files for the dashboard
python save_deployment.py

# This creates:
# - event_to_id.pkl (event encoding mapping)
# - count_scaler.pkl (feature scaler)
# - feature_names.pkl (feature reference)
```

**Model Architecture:**
- Dual-input LSTM architecture (sequences + count features)
- Bidirectional LSTM layers for temporal pattern learning

### 5. Dashboard

#### Option A: Published Streamlit App (Recommended)

```bash
cd app1
streamlit run streamlit_anomaly_dashboard.py
```
Visit the live dashboard: **[Log Anomaly Detection Dashboard](https://cloud-computing-dashboard-test.streamlit.app/)**

#### Option B: Alternative Dashboard

```bash
cd app2
streamlit run dashboard.py
```
<img width="1637" height="883" alt="Screenshot 2025-12-16 at 3 45 26 AM" src="https://github.com/user-attachments/assets/5006e79b-3fa9-4d37-aac4-dfa5fb86bed2" />

#### Option C: React Based Dashboard (Experimental)

```bash
cd gcp_dashboard
# Open HTML files in browser for static visualization
```


## Requirements

See `requirements.txt` for complete dependencies. Key libraries include:

```
tensorflow>=2.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
streamlit>=1.20.0
plotly>=5.10.0
joblib>=1.1.0
```


### References

1. Du, M., Li, F., Zheng, G., & Srikumar, V. (2017). *DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning*. ACM SIGSAC Conference on Computer and Communications Security (CCS).

2. Zhu, J., He, S., Liu, J., He, P., Xie, Q., Zheng, Z., & Lyu, M. R. (2023). *A Large Collection of System Log Datasets for AI-driven Log Analytics*. IEEE International Symposium on Software Reliability Engineering (ISSRE).

3. Meng, W., et al. (2019). *LogAnomaly: Unsupervised Detection of Sequential and Quantitative Anomalies in Unstructured Logs*. IJCAI.
