# Cloud-Reliability-Analysis-With-Log-Based-Anomaly-Detection-

 
This project focus on Cloud Reliability Analysis with Log-Based Anomaly Detection, to improve cloud system reliability by automatically detecting abnormal patterns in system logs using AI and machine learning techniques. The work builds upon the use of large-scale distributed system logs, specifically the HDFS dataset from LogHub, to identify anomalies that may indicate system faults or performance degradation. The project combines data preprocessing, feature engineering, and both supervised and unsupervised modeling approaches to detect unusual behaviors within system components. It also includes a prototype monitoring dashboard built on Google Cloud Platform (GCP) to visualize real-time anomalies and reliability metrics such as Mean Time to Detect (MTTD) and Mean Time to Resolve (MTTR). The end goal is to demonstrate how AI-based anomaly detection can enhance reliability monitoring and operational visibility in modern cloud environments.


# 
## Repository Structure

```
├── datasets/                    # HDFS datasets (not included - large files)
├── eda_notebook/               # Exploratory data analysis
├── modeling/                   # Supervised ML models
├── unsupervised_modeling/      # Unsupervised models (Isolation Forest & Autoencoder)
└── gcp_dashboard/              # Dashboard HTML prototype
```

**Note**: Dataset files are not uploaded due to large file size. Download HDFS_v1 dataset from [LogHub](https://github.com/logpai/loghub) and place in `datasets/` folder.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Unsupervised Models
```bash
cd unsupervised_modeling
python isolation_forest_final.py
python autoencoder_final.py
python vae_final.py
```

### Supervised Models
```bash
cd modeling
# Run modeling notebooks or scripts
```

### Dashboard (In Progress)
```bash
cd gcp_dashboard
# Open HTML files in browser
```
