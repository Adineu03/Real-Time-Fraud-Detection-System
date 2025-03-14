
# Real-Time Fraud Detection System 🔍

A sophisticated fraud detection dashboard built with Streamlit that leverages multiple machine learning models to detect fraudulent transactions in real-time.

[link-to-app-created-in-hackathon](https://aditya-maib-rdmu-fraud-insight-dashboard.replit.app/)

## 🌟 Features

- **Real-Time Monitoring**: Live transaction monitoring and instant fraud detection
- **Multi-Model Approach**: Combines multiple scoring models:
  - Bayesian Inference
  - Maximum Likelihood Estimation (MLE)
  - Multi-Armed Bandit Algorithm
  - Fuzzy Logic Scoring

- **Interactive Dashboard**:
  - Risk score distribution visualization
  - Transaction pattern analysis
  - ROC curve performance metrics
  - Detailed transaction inspection

- **Advanced Analytics**:
  - Model performance comparisons
  - Risk factor breakdown
  - Trend analysis
  - Feature importance visualization

## 🛠️ Technical Architecture

### Core Components

1. **Data Processing** (`utils/data_processor.py`):
   - File format handling (CSV/Excel)
   - Data cleaning and preprocessing
   - Real-time data updates

2. **Risk Models** (`utils/risk_models.py`):
   - Bayesian scoring implementation
   - MLE-based risk assessment
   - Multi-Armed Bandit algorithm
   - Fuzzy logic scoring system

3. **Visualization Engine** (`utils/visualization.py`):
   - Interactive charts and graphs
   - Risk distribution plotting
   - Performance metric visualization

4. **Main Application** (`app.py`):
   - Streamlit dashboard interface
   - Real-time monitoring system
   - Transaction simulation capabilities

## 📊 Dashboard Features

1. **Main Dashboard**:
   - Transaction metrics overview
   - Risk score distribution charts
   - Real-time alerts
   - Performance indicators

2. **Transaction Analysis**:
   - Detailed transaction view
   - Risk factor breakdown
   - Contributing factors analysis
   - Recommended actions

3. **Model Performance**:
   - ROC curves
   - Model comparison metrics
   - Feature importance charts
   - Real-time performance tracking

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📝 Usage

1. **Upload Data**:
   - Use the sidebar to upload transaction data (CSV/Excel)
   - Supported columns: Transaction ID, Amount, Date/Time, Location, Merchant

2. **Monitor Transactions**:
   - View real-time risk scores
   - Check flagged transactions
   - Analyze risk patterns

3. **Analyze Results**:
   - Examine risk distributions
   - Review model performance
   - Investigate flagged transactions

## 🔧 System Requirements

- Python 3.7+
- Required libraries:
  - Streamlit
  - Pandas
  - NumPy
  - Plotly
  - SciPy
  - Scikit-fuzzy

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 👥 Authors

- Aditya Tripathi

## 🙏 Acknowledgments

- Special thanks to all contributors
- Built with Streamlit and Python
- Powered by advanced machine learning algorithms
