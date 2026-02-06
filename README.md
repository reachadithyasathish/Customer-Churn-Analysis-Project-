# Customer Churn Prediction System

An end-to-end machine learning application that predicts customer churn using a Random Forest classifier, complete with an interactive Streamlit dashboard.

## 📁 Project Structure

```
churn_prediction/
├── generate_data.py    # Synthetic data generator
├── train_model.py      # Model training pipeline
├── app.py              # Streamlit dashboard
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── data/               # Generated dataset (created after running generate_data.py)
│   └── customer_churn.csv
└── models/             # Trained model artifacts (created after running train_model.py)
    └── churn_model.pkl
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python generate_data.py
```

This creates a realistic Telco customer dataset with ~1000 records.

### 3. Train the Model

```bash
python train_model.py
```

This trains a Random Forest classifier and saves the model artifacts.

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## 📊 Features

- **Synthetic Data Generation**: Creates realistic customer churn data with correlated features
- **ML Pipeline**: Preprocessing, training, and evaluation with scikit-learn
- **Interactive Dashboard**: Real-time churn predictions with visual risk indicators
- **Feature Importance**: Understand which factors drive customer churn

## 🛠️ Tech Stack

- **Python 3.8+**
- **pandas** - Data manipulation
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning
- **Streamlit** - Web dashboard
- **joblib** - Model serialization

## 📈 Model Performance

The Random Forest model typically achieves:
- **Accuracy**: ~80-85%
- **Key Features**: Contract type, Tenure, and Payment Method are usually the strongest predictors

## 📝 License

MIT License - Feel free to use this for your portfolio!
