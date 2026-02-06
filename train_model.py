"""
Customer Churn Model Training Pipeline
Trains a Random Forest classifier and saves model artifacts.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_and_preprocess_data(filepath: str):
    """Load data and apply preprocessing transformations."""
    print("Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df.drop(columns=["CustomerID", "Churn"])
    y = df["Churn"]
    
    # Identify column types
    categorical_cols = ["Contract", "PaymentMethod"]
    numerical_cols = ["Tenure", "MonthlyCharges", "TotalCharges"]
    
    # Initialize encoders and scaler
    label_encoders = {}
    scaler = StandardScaler()
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    label_encoders["Churn"] = target_encoder
    
    # Scale numerical features
    print("Scaling numerical features...")
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y_encoded, label_encoders, scaler, numerical_cols, categorical_cols


def train_model(X, y):
    """Train Random Forest classifier."""
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives:  {cm[0][0]}")
    print(f"  False Positives: {cm[0][1]}")
    print(f"  False Negatives: {cm[1][0]}")
    print(f"  True Positives:  {cm[1][1]}")
    
    return model, X.columns.tolist()


def save_artifacts(model, label_encoders, scaler, feature_names):
    """Save model and preprocessing artifacts."""
    os.makedirs("models", exist_ok=True)
    
    artifacts = {
        "model": model,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "feature_names": feature_names
    }
    
    output_path = "models/churn_model.pkl"
    joblib.dump(artifacts, output_path)
    print(f"\n✓ Model artifacts saved to: {output_path}")
    
    # Print feature importances
    print("\nFeature Importances:")
    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    for _, row in importances.iterrows():
        bar = "█" * int(row["Importance"] * 40)
        print(f"  {row['Feature']:15} {bar} {row['Importance']:.4f}")
    
    return output_path


def main():
    """Main training pipeline."""
    data_path = "data/customer_churn.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run 'python generate_data.py' first.")
        return
    
    # Load and preprocess
    X, y, label_encoders, scaler, num_cols, cat_cols = load_and_preprocess_data(data_path)
    
    # Train model
    model, feature_names = train_model(X, y)
    
    # Save artifacts
    save_artifacts(model, label_encoders, scaler, feature_names)
    
    print("\n" + "=" * 50)
    print("Training complete! Run 'streamlit run app.py' to launch the dashboard.")
    print("=" * 50)


if __name__ == "__main__":
    main()
