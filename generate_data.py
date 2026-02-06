"""
Synthetic Telco Customer Churn Data Generator
Generates ~1000 realistic customer records for churn prediction modeling.
"""

import pandas as pd
import numpy as np
import os

def generate_churn_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset."""
    np.random.seed(seed)
    
    # CustomerID
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_samples + 1)]
    
    # Contract types with realistic distribution
    contracts = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20]
    )
    
    # Payment methods
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.35, 0.20, 0.25, 0.20]
    )
    
    # Tenure (months) - newer customers more likely to churn
    tenure = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure = np.clip(tenure, 1, 72)
    
    # Monthly charges - varies by contract type
    monthly_charges = np.zeros(n_samples)
    for i, contract in enumerate(contracts):
        if contract == "Month-to-month":
            monthly_charges[i] = np.random.normal(70, 25)
        elif contract == "One year":
            monthly_charges[i] = np.random.normal(60, 20)
        else:
            monthly_charges[i] = np.random.normal(50, 15)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0).round(2)
    
    # Total charges based on tenure and monthly
    total_charges = (tenure * monthly_charges * np.random.uniform(0.9, 1.1, n_samples)).round(2)
    
    # Generate churn based on realistic factors
    churn_probability = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.15  # Base churn rate
        
        # Contract type influence
        if contracts[i] == "Month-to-month":
            prob += 0.25
        elif contracts[i] == "One year":
            prob += 0.05
        
        # Payment method influence
        if payment_methods[i] == "Electronic check":
            prob += 0.15
        
        # Tenure influence (newer customers churn more)
        if tenure[i] < 6:
            prob += 0.20
        elif tenure[i] < 12:
            prob += 0.10
        elif tenure[i] > 48:
            prob -= 0.15
        
        # High monthly charges increase churn
        if monthly_charges[i] > 80:
            prob += 0.10
        elif monthly_charges[i] < 40:
            prob -= 0.05
        
        # Add some noise
        prob += np.random.normal(0, 0.05)
        churn_probability[i] = np.clip(prob, 0.05, 0.95)
    
    # Generate churn labels
    churn = np.random.binomial(1, churn_probability).astype(str)
    churn = np.where(churn == "1", "Yes", "No")
    
    # Create DataFrame
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contracts,
        "PaymentMethod": payment_methods,
        "Churn": churn
    })
    
    return df


if __name__ == "__main__":
   
    
    # Generate and save dataset
    print("Generating synthetic customer churn data...")
    df = generate_churn_data(n_samples=1000)
    
    output_path = "customer_churn.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))

