import pandas as pd
import numpy as np

print("Generating large synthetic financial dataset...")
np.random.seed(42)

# 1. Generate 1000 Normal Transactions
dates = pd.date_range(start="2023-01-01", periods=1000, freq='h') # lowercase 'h' for hourly
students = [f"STD_{np.random.randint(1, 300):03d}" for _ in range(1000)]
amounts = np.random.normal(loc=1500, scale=150, size=1000)
methods = np.random.choice(['Bank Transfer', 'Credit Card', 'Cash'], 1000, p=[0.6, 0.35, 0.05])
types = np.random.choice(['Tuition Fee', 'Bus Fee', 'Library Fine'], 1000, p=[0.7, 0.25, 0.05])

df_normal = pd.DataFrame({
    'Date': dates, 
    'Transaction_ID': [f"TXN_{i:05d}" for i in range(1, 1001)], 
    'Student_ID': students, 
    'Amount': amounts, 
    'Method': methods, 
    'Type': types
})

# 2. Inject 15 Hard Anomalies (Massive payments, weird refunds, etc.)
df_anomalies = pd.DataFrame({
    'Date': pd.date_range(start="2023-02-01", periods=15, freq='10D'),
    'Transaction_ID': [f"TXN_99{i:03d}" for i in range(15)],
    'Student_ID': [f"STD_{np.random.randint(900, 999)}" for _ in range(15)],
    'Amount': [25000, -8000, 5, 45000, -2500, 30000, -10000, 2, 50000, -50, 18000, -7000, 8, 60000, -3000],
    'Method': ['Cash', 'Bank Transfer', 'Credit Card', 'Cash', 'Crypto', 'Cash', 'Bank Transfer', 'Credit Card', 'Cash', 'Bank Transfer', 'Crypto', 'Credit Card', 'Cash', 'Crypto', 'Bank Transfer'],
    'Type': ['Tuition Fee', 'Refund', 'Bus Fee', 'Donation', 'Tuition Fee', 'Tuition Fee', 'Refund', 'Library Fine', 'Donation', 'Refund', 'Tuition Fee', 'Refund', 'Bus Fee', 'Donation', 'Tuition Fee']
})

df = pd.concat([df_normal, df_anomalies], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle the data
df.to_csv("large_school_finances.csv", index=False)

print("✅ Successfully created 'large_school_finances.csv' with 1,015 records!")