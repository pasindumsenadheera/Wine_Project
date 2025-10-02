# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Create sample data for training
np.random.seed(42)
n_samples = 1000

data = {
    'fixed acidity': np.random.uniform(4.0, 16.0, n_samples),
    'volatile acidity': np.random.uniform(0.1, 1.6, n_samples),
    'citric acid': np.random.uniform(0.0, 1.0, n_samples),
    'residual sugar': np.random.uniform(0.9, 16.0, n_samples),
    'chlorides': np.random.uniform(0.01, 0.6, n_samples),
    'free sulfur dioxide': np.random.uniform(1.0, 70.0, n_samples),
    'total sulfur dioxide': np.random.uniform(6.0, 290.0, n_samples),
    'density': np.random.uniform(0.99, 1.004, n_samples),
    'pH': np.random.uniform(2.7, 4.0, n_samples),
    'sulphates': np.random.uniform(0.3, 2.0, n_samples),
    'alcohol': np.random.uniform(8.0, 15.0, n_samples),
}

df = pd.DataFrame(data)

# Create target variable (quality) based on features
df['quality'] = (
    (df['alcohol'] > 11).astype(int) +
    (df['volatile acidity'] < 0.5).astype(int) +
    (df['sulphates'] > 0.6).astype(int) +
    (df['citric acid'] > 0.3).astype(int) +
    3  # base quality
)

# Ensure quality is between 3 and 8
df['quality'] = np.clip(df['quality'], 3, 8)

# Prepare features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl")
print(f"Model accuracy on training data: {model.score(X, y):.2f}")