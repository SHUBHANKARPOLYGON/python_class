import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic training data
np.random.seed(42)
data = {
    'risk_level': np.random.randint(1, 4, 1000),
    'profit_ratio': np.random.uniform(0.1, 1.5, 1000),
    'diversification': np.random.randint(1, 5, 1000),
    'approval': np.where((np.random.uniform(0, 1, 1000) > 0.3), 1, 0)
}
df = pd.DataFrame(data)

# Train model
X = df[['risk_level', 'profit_ratio', 'diversification']]
y = df['approval']
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'portfolio_model.joblib')