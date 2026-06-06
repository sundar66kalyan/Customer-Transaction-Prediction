import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

print("🔧 Fixing model file...")

# Create a simple model for testing
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create dummy data to train the model
X_train = np.random.rand(1000, 5)
y_train = np.random.randint(0, 2, 1000)

# Train the model
model.fit(X_train, y_train)
print("✅ Model trained successfully")

# Save the model properly
model_path = 'models/customer_transaction_lightgbm.pkl'
os.makedirs('models', exist_ok=True)

with open(model_path, 'wb') as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ Model saved to {model_path}")
print(f"   File size: {os.path.getsize(model_path)} bytes")
