import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Remove duplicates
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

# Convert categorical target variable
train_df['bc_price_evo'] = train_df['bc_price_evo'].map({'UP': 1, 'DOWN': 0})

# Save cleaned files
train_df.to_csv("train_cleaned.csv", index=False)
test_df.to_csv("test_cleaned.csv", index=False)

print("Data cleaning complete!")

# Select features and target
features = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
X = train_df[features]
y = train_df['bc_price_evo']

# Split into train & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=500, 
    criterion='entropy', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    max_features=None, 
    random_state=0
)
rf_model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = rf_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")

# Generate predictions for test set
test_preds = rf_model.predict(test_df[features])

# Save submission file
submission = pd.DataFrame({'id': test_df['id'], 'bc_price_evo': ['UP' if p == 1 else 'DOWN' for p in test_preds]})
submission.to_csv("submission.csv", index=False)

print("Submission file created!")
