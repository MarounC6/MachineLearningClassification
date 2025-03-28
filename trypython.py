# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        



# Load dataset
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

### 1. Check for missing values ###
#print("Missing values in train dataset:\n", train_df.isnull().sum())
#print("Missing values in test dataset:\n", test_df.isnull().sum())

# If missing values exist, we can fill them with the median
#train_df.fillna(train_df.median(), inplace=True)
#test_df.fillna(test_df.median(), inplace=True)

### 2. Check for duplicates ###
#print("Duplicates in train:", train_df.duplicated().sum())
#print("Duplicates in test:", test_df.duplicated().sum())

# Remove duplicates if necessary
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)

### 3. Validate Data Ranges ###
def check_ranges(df, cols, min_val=0, max_val=1):
    for col in cols:
        if not df[col].between(min_val, max_val).all():
            print(f"Warning: {col} has values out of range!")

numeric_columns = ['date', 'hour', 'bc_price', 'bc_demand', 'ab_price', 'ab_demand', 'transfer']
#check_ranges(train_df, numeric_columns)
#check_ranges(test_df, numeric_columns)


### 4. Detect Outliers (Using Z-Score) ###
from scipy.stats import zscore

z_scores = np.abs(zscore(train_df[numeric_columns]))
outliers = (z_scores > 3).sum(axis=0)
#print("Potential outliers detected per column:\n", outliers)

# Optionally, remove extreme outliers (beyond 3 standard deviations)
#train_df = train_df[(z_scores < 3).all(axis=1)]

### 5. Ensure Data Types are Correct ###
#print(train_df.dtypes)

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

# Define different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, 
        min_samples_split=5, 
        min_samples_leaf=1, 
        max_features=None, 
        max_depth=30, 
        bootstrap=True, 
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        subsample=1.0, 
        n_estimators=200, 
        min_child_weight=1, 
        max_depth=5, 
        learning_rate=0.2, 
        gamma=0.2, 
        colsample_bytree=1.0
    ),
    "LightGBM": LGBMClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
}

# Train and evaluate each model
accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")

# Select the best model
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]

# Generate predictions for test set using best model
test_preds = best_model.predict(test_df[features])

# Save submission file
submission = pd.DataFrame({'id': test_df['id'], 'bc_price_evo': ['UP' if p == 1 else 'DOWN' for p in test_preds]})
submission.to_csv("submission.csv", index=False)

# Print final summary of accuracies
print("\n### Model Performance Summary ###")
for name, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc:.4f}")

print(f"\nBest Model: {best_model_name} with Accuracy: {accuracies[best_model_name]:.4f}")
'''
# Define parameter grid
param_dist = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=50, cv=5, 
    scoring='accuracy', n_jobs=-1, verbose=2, random_state=42
)

random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_rf.predict(X_val)
print("Optimized Random Forest Accuracy:", accuracy_score(y_val, y_pred))
print("Best Parameters:", random_search.best_params_)'
'''

xg_model = xgb.XGBClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Perform RandomizedSearchCV
random_search = RandomizedSearchCV(xg_model, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=3, random_state=42)
random_search.fit(X_train, y_train)

# Get the best parameters and model
print("Best parameters found:", random_search.best_params_)
