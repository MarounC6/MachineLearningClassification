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
    #"Random Forest": RandomForestClassifier(n_estimators=460, min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=24, bootstrap=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=500,  criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=0 ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, 
        eval_metric='logloss', 
        subsample=0.95, 
        n_estimators=1500, 
        min_child_weight=1, 
        max_depth=15, 
        learning_rate=0.1, 
        gamma=0.1, 
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




# Define parameter grid
param_dist = {
    'n_estimators': [400, 500, 600],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', None],
    'random_state': [0, 42],
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    rf, param_distributions=param_dist, n_iter=20, cv=2, 
    scoring='accuracy', n_jobs=-1, verbose=2, random_state=42
)

random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_rf.predict(X_val)
print("Optimized Random Forest Accuracy:", accuracy_score(y_val, y_pred))
print("Best Parameters:", random_search.best_params_)



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
'
'''


'''
import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Define XGBoost parameter search space
xgb_params = {
    "n_estimators": [300, 350, 400, 450],
    "learning_rate": [0.1, 0.15, 0.2, 0.25],
    "max_depth": [6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 4],
    "subsample": [0.8, 0.85, 0.9],
    "colsample_bytree": [0.9, 1.0],
    "gamma": [0.3, 0.35, 0.4, 0.45],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2],
}

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)

# Perform RandomizedSearchCV
xgb_search = RandomizedSearchCV(
    xgb_model, param_distributions=xgb_params, n_iter=100, scoring="accuracy",
    cv=4, random_state=42, verbose=2, n_jobs=-1
)

# Start timer
start_time = time.time()

# Fit the model to find the best parameters
xgb_search.fit(X_train, y_train)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# Best model
best_xgb = xgb_search.best_estimator_
best_xgb_params = xgb_search.best_params_

# Evaluate on validation set
y_pred_xgb = best_xgb.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)

# Print results
print("\n🎯 Final Best Parameters Found:")
print(best_xgb_params)
print(f"\n🚀 Optimized XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"\n⏳ Time Taken: {elapsed_time:.2f} seconds")

'''





















'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

# Define parameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'max_depth': [10, 20, 30, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Initialize RandomizedSearchCV
random_search_rf = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=rf_params,
    n_iter=50,  # Number of parameter settings sampled
    cv=3,  # 5-fold cross-validation
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# Fit RandomizedSearchCV
random_search_rf.fit(X_train, y_train)

# Get the best model and parameters
best_rf_model = random_search_rf.best_estimator_
best_rf_params = random_search_rf.best_params_
best_rf_accuracy = random_search_rf.best_score_

# Print the best parameters and accuracy
print("\n🎯 Best Parameters Found for Random Forest:")
print(best_rf_params)
print(f"\n🚀 Random Forest Cross-Validation Accuracy: {best_rf_accuracy:.4f}")

# Evaluate the best model on the validation set
y_pred_rf = best_rf_model.predict(X_val)
val_rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"\n✅ Random Forest Validation Accuracy: {val_rf_accuracy:.4f}")

# Generate predictions for the test set
test_preds_rf = best_rf_model.predict(test_df[features])

# Save submission file
submission_rf = pd.DataFrame({'id': test_df['id'], 'bc_price_evo': ['UP' if p == 1 else 'DOWN' for p in test_preds_rf]})
submission_rf.to_csv("submission_rf.csv", index=False)


'''
'''
import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Define XGBoost parameter search space
xgb_params = {
    "n_estimators": [300, 350, 400, 450],
    "learning_rate": [0.1, 0.15, 0.2, 0.25],
    "max_depth": [6, 7, 8, 9],
    "min_child_weight": [1, 2, 3, 4],
    "subsample": [0.8, 0.85, 0.9],
    "colsample_bytree": [0.9, 1.0],
    "gamma": [0.3, 0.35, 0.4, 0.45],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2],
}

# Initialize XGBoost model
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)

# Perform RandomizedSearchCV
xgb_search = RandomizedSearchCV(
    xgb_model, param_distributions=xgb_params, n_iter=100, scoring="accuracy",
    cv=2, random_state=42, verbose=2, n_jobs=-1
)

# Start timer
start_time = time.time()

# Fit the model to find the best parameters with early stopping
for train_idx, val_idx in StratifiedKFold(n_splits=2, shuffle=True, random_state=42).split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
    xgb_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        early_stopping_rounds=10,
        verbose=False
    )
xgb_search.fit(X_train, y_train)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time

# Best model
best_xgb = xgb_search.best_estimator_
best_xgb_params = xgb_search.best_params_

# Evaluate on validation set
y_pred_xgb = best_xgb.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)

# Print results
print("\n🎯 Final Best Parameters Found:")
print(best_xgb_params)
print(f"\n🚀 Optimized XGBoost Accuracy: {xgb_accuracy:.4f}")
print(f"\n⏳ Time Taken: {elapsed_time:.2f} seconds")
'''