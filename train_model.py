"""
Advanced Iris Classification with Multiple Models and Feature Engineering
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IRIS CLASSIFICATION - ADVANCED MODEL TRAINING")
print("="*70)

# 1. Load dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, header=None, names=columns)

print("\n📊 Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"\nSpecies distribution:\n{df['species'].value_counts()}")
print(f"\nBasic Statistics:\n{df.describe()}")

# 2. Feature Engineering - Create additional features
def engineer_features(df):
    """Create engineered features for better model performance"""
    df_eng = df.copy()
    
    # Ratio features
    df_eng['sepal_ratio'] = df_eng['sepal_length'] / df_eng['sepal_width']
    df_eng['petal_ratio'] = df_eng['petal_length'] / df_eng['petal_width']
    
    # Area features (approximate)
    df_eng['sepal_area'] = df_eng['sepal_length'] * df_eng['sepal_width']
    df_eng['petal_area'] = df_eng['petal_length'] * df_eng['petal_width']
    
    # Interaction features
    df_eng['sepal_petal_length_ratio'] = df_eng['sepal_length'] / (df_eng['petal_length'] + 1e-5)
    df_eng['sepal_petal_width_ratio'] = df_eng['sepal_width'] / (df_eng['petal_width'] + 1e-5)
    
    # Polynomial features
    df_eng['petal_length_squared'] = df_eng['petal_length'] ** 2
    df_eng['petal_width_squared'] = df_eng['petal_width'] ** 2
    
    # Total size features
    df_eng['total_length'] = df_eng['sepal_length'] + df_eng['petal_length']
    df_eng['total_width'] = df_eng['sepal_width'] + df_eng['petal_width']
    
    return df_eng

df_engineered = engineer_features(df)
print(f"\n🔧 Feature Engineering Complete!")
print(f"Original features: 4")
print(f"Engineered features: {len(df_engineered.columns) - 1}")

# 3. Prepare features and target
feature_cols = [col for col in df_engineered.columns if col != 'species']
X = df_engineered[feature_cols]
y = df_engineered['species']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features (important for LogisticRegression and SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📦 Data Split:")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# 6. Train Multiple Models
print("\n" + "="*70)
print("MODEL TRAINING & EVALUATION")
print("="*70)

models = {}
results = {}

# Model 1: Logistic Regression with hyperparameter tuning
print("\n1️⃣  Training Logistic Regression...")
lr_params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000]
}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='accuracy')
lr_grid.fit(X_train_scaled, y_train)
lr_model = lr_grid.best_estimator_

y_pred_lr = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5)

models['logistic_regression'] = lr_model
results['logistic_regression'] = {
    'accuracy': lr_accuracy,
    'cv_mean': lr_cv_scores.mean(),
    'cv_std': lr_cv_scores.std(),
    'best_params': lr_grid.best_params_
}

print(f"   ✓ Best params: {lr_grid.best_params_}")
print(f"   ✓ Test Accuracy: {lr_accuracy:.4f}")
print(f"   ✓ CV Score: {lr_cv_scores.mean():.4f} (±{lr_cv_scores.std():.4f})")

# Model 2: Support Vector Machine (SVM)
print("\n2️⃣  Training Support Vector Machine...")
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto']
}
svm_grid = GridSearchCV(SVC(random_state=42, probability=True), svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
svm_model = svm_grid.best_estimator_

y_pred_svm = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)

models['svm'] = svm_model
results['svm'] = {
    'accuracy': svm_accuracy,
    'cv_mean': svm_cv_scores.mean(),
    'cv_std': svm_cv_scores.std(),
    'best_params': svm_grid.best_params_
}

print(f"   ✓ Best params: {svm_grid.best_params_}")
print(f"   ✓ Test Accuracy: {svm_accuracy:.4f}")
print(f"   ✓ CV Score: {svm_cv_scores.mean():.4f} (±{svm_cv_scores.std():.4f})")

# Model 3: Random Forest with hyperparameter tuning
print("\n3️⃣  Training Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(X_train_scaled, y_train)
rf_model = rf_grid.best_estimator_

y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)

models['random_forest'] = rf_model
results['random_forest'] = {
    'accuracy': rf_accuracy,
    'cv_mean': rf_cv_scores.mean(),
    'cv_std': rf_cv_scores.std(),
    'best_params': rf_grid.best_params_
}

print(f"   ✓ Best params: {rf_grid.best_params_}")
print(f"   ✓ Test Accuracy: {rf_accuracy:.4f}")
print(f"   ✓ CV Score: {rf_cv_scores.mean():.4f} (±{rf_cv_scores.std():.4f})")

# 7. Compare Models and Select Best
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(f"\n{'Model':<25} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<12}")
print("-"*70)
for name, result in results.items():
    print(f"{name.replace('_', ' ').title():<25} {result['accuracy']:<12.4f} {result['cv_mean']:<12.4f} {result['cv_std']:<12.4f}")

best_model_name = max(results.items(), key=lambda x: x[1]['cv_mean'])[0]
best_model = models[best_model_name]

print(f"\n🏆 Best Model: {best_model_name.replace('_', ' ').title()}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")

# 8. Detailed Report for Best Model
print(f"\n📋 Detailed Classification Report ({best_model_name.replace('_', ' ').title()}):")
if best_model_name == 'logistic_regression':
    y_pred_best = y_pred_lr
elif best_model_name == 'svm':
    y_pred_best = y_pred_svm
else:
    y_pred_best = y_pred_rf

print(classification_report(y_test, y_pred_best))

# 9. Feature Importance (for Random Forest)
if best_model_name == 'random_forest':
    print("\n📊 Top 10 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']:<30} {row['importance']:.4f}")

# 10. Save all models and preprocessing objects
artifacts = {
    'models': models,
    'best_model_name': best_model_name,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'results': results
}

with open('model.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\n" + "="*70)
print("✅ ALL MODELS SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nSaved artifacts:")
print(f"   • All 3 trained models")
print(f"   • Feature scaler")
print(f"   • Feature column names")
print(f"   • Model performance metrics")
print(f"\nFile: model.pkl")

# 11. Test loading and prediction
print("\n" + "="*70)
print("TESTING MODEL RELOAD")
print("="*70)

with open('model.pkl', 'rb') as f:
    loaded_artifacts = pickle.load(f)

print("\n✓ Model loaded successfully!")
print(f"✓ Available models: {list(loaded_artifacts['models'].keys())}")
print(f"✓ Best model: {loaded_artifacts['best_model_name']}")

# Test prediction
sample_input = df.iloc[0][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.reshape(1, -1)
sample_df = pd.DataFrame(sample_input, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
sample_engineered = engineer_features(sample_df)[feature_cols]
sample_scaled = loaded_artifacts['scaler'].transform(sample_engineered)
prediction = loaded_artifacts['models'][loaded_artifacts['best_model_name']].predict(sample_scaled)

print(f"\n✓ Test prediction successful: {prediction[0]}")
print(f"✓ Expected: {df.iloc[0]['species']}")
print("\n" + "="*70)