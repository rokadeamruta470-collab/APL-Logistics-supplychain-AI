import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import joblib

print("==================================================")
print("  APL LOGISTICS - ML MODEL PERFORMANCE EVALUATION ")
print("==================================================\n")

# 1. Load Dataset
df = pd.read_csv('APL_Logistics (2).csv', encoding='latin1')

# 2. Key Features Selection
features = [
    'Days for shipping (real)', 'Days for shipment (scheduled)', 
    'Order Item Quantity', 'Order Item Product Price', 
    'Order Item Discount Rate', 'Sales', 'Shipping Mode', 
    'Customer Segment', 'Market', 'Department Name'
]
target = 'Late_delivery_risk'

data = df[features + [target]].dropna().copy()

# Encoding Categoricals
le_dict = {}
for col in ['Shipping Mode', 'Customer Segment', 'Market', 'Department Name']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

X = data[features]
y = data[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train Models
print("Training Logistic Regression (Baseline)...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

print("Training Random Forest Classifier (Advanced)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save best model for backend dashboard
joblib.dump(rf_model, 'supply_chain_model.pkl')

# 5. Evaluate Function
def print_metrics(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    auc = roc_auc_score(y_test, y_proba)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n--- {model_name} Metrics ---")
    print(f"ROC-AUC Score  : {auc:.4f}")
    print(f"Precision      : {prec:.4f}")
    print(f"Recall         : {rec:.4f}")
    print(f"F1 Score       : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return auc, prec, rec, f1, y_pred, y_proba

print_metrics("Logistic Regression Benchmark", log_reg, X_test, y_test)
rf_auc, rf_prec, rf_rec, rf_f1, rf_pred, rf_proba = print_metrics("Random Forest Model", rf_model, X_test, y_test)

# 6. Generate & Save Confusion Matrix Plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['On-Time (0)', 'Late (1)'], yticklabels=['On-Time (0)', 'Late (1)'])
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\n[+] Confusion Matrix plot saved as 'confusion_matrix.png'")

# 7. Generate & Save Feature Importance Plot
plt.figure(figsize=(8, 5))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title("Feature Importances for Delay Risk Prediction")
plt.bar(range(X.shape[1]), importances[indices], align="center", color="#6366f1")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("[+] Feature Importance plot saved as 'feature_importance.png'")

print("\n==================================================")
print("  EVALUATION COMPLETED SUCCESSFULLY! ")
print("==================================================")