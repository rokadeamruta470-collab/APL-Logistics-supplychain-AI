import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("1. Loading Dataset...")
df = pd.read_csv('APL_Logistics (2).csv', encoding='ISO-8859-1')

print("2. Feature Engineering...")
df['shipping_pressure_index'] = df['Days for shipment (scheduled)'] / (df['Order Item Quantity'] + 1)
df['price_discount_ratio'] = df['Order Item Discount Rate'] * df['Product Price']

features = [
    'Type', 'Shipping Mode', 'Customer Segment', 'Market', 'Order Region',
    'Department Name', 'Days for shipment (scheduled)', 'Order Item Quantity',
    'Product Price', 'Order Item Discount Rate', 'shipping_pressure_index', 'price_discount_ratio'
]
target = 'Late_delivery_risk'

X = df[features]
y = df[target]

print("3. Building Preprocessor...")
categorical_cols = ['Type', 'Shipping Mode', 'Customer Segment', 'Market', 'Order Region', 'Department Name']
numerical_cols = ['Days for shipment (scheduled)', 'Order Item Quantity', 'Product Price', 
                  'Order Item Discount Rate', 'shipping_pressure_index', 'price_discount_ratio']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("4. Training Model...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
])

model_pipeline.fit(X_train, y_train)

print("5. Evaluating Model...")
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

print("\n---------------- MODEL METRICS ----------------")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("-----------------------------------------------\n")

print("6. Saving Model File...")
with open('supply_chain_model.pkl', 'wb') as f:
    pickle.dump(model_pipeline, f)

print("✅ Model saved successfully as 'supply_chain_model.pkl'!")

