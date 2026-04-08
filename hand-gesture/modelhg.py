import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_FILE = "C:/Users/daksh/code/ml-lab_hand-gesture/gesture_data.csv"  

df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} rows")
print(f"\nSamples per gesture:")
print(df['label'].value_counts())

X = df.drop(columns=['label']).values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
print("\nTraining model...")
model.fit(X_train, y_train)
print("Training complete!")

y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nPer-gesture breakdown:")
print(classification_report(y_test, y_pred))

MODEL_OUT = "C:/Users/daksh/code/ml-lab_hand-gesture/modelf.pkl"
joblib.dump(model, MODEL_OUT)
print(f"Model saved to: {MODEL_OUT}")