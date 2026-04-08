import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report



model = joblib.load("C:/Users/daksh/code/ml-lab_hand-gesture/modelf.pkl")
df_new = pd.read_csv("C:/Users/daksh/code/ml-lab_hand-gesture/gesture_data2.csv")  

X_new = df_new.drop(columns=['label']).values
y_new = df_new['label'].values

y_pred_new = model.predict(X_new)

print(f"Fresh data accuracy: {accuracy_score(y_new, y_pred_new) * 100:.2f}%")
print("\nPer-gesture breakdown:")
print(classification_report(y_new, y_pred_new))

