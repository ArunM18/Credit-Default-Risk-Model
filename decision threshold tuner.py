import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

dataset = pd.read_csv('clean training data.csv')

string_cols = dataset.select_dtypes(include=['object', 'string']).columns
label_encoders = {}

for col in string_cols:
    dataset[col] = dataset[col].fillna('Unknown')
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    label_encoders[col] = le

target_column = 'TARGET'
X = dataset.drop(target_column, axis=1)
y = dataset[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

xgb_model = joblib.load('credit risk model 4.joblib')

y_proba = xgb_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

f1_scores = 2 * (precision * recall) / (precision + recall)

best_threshold = thresholds[f1_scores.argmax()]
best_f1_score = f1_scores.max()

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1-score at optimal threshold: {best_f1_score:.4f}")

y_pred_adjusted = (y_proba >= best_threshold).astype(int)

print("Classification Report with Adjusted Threshold:\n")
print(classification_report(y_test, y_pred_adjusted))

joblib.dump(xgb_model, 'credit risk model 5.joblib')