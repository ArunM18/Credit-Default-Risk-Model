import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import pickle as pkl

dataset = pd.read_csv('clean_training_data.csv')

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

credit_risk_model = pkl.load(open("credit risk model 5.pkl", "rb"))

train_score = credit_risk_model.score(X_train, y_train)
test_score = credit_risk_model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")

y_pred = credit_risk_model.predict(X_test)

if hasattr(credit_risk_model, "predict_proba"):
    y_proba = credit_risk_model.predict_proba(X_test)[:, 1]
else:
    from sklearn.preprocessing import minmax_scale
    y_proba = minmax_scale(credit_risk_model.decision_function(X_test))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
