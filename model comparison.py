import pandas as pd
import pickle as pkl
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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

model_files = [
    ('credit risk model 1.pkl', 'pickle'),
    ('credit risk model 2.pkl', 'pickle'),
    ('credit risk model 3.pkl', 'pickle'),
    ('credit risk model 4.joblib', 'joblib'),
    ('credit risk model 5.joblib', 'joblib')
]

model_names = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_aucs = []

plt.figure(figsize=(12, 10))

for idx, (model_file, loader_type) in enumerate(model_files):
    if loader_type == 'pickle':
        with open(model_file, 'rb') as file:
            model = pkl.load(file)
    elif loader_type == 'joblib':
        model = joblib.load(model_file)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    precision_scores.append(report['1']['precision'])
    recall_scores.append(report['1']['recall'])
    f1_scores.append(report['1']['f1-score'])
    model_names.append(f'Model {idx + 1}')
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    
    plt.subplot(2, 1, 2)
    plt.plot(fpr, tpr, label=f'Model {idx + 1} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

bar_width = 0.2
index = np.arange(len(model_names))

plt.subplot(2, 1, 1)
plt.bar(index, precision_scores, bar_width, label='Precision', color='b')
plt.bar(index + bar_width, recall_scores, bar_width, label='Recall', color='r')
plt.bar(index + 2 * bar_width, f1_scores, bar_width, label='F1 Score', color='g')

plt.xlabel('')
plt.ylabel('Scores')
plt.title('Precision, Recall, and F1-Score Comparison')
plt.xticks(index + bar_width, model_names)
plt.legend()

plt.subplot(2, 1, 2)
plt.title('ROC Curves for All Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
