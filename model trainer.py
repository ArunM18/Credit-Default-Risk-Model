import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint
import pickle as pkl
from xgboost import XGBClassifier
import joblib

# Load dataset
dataset = pd.read_csv('application_train.csv')

# Drop the irrelevant column
dataset = dataset.drop('SK_ID_CURR', axis=1)

# Fill missing values for numeric columns
numeric_cols = dataset.select_dtypes(include=['number']).columns
dataset[numeric_cols] = dataset[numeric_cols].fillna(dataset[numeric_cols].mean())

# Encode categorical columns
string_cols = dataset.select_dtypes(include=['object', 'string']).columns
label_encoders = {}

for col in string_cols:
    dataset[col] = dataset[col].fillna('Unknown')
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col].astype(str))
    label_encoders[col] = le

# Separate target and features
target_column = 'TARGET'
X = dataset.drop(target_column, axis=1)
y = dataset[target_column]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# model 1: unweighted mlp classifier model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)

mlp.fit(X_train, y_train)

train_score = mlp.score(X_train, y_train)
test_score = mlp.score(X_test, y_test)
'''

'''
# model 2: weighted randomforestclassifer mordel
weight_for_0 = 1 / (1 - 0.0807)  # 0 appears 91.93% of the time
weight_for_1 = 1 / 0.0807        # 1 appears 8.07% of the time

class_weights = {0: weight_for_0, 1: weight_for_1}

credit_risk_model = RandomForestClassifier(class_weight=class_weights, random_state=42)

credit_risk_model.fit(X_train, y_train)

train_score = credit_risk_model.score(X_train, y_train)
test_score = credit_risk_model.score(X_test, y_test)
'''

'''
# model 3: randomised search optimisation
weight_for_0 = 1 / (1 - 0.0807)  
weight_for_1 = 1 / 0.0807      

class_weights = {0: weight_for_0, 1: weight_for_1}

credit_risk_model = RandomForestClassifier(class_weight=class_weights, random_state=42)

param_dist = {
    'n_estimators': randint(100, 500), 
    'max_depth': randint(10, 50),     
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4),  
    'bootstrap': [True, False]          
}


random_search = RandomizedSearchCV(
    estimator=credit_risk_model,
    param_distributions=param_dist,
    n_iter=10, 
    cv=3,   
    scoring='recall', 
    verbose=2,
    random_state=42,
    n_jobs=-1 
)

random_search.fit(X_train, y_train)

print(f"Best Parameters: {random_search.best_params_}")

best_model = random_search.best_estimator_

train_score = best_model.score(X_train, y_train)
test_score = best_model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")
'''

# model 4: xgb model
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)  # Equivalent to class weights

xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)

xgb_model.fit(X_train, y_train)

train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)

print(f"Training Accuracy: {train_score:.2f}")
print(f"Test Accuracy: {test_score:.2f}")


joblib.dump(xgb_model, 'credit risk model 4.joblib')
