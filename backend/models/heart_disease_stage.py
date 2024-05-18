import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv('./data/heart_disease_stage.csv')
data = data.drop(['dataset', 'id'], axis=1)
X = data.drop('num', axis=1)
y = data['num']

# Encode categorical features
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'category':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from grid search
best_rf_model = grid_search.best_estimator_

# Fit the best model on the entire training set
best_rf_model.fit(X_train_scaled, y_train)

# Function for making predictions
def stage_prediction(patient_data):
    patient_df = pd.DataFrame([patient_data])

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in patient_df.columns:
            patient_df[col] = patient_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Scale the features
    patient_df_scaled = scaler.transform(patient_df)

    # Make prediction
    prediction = best_rf_model.predict(patient_df_scaled)[0]

    return {"predictedRisk": prediction}