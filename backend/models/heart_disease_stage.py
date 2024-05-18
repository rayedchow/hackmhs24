import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the data
df = pd.read_csv('./data/heart_disease_stage.csv')
df = df.drop(['dataset', 'id'], axis=1)

# Identify categorical and numerical columns
categorical_cols = ['slope', 'exang', 'restecg', 'fbs', 'cp', 'sex', 'thal']
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# Impute missing values
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data into features and target
X = df.drop('num', axis=1)
y = df['num']

# Check class distribution
print("Class distribution in the dataset:\n", y.value_counts())

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into train and test sets using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)

# Create a pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model
best_pipeline = grid_search.best_estimator_

# Evaluate the model
y_pred = best_pipeline.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def stage_prediction(patient_data):
    # Ensure the patient data includes only necessary features
    patient_data = {col: patient_data[col] for col in X.columns if col in patient_data}

    # Create DataFrame from patient data
    patient_df = pd.DataFrame([patient_data])

    # Encode categorical features
    for col, le in label_encoders.items():
        if col in patient_df.columns:
            try:
                patient_df[col] = patient_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            except ValueError:
                patient_df[col] = -1  # Handle unseen labels

    # Handle missing or extra features
    missing_features = set(X.columns) - set(patient_df.columns)
    extra_features = set(patient_df.columns) - set(X.columns)
    if missing_features:
        print("Missing features:", missing_features)
        raise ValueError("Missing features in the input data")
    if extra_features:
        print("Extra features in input data:", extra_features)
        patient_df.drop(extra_features, axis=1, inplace=True)

    # Scale the features
    patient_df = best_pipeline.named_steps['scaler'].transform(patient_df)

    # Make prediction
    prediction = best_pipeline.named_steps['model'].predict(patient_df)[0]

    # Get feature importances
    feature_importances = best_pipeline.named_steps['model'].feature_importances_

    # Pair feature importances with feature names
    feature_importance_dict = dict(zip(X.columns, feature_importances))

    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Get top 3 impactful factors
    top_factors = sorted_features[:3]

    return {
        "predictedRisk": int(prediction),
        "mostImpactfulFactors": list(feature_importance_dict.items())
    }