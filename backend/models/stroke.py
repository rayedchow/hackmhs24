import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load and preprocess the dataset
df = pd.read_csv('./data/stroke.csv')

df["bmi"] = df["bmi"].apply(lambda x: 50 if x > 50 else x)
df["bmi"] = df["bmi"].fillna(28.4)

df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x == "Urban" else 0)
df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x == "Yes" else 0)
df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)

df = pd.get_dummies(data=df, columns=['smoking_status', 'work_type'])

std = StandardScaler()
columns_to_scale = ['avg_glucose_level', 'bmi', 'age']
scaled = std.fit_transform(df[columns_to_scale])
scaled_df = pd.DataFrame(scaled, columns=columns_to_scale)

df = df.drop(columns=columns_to_scale, axis=1)
df = df.merge(scaled_df, left_index=True, right_index=True, how="left")
df = df.drop(columns='id', axis=1, errors='ignore')

# Split data into features and target
X = df.drop(['stroke'], axis=1)
y = df['stroke']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to preprocess and predict new patient data
def stroke_prediction(patientData):
    patient_df = pd.DataFrame([patientData])
    
    # Apply the same preprocessing steps as training data
    patient_df["bmi"] = patient_df["bmi"].apply(lambda x: 50 if x > 50 else x)
    patient_df["bmi"] = patient_df["bmi"].fillna(28.4)
    patient_df["Residence_type"] = patient_df["Residence_type"].apply(lambda x: 1 if x == "Urban" else 0)
    patient_df["ever_married"] = patient_df["ever_married"].apply(lambda x: 1 if x == "Yes" else 0)
    patient_df["gender"] = patient_df["gender"].apply(lambda x: 1 if x == "Male" else 0)
    patient_df = pd.get_dummies(data=patient_df, columns=['smoking_status', 'work_type'])

    # Ensure all columns present
    for col in X.columns:
        if col not in patient_df.columns:
            patient_df[col] = 0

    # Scale numerical columns
    scaled = std.transform(patient_df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled, columns=columns_to_scale)
    patient_df = patient_df.drop(columns=columns_to_scale, axis=1)
    patient_df = patient_df.merge(scaled_df, left_index=True, right_index=True, how="left")

    # Ensure the order of columns matches the training data
    patient_df = patient_df[X.columns]

    # Make prediction
    prediction = model.predict(patient_df)[0]
    
    return {
        "predictedRisk": prediction
    }