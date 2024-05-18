import pandas as pd
from sklearn.compose import ColumnTransformer
from category_encoders import OneHotEncoder
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split 

df = pd.read_csv('./data/heart_disease.csv')

# drop 'State' and 'Sex' columns
df.drop(columns = ['State', 'Sex'], inplace = True)

# Encoding columns
replacement_dict = {'Yes': 1, 'No': 0}
df['HadHeartAttack'] = df['HadHeartAttack'].replace(replacement_dict)
df['HadAngina'] = df['HadAngina'].replace(replacement_dict)

# Create a new column that will be our target column
df['HeartDisease'] = df['HadHeartAttack'] | df['HadAngina']

# Drop old columns
df.drop(columns = ['HadHeartAttack','HadAngina'], inplace = True)
df.drop(columns = 'WeightInKilograms', inplace = True)

df_cat = df.select_dtypes('object')
df_categorical=df_cat.columns

preprocessor = ColumnTransformer(
    transformers=[('onehot', OneHotEncoder(), df_categorical)],
    remainder='passthrough')

X_transformed = preprocessor.fit_transform(df.drop(columns = 'HeartDisease'))
target = df['HeartDisease']

X_train , X_test , y_train , y_test = train_test_split(X_transformed ,target ,test_size=None, random_state=42 )

over = SMOTE(sampling_strategy = 1)
under = RandomUnderSampler(sampling_strategy = 0.1)

X_train_resampled, y_train_resampled = under.fit_resample(X_train, y_train)
X_train_resampled, y_train_resampled = over.fit_resample(X_train_resampled, y_train_resampled)
dummy_classifier = DummyClassifier(strategy = 'most_frequent') 
dummy_classifier.fit(X_train, y_train)

def disease_prediction(patientData):
	# Convert patient data to DataFrame
    patient_df = pd.DataFrame([patientData], columns=df.columns.drop('HeartDisease'))
    
    # Encode categorical variables
    patient_transformed = preprocessor.transform(patient_df)
    prediction = dummy_classifier.predict(patient_transformed)[0]
    return {
		"predictedRisk": prediction
    }