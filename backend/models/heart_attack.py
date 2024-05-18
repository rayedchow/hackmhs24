import pandas as pd
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# inputting data
data = pd.read_csv('./data/heart_attack.csv')

data.drop(columns = 'education', inplace = True)
data['cigsPerDay'].fillna(value=0.0,inplace=True)

# splitting training sets
X = data.drop(columns=['TenYearCHD'])
target = data['TenYearCHD']

X_train , X_test , y_train , y_test = train_test_split(X ,target ,test_size=None, random_state=42 )

LR_model = make_pipeline(
    SimpleImputer(strategy='mean'),
    MinMaxScaler(),
    LogisticRegression()
)
LR_model.fit(X_train,y_train)

def attack_prediction(patientData):
    patientData = pd.DataFrame([patientData], columns=X.columns)
    prediction = LR_model.predict(patientData)[0]
    
    # Calculate feature importances (coefficients)
    coefficients = LR_model.named_steps['logisticregression'].coef_[0]
    feature_importance = dict(zip(X.columns, coefficients))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Get top 3 impactful factors
    top_factors = sorted_features[:3]
    
    return {
        "predictedRisk": int(round(prediction)),
        "mostImpactfulFactors": list(feature_importance.items())
    }