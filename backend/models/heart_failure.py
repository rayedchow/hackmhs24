import pandas as pd
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

ds = pd.read_csv('./data/heart_failure.csv')

X = ds.drop('DEATH_EVENT', axis=1)
y = ds['DEATH_EVENT']

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=3, test_size=None)
voting_classifier = VotingClassifier(
    estimators=[('xg', XGBClassifier()), ('lgbm', LGBMClassifier(verbose=-1))],
    voting='soft',
    verbose=False
)

voting_classifier.fit(x_train, y_train)

def failure_prediction(patientData):
    patientData = pd.DataFrame([patientData], columns=X.columns)
    
    # Get predictions from each base model
    xgb_prediction = voting_classifier.named_estimators_['xg'].predict_proba(patientData)[:, 1]
    lgbm_prediction = voting_classifier.named_estimators_['lgbm'].predict_proba(patientData)[:, 1]
    
    # Combine predictions using voting='soft'
    combined_prediction = (xgb_prediction + lgbm_prediction) / 2
    
    # Predicted risk is the average of predictions from base models
    prediction = round(combined_prediction[0])

    # Feature importances approximation based on contributions of base models
    xgb_importance = voting_classifier.named_estimators_['xg'].feature_importances_
    lgbm_importance = voting_classifier.named_estimators_['lgbm'].feature_importances_
    
    # Combine feature importances from base models
    combined_importance = (xgb_importance + lgbm_importance) / 2
    
    # Pair feature importances with feature names
    feature_importance_dict = dict(zip(X.columns, combined_importance))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Get top 3 impactful factors
    top_factors = sorted_features[:3]

    return {
        "predictedRisk": int(prediction),
        "mostImpactfulFactors": list(feature_importance_dict.items())
    }