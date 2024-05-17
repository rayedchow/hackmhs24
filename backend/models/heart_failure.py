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
	patientData = patientData[ds.columns]
	prediction = voting_classifier.predict(patientData)[0]
	return {
		"predictedRisk": prediction
    }