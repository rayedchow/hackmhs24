import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

data = pd.read_csv('./data/heart_disease_stage.csv')
data = data.drop('dataset', axis=1)
data = data.drop('id', axis=1)
X = data.drop('num', axis=1)
y = data['num']

label_encoder = LabelEncoder()

for col in X.columns:
	if X[col].dtype == 'object' or X[col].dtype == 'category':
		X[col] = label_encoder.fit_transform(X[col])
	else:
		pass

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

xgb_model = XGBClassifier(random_state=0)

param_grid = {
	'n_estimators': [50, 100, 150],
	'max_depth': [3, 5, 7],
	'learning_rate': [0.01, 0.1, 0.2],
	'subsample': [0.8, 1.0],
	'colsample_bytree': [0.8, 1.0],
	'gamma': [0, 1, 2]
}

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train, y_train)

def stage_prediction(patientData):
	patientData = patientData[X.columns]
	prediction = best_xgb_model.predict(patientData)[0]
	return {
		"predictedRisk": prediction
    }