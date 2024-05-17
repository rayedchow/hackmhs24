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

# Extreme Gradient Boost
# xgb = XGBClassifier(learning_rate=0.01, n_estimators=25, max_depth=15,gamma=0.6, subsample=0.52,colsample_bytree=0.6,seed=27, 
#                     reg_lambda=2, booster='dart', colsample_bylevel=0.6, colsample_bynode=0.5)
# xgb.fit(X_train, y_train)

# svc =  SVC(kernel='rbf', C=2)
# svc.fit(X_train, y_train)

def attack_prediction(patientData):
	patientData = pd.DataFrame(patientData)
	patientData = patientData[X.columns]
	print(patientData)
	prediction = LR_model.predict(patientData)[0]
	print(prediction)
	return {
		"predictedRisk": int(prediction)
    }