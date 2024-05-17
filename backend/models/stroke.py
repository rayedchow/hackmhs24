import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('./data/stroke.csv')

df["bmi"] = df["bmi"].apply(lambda x: 50 if x>50 else x)
df["bmi"] = df["bmi"].fillna(28.4)

df["Residence_type"] = df["Residence_type"].apply(lambda x: 1 if x=="Urban" else 0)
df["ever_married"] = df["ever_married"].apply(lambda x: 1 if x=="Yes" else 0)
df["gender"] = df["gender"].apply(lambda x: 1 if x=="Male" else 0)

 
df = pd.get_dummies(data=df, columns=['smoking_status'])
df = pd.get_dummies(data=df, columns=['work_type'])

std=StandardScaler()
columns = ['avg_glucose_level','bmi','age']
scaled = std.fit_transform(df[['avg_glucose_level','bmi','age']])
scaled = pd.DataFrame(scaled,columns=columns)
df=df.drop(columns=columns,axis=1)
df=df.merge(scaled, left_index=True, right_index=True, how = "left")
df=df.drop(columns='id',axis=1)

X = df.drop(['stroke'], axis=1).values 
y = df['stroke'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None)

model = LogisticRegression()
model.fit(X_train, y_train)

def stroke_prediction(patientData):
	patientData = patientData[X.columns]
	prediction = model.predict(patientData)[0]
	return {
		"predictedRisk": prediction
    }