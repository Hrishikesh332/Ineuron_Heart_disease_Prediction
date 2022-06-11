import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import mean_absolute_error

df=pd.read_csv('heart_disease_dataset.csv')
print(df.head())
print(df.info())
print(df.describe())
sns.boxplot(x=df['age'])
sns.boxplot(x=df['chol'])
sns.boxplot(x=df['ca'])
sns.boxplot(x=df['thal'])

Q1_ca = df.ca.quantile(0.25)
Q3_ca = df.ca.quantile(0.75)
Q1_thal = df.thal.quantile(0.25)
Q3_thal = df.thal.quantile(0.75)

IQR_ca=Q3_ca-Q1_ca
IQR_thal=Q3_thal-Q1_thal

lower_ca=Q1_ca-1.5*IQR_ca
upper_ca=Q3_ca+1.5*IQR_ca

lower_thal=Q1_thal-1.5*IQR_thal
upper_thal=Q3_thal+1.5*IQR_thal

df1=df[(df.ca>lower_ca)&(df.ca<upper_ca)]
df2=df1[(df1.thal>lower_thal)&(df1.thal<upper_thal)]

sns.boxplot(x=df2['ca'])
sns.boxplot(x=df2['thal'])
print(df2.isnull().sum())

features=['age','trestbps','chol','thalach','oldpeak']
s=StandardScaler()
df2[features]=s.fit_transform(df2[features])

x=df2.drop('num', axis=1)
y=df2.num

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=4)
model=GradientBoostingClassifier(n_estimators=500)
model.fit(x_train, y_train)
pred=model.predict(x_test)
print(precision_score(y_test, pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, pred))
