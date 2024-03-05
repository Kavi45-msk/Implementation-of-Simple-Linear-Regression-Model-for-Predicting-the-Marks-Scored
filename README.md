# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kavi M S
RegisterNumber:  21222322004
*/
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
*/
```

## Output:
df.head()
![Screenshot 2024-03-05 091733](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/e725baec-c503-43fc-8e67-717b1e57c874)

df.tail()
![Screenshot 2024-03-05 092003](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/b41af24e-b590-4ab5-8c4d-5eb4123317e0)

Values of X:
![Screenshot 2024-03-05 092024](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/319378a4-9464-4b1d-ba71-18f7c30f79f4)

Values of Y:
![Screenshot 2024-03-05 092032](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/267c60ca-c93b-4d31-af55-7f4a9d359b69)

Values of Y prediction:
![Screenshot 2024-03-05 092038](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/a458b09f-f040-4a57-a6e1-348890980b74)

Values of Y test:
![Screenshot 2024-03-05 092044](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/2ed6e61f-8514-4e46-9526-5c4c4593a07f)

Training set graph:
![Screenshot 2024-03-05 092057](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/14dd6ce1-a30c-403d-954f-b314e794f480)

Test set graph:
![Screenshot 2024-03-05 092108](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/78bf8e11-7d84-43d8-a1e1-4797877bf0fb)

Value of MSE,MAE & RMSE:
![Screenshot 2024-03-05 092114](https://github.com/Kavi45-msk/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147457752/151847c6-78ee-41ab-b893-3ad658d7b141)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
