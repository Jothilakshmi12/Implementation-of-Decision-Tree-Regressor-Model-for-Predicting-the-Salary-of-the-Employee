# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JOTHILAKSHMI P
RegisterNumber:212223110017 
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
X=data[["Position","Level"]]
Y=data[["Salary"]]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,Y_train)
Y_pred = dt.predict(X_test)
from sklearn import metrics
mse=metrics.mean_squared_error(Y_test,Y_pred)
print(mse)
r2=metrics.r2_score(Y_test,Y_pred)
print(r2)
dt.predict([[4,2]])
```

## Output:
## Mean squared error
![image](https://github.com/user-attachments/assets/c5b95cb0-3cef-4e8e-92a4-e04556ed9890)

## r2 score
![image](https://github.com/user-attachments/assets/941c1c2e-8f34-4939-afcb-f1771ff009e2)

## predicted value
![image](https://github.com/user-attachments/assets/d27d7128-8972-4b00-8a6b-2d4c0a776105)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
