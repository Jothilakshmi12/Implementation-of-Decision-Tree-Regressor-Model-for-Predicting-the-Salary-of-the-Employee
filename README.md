# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and display basic information, including null values and class
distribution of the left column.
2. Encode the categorical salary column using Label Encoding.
3. Define the features ( X ) and target ( y ) by selecting relevant columns.
4. Split the data into training and testing sets (80-20 split).
5. Initialize a Decision Tree Classifier with the entropy criterion and train it on the training data.
6. Predict the target values for the test set.
7. Calculate and display the model's accuracy.
8. Compute and display the confusion matrix for the predictions.
9. Predict the left status for a new employee sample.

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
