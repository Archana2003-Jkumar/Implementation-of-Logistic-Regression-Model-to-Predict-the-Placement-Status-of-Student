# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Find the null and duplicate values.
3. Using logistic regression find the predicted values of accuracy , confusion matrices.
4. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: J. Archana priya
RegisterNumber:  212221230007
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
### Placement data
![image](https://user-images.githubusercontent.com/93427594/235615837-0f68aab4-0d83-4962-afdd-110e5f48f24c.png)
### Salary data
![image](https://user-images.githubusercontent.com/93427594/235615900-375419ec-cb67-45de-a72b-f46ce4b665e4.png)
### Checking the null function()
![image](https://user-images.githubusercontent.com/93427594/235615992-a4ac2851-5055-41d0-833c-b36b1593862a.png)
### Data  duplicate
![image](https://user-images.githubusercontent.com/93427594/235616027-103fd02e-f264-4508-9672-8a171786aeab.png)
###  print data
![image](https://user-images.githubusercontent.com/93427594/235616611-50f31644-81f7-48c5-a5b8-81696992b72d.png)
### Data status
![image](https://user-images.githubusercontent.com/93427594/235616738-900b22b0-8034-404e-a10a-94f356984902.png)
![image](https://user-images.githubusercontent.com/93427594/235616799-25eeaed7-2825-4e60-b66a-96147c7191b8.png)
### Y prediction array 
![image](https://user-images.githubusercontent.com/93427594/235616865-e176bd65-41a6-494e-85da-26f994f24e6b.png)
### Accuracy value
![image](https://user-images.githubusercontent.com/93427594/235617157-aec07f90-8695-48ec-8f92-c49f1ba0c368.png)
### Confusion array
![image](https://user-images.githubusercontent.com/93427594/235617295-e1192aea-e468-4c1d-a9c8-d133dedd37d4.png)
### Classification report
![image](https://user-images.githubusercontent.com/93427594/235617396-eea6964d-64de-44a6-9a83-75971bf5124e.png)
### Prediction of LR
![image](https://user-images.githubusercontent.com/93427594/235617428-54fc6cce-3b64-40aa-b46e-af949bf8ab34.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
