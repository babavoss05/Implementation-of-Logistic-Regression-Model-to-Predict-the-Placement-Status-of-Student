# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries such as pandas module to read the corresponding csv file.
2. Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the corresponding dataset values.
4. Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.
5. Predict the values of array using the variable y_pred.
6. Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.
7. Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.
8. End the program.

## Program:

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Gokul   
RegisterNumber:  212221220013
```py
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### HEAD OF THE DATA :
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/cf3d47f5-34c4-468f-9aa6-6b2752f4bb48)
### COPY HEAD OF THE DATA :
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/5b80d2fd-7abe-46d2-be94-546100736dbb)
### NULL AND SUM :
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/d33cb49b-14b6-4171-a256-e05d765f17c0)
### DUPLICATED :
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/16c87faf-72e1-4f23-b1ec-905c35619a58)
### X VALUE :
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/0616d6ad-db31-44b2-8b62-4d3d07fe2717)
### Y VALUE:
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/cd78ba19-ca6d-47c3-b50d-4534eadbb0b5)
### PREDICTED VALUES:
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/07d53740-ca55-4a38-b158-0c7915967dec)
### ACCURACY:
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/70e8c580-7ca7-47df-8728-6886d73a5440)
### CONFUSION MATRIX:
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/426cbb55-ca39-43ae-bb86-27e19efa23e4)
### CLASSIFICAION REPORT:
![image](https://github.com/babavoss05/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/103019882/0f8e95d6-946b-4417-83e4-f5795a9ed528)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
