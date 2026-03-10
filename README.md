# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary python packages using import statements.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Split the dataset using train_test_split.

4.Calculate Y_Pred and accuracy.

5.Print all the outputs.

6.End the Program

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: 
RegisterNumber:  
*/

import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

```

## Output:
## ENCODING :


<img width="845" height="41" alt="image" src="https://github.com/user-attachments/assets/fedddb4b-0456-4f6e-a771-5a73a90c7c6e" />


## HEAD():


<img width="723" height="209" alt="image" src="https://github.com/user-attachments/assets/71097158-cf6c-428a-83d0-7a9285d3b121" />


## isnul().sum :


<img width="317" height="146" alt="image" src="https://github.com/user-attachments/assets/4c42017c-c1eb-471a-bcb3-ae2e38caac01" />


## Prediction of Y :


<img width="879" height="94" alt="image" src="https://github.com/user-attachments/assets/72c2f896-720f-4f15-be5a-433705498efe" />


## Accuracy :


<img width="285" height="42" alt="image" src="https://github.com/user-attachments/assets/7d79d78a-eec1-4cbb-97e1-2a666b8eeaa5" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
