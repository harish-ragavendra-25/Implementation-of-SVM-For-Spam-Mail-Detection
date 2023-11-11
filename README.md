# Implementation-of-SVM-For-Spam-Mail-Detection
## AIM:
To write a program to implement the SVM For Spam Mail Detection.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the required packages.
2. Import the dataset to operate on.
3. Split the dataset.
4. Predict the required output.
5. End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARISH RAGAVENDRA S
RegisterNumber:  212222230045
*/
```
```

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

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
accuracy
```

## Output:
![Screenshot 2023-11-11 171234](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/c63b4886-843a-46f7-89b5-e141a2e4f966)
![Screenshot 2023-11-11 171239](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/438dd2ab-c3eb-4e96-b04c-de55652381ca)
![Screenshot 2023-11-11 171244](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/a9d5a838-4603-423f-ae11-805f8aeea5e6)
![Screenshot 2023-11-11 171248](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/2481e584-ca5f-44d6-80f5-733269ff6ab7)
![Screenshot 2023-11-11 171253](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/aabfe22a-700e-4eb0-aa37-2e610a80501b)
![Screenshot 2023-11-11 171258](https://github.com/harish-ragavendra-25/Implementation-of-SVM-For-Spam-Mail-Detection/assets/114852180/b0142e2e-9d08-4ccb-b4f1-742a4a676077)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
