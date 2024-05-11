## Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### DATE:12.03.2024
### AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

### Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
1. Import the required libraries
2. Upload and read the guven dataset
3. Check for any null values using the isnull() function
4. From sklearn.tree import DecisionTreeClassifier and use criterian as entropy
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn

### Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Lathika Sunder
RegisterNumber:  212221230054
```
```python
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Employee_EX6.csv")
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
### Output:
### HEAD VALUES:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787343/1b554fa8-a20d-4413-a7a7-95625880f026)
### INFO:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787343/01f7e59d-a396-4a53-9ad4-59ee1d6fb1fc)
### USING LABEL ENCODER
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787343/248b8dcd-fe59-4afc-ab04-80ff98ec2ac2)
### ACCURACY:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787343/a2a99c9c-27df-44e5-9c61-19a88198554d)
### DECISION TREE:
![image](https://github.com/gpavana/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118787343/070a1170-1dd6-48ad-8047-61a9af6cb5dd)
### Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
