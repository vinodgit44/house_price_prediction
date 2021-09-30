#import libraries
import pandas as pd
import numpy as np
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn import  linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
#Load data from csv file

url="../input/house-rent/houses_to_rent.csv"
data=pd.read_csv(url,sep=',')
print(data.head())
#process & filter data
print(data.isnull().any())
data["rent amount"]=data['rent amount'].map(lambda i:int(i[2:].replace(',','')))
data["hoa"]=data['hoa'].map(lambda i:i[2:].replace(',',''))
data['hoa']=data['hoa'].map(lambda i:i.replace('m info','0'))
data['hoa']=data['hoa'].map(lambda i:int(i.replace('cluso','0')))
data["property tax"]=data['property tax'].map(lambda i:i[2:].replace(',',''))
data['property tax']=data['property tax'].map(lambda i:i.replace('m info','0'))
data['property tax']=data['property tax'].map(lambda i:int(i.replace('cluso','0')))
data["fire insurance"]=data['fire insurance'].map(lambda i:int(i[2:].replace(',','')))
data["total"]=data['total'].map(lambda i:int(i[2:].replace(',','')))
print(type('total'))
print(data.head())
le=LabelEncoder()
data['animal']=le.fit_transform(data['animal'])
data['furniture']=le.fit_transform(data['furniture'])
#convert data into numpy array and drop unuseful data
x=np.array(data.drop(['id','total','floor'],axis=1))
y=np.array(data['total'])
print(x)
print(y)
print(len(x))
#split data into test and train dataset
(train_x ,test_x,train_y,test_y)=train_test_split(x,y,train_size=.8,random_state=None,shuffle=True)
print(len(train_x))
print(len(test_x))
#create model
model=RandomForestRegressor()

#test model
model.fit(train_x,train_y)
acc=-model.score(test_x,test_y)
print(acc*100)