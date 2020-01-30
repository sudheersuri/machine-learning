#import necessary modules
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#step 1 : load data
df=pd.read_csv("iris.csv")

#step 2 : create instance of your classifier
clf=DecisionTreeClassifier()

#step 3 : decide feautures
features=np.array(df.drop(['species'],1))

#step 3 : decide labels
labels=np.array(df["species"])

#step 4: execute classifier and create model
model=clf.fit(features,labels)

#step 5: check accuracy
print(clf.score(features,labels))

#step 5: load test data
df_test=pd.read_csv("iris_test.csv");

#step 5:decide feautures of test data 
test_features=np.array(df_test)

#step 5:predict class of flower
print(model.predict(test_features))
