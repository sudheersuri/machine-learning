#import necessary modules
import pandas as pd 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import  svm

#step 1 : load data
df=pd.read_csv("winequality.csv")

#step 2 : create instance of your classifier
clf=LinearRegression()

#step 3 : create a data frame containing features
features=np.array(df.drop(['quality'],1))

#step 4 : create a data frame containing labels
labels=np.array(df["quality"])

#step 5 : execute your classifier on given features and labels and create a model
model=clf.fit(features,labels)

#step 6 : check the accuracy of your classifier for a given 
#feautures and labels
#model is accurate if the value is near 1 
print(clf.score(features,labels))

#step 7 : load test data, for which we need to predict the quality of the wine
df_test=pd.read_csv("wine_test.csv")

#step 8 : specify features of test data
test_feautures= np.array(df_test)

#step 9 : predict quality
print(model.predict(test_feautures))
