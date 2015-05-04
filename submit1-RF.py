# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 21:50:54 2015

@author: kunwang
"""

import pandas as pd
import numpy as np
import pylab as p
import csv as csv


### ### ### ### ### missing value is not allowed ### ### ### ### ### 
# Age, Cabin, Embarked
#df[['Age','Pclass','Fare','Sex']]
#df.Fare.describe()
#df.Fare.hist(bins=16)
#df[df.Fare<100].Fare.hist(bins=100, range=(0,100),alpha=.6)
def add_binfare(frame):
    frame['FareBin'] = frame['Fare']
    frame.loc[frame.Fare<10, 'FareBin'] = 0
    frame.loc[(frame.Fare>=10)&(frame.Fare<20),'FareBin'] = 1
    frame.loc[(frame.Fare>=20)&(frame.Fare<30),'FareBin'] = 2
    frame.loc[(frame.Fare>=30)&(frame.Fare<40),'FareBin'] = 3
    frame.loc[frame.Fare>=40,'FareBin'] = 3
    frame.loc[frame.Fare.isnull(),'FareBin'] = 0
    frame.loc[frame.Fare.isnull(),'Fare'] = 0

def add_gender(frame):
    frame['Gender'] = frame['Sex'].map({'female':0,'male':1}).astype(int)

def add_fillage(frame):
    frame['AgeFill'] = frame['Age']
    for i in range(0,2):
        for j in range(0,3):
            median_age=frame[(frame.Gender==i)&(frame.Pclass==j+1)]['Age'].dropna().median()
            frame.loc[(frame.Age.isnull())&(frame.Gender==i)&(frame.Pclass==j+1),'AgeFill'] = median_age

def add_mapEmbarked(frame):
    frame['EmbarkedMap'] = frame['Embarked']
    frame.loc[frame.Embarked.isnull(),'EmbarkedMap'] = 'S'
    frame['EmbarkedMap'] = frame['EmbarkedMap'].map({'S':0,'C':1,'Q':2}).astype(int)

def handle_missing_na(frame):
    add_binfare(frame)
    add_gender(frame)
    add_fillage(frame)
    add_mapEmbarked(frame)

### ### ### ### ### add more features ### ### ### ### ### 
def add_more_features(frame):
    frame['FamilySize'] = frame['Parch']+frame['SibSp']
    frame['Age*Pclass'] = frame.AgeFill*frame.Pclass

### ### ### ### ### only float allowed ### ### ### ### ### 
#train.info()
# remove: name, sex, age, ticket, Fare, Cabin, Embarked
def drop_na_and_float(frame):
    frame_ready = frame.drop(['Name','Sex','Age','Ticket','Cabin','Embarked'], axis=1)
    frame_ready = frame_ready.dropna()
    return frame_ready
    
### ### ### ### ### feature engineering ### ### ### ### ### 
def get_feature_mat(fname):
    #feature engineering in this funciton is applied to both test and train
    df = pd.read_csv("../data/"+fname)
    handle_missing_na(df)
    add_more_features(df)
    return df

train, test = [get_feature_mat(fname) for fname in ['train.csv', 'test.csv']]

#test[test.PassengerId==1044]
#test.info()

### ### ### ### ### back to array ### ### ### ### ### 
train_data = drop_na_and_float(train).values
test_data = drop_na_and_float(test).values

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)
# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,2::],train_data[0::,1])
# Take the same decision trees and run it on the test data
output = forest.predict(test_data[0::,1::])

### ### ### write to file
submission_file = open('submission.csv','wb')
s = csv.writer(submission_file)
s.writerow(['PassengerId', 'Survived'])
for i in range(0,len(output)):
    s.writerow([test_data[i,0].astype(int), output[i].astype(int)])
submission_file.close()
