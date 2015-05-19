# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:44:13 2015

@author: kunwang
"""

import pandas as pd
import numpy as np
import pylab as p
import string
from sklearn.ensemble import RandomForestClassifier

path = ['../data/train.csv', '../data/test.csv']
traindf, testdf = [pd.read_csv(x) for x in path]

def get_substring_from_list(a_string, a_list):
    for s in a_list:
        if string.find(a_string, s)!=-1:
            return s
    return np.nan
    
def cleanData(df):
    # clear Fare
    df.Fare = df.Fare.map(lambda x: np.nan if(x==0) else x)
    classmean = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df.Fare = df[['Fare','Pclass']].apply(lambda x: classmean[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)
    
    # clear Age
    meanage = np.mean(df.Age)
    df.Age = df.Age.fillna(meanage)
    
    # clear Cabin
    df.Cabin = df.Cabin.fillna('Unknown')
    
    # clear Embarked
    from scipy.stats import mode
    embarkedmode = mode(df.Embarked)[0][0]
    df.Embarked = df.Embarked.fillna(embarkedmode)
    
    # add Title
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                'Don', 'Jonkheer']
    df['Title'] = df.Name.map(lambda x: get_substring_from_list(x,title_list))
    #replacing all titles with mr, mrs, miss, master
    def replace_titles(x):
        title=x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
    df['Title']=df.apply(replace_titles, axis=1)
    
    #Turning cabin number into Deck
    cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
    df['Deck']=df['Cabin'].map(lambda x: get_substring_from_list(x, cabin_list))

    #Creating new family_size column
    df['Family_Size']=df['SibSp']+df['Parch']

    df['Age*Class']=df['Age']*df['Pclass']
    
    df['Fare_Per_Person']=df['Fare']/(df['Family_Size']+1)
    
    return df
    
def clean_convert_to_float(df):
    df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    df['Title'] = df['Title'].map({'Mr':0,'Mrs':1,'Miss':2, 'Master':3}).astype(int)
    df['Deck'] = df['Deck'].map({'Unknown':0,'C':1,'E':2, 'G':3, 'D':4, 'A':5, 'B':6, 'F':7, 'T':8}).astype(int)
    df['Sex'] = df['Sex'].map({'male':0,'female':1}).astype(int)
    df['AgeCode'] = df.Age.values.codes
    df['Age*ClassCode'] = df['Age*Class'].values.codes
    df['FareCode'] = df.Fare.values.codes
    df['Fare_Per_PersonCode'] = df.Fare_Per_Person.values.codes
    return df
    
def clean_bin(train, test):
    N=len(train)
    M=len(test)
    test=test.rename(lambda x: x+N)
    joint_df=train.append(test)
    
    data_type_dict={'Pclass':'ordinal', 'Sex':'nominal', 
                    'Age':'numeric', 
                    'Fare':'numeric', 'Embarked':'nominal', 'Title':'nominal',
                    'Deck':'nominal', 'Family_Size':'ordinal', 
                    'Fare_Per_Person':'numeric', 'Age*Class':'numeric'}
    for column in data_type_dict:
        if data_type_dict[column]=='numeric':
            joint_df[column] = pd.qcut(joint_df[column], 7)
     
    train=joint_df.ix[range(N)]
    test=joint_df.ix[range(N,N+M)]
    return train, test


traindf = cleanData(traindf)
testdf = cleanData(testdf)
traindf, testdf = clean_bin(traindf, testdf)
traindf = clean_convert_to_float(traindf)
testdf = clean_convert_to_float(testdf)

drop_columns = ['Cabin','Name','Ticket','Age','Age*Class','Fare','Fare_Per_Person']

traindf_target = traindf['Survived']
traindf_features = traindf.drop('Survived',axis=1).drop(drop_columns,axis=1)
testdf_features = testdf.drop(['Survived'],axis=1).drop(drop_columns,axis=1)

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(traindf_features.values, traindf_target.values)
output = forest.predict(testdf_features.values)

predictiondf = pd.DataFrame(testdf['PassengerId'])
predictiondf['Survived'] = output
predictiondf.to_csv('submit.csv', index=False)
