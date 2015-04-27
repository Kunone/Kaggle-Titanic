# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 15:14:55 2015

@author: kunwang
"""

import csv as csv
import numpy as np
import pandas as pd
import pylab as p

# load train set
train_file_path = '../data/train.csv'
train_file = open(train_file_path)
train_file_object = csv.reader(train_file)
header= train_file_object.next()
data=[]
for row in train_file_object:
    data.append(row)
data = np.array(data)

#type(data[0:15,5])
#data[:,5].astype(float)
#data[data[:,5]=='',5]
df = pd.read_csv(train_file_path,header=0)
# df.head(3)
#df.dtypes
#df.info()
#df.describe()
#type(df['Age'])
#type(df['Age'][0:10])
#df['Age'].mean()

#df[['Sex','Pclass','Age']]

#df['Age']>60
#len(df[df['Age']>60])
#df[df['Age']>60][['Sex','Pclass','Age','Survived']].head(3)
#df[df['Age'].isnull()][['Sex','Pclass','Age','Survived']]

#for i in range(1,4):
#    print i, len(df[(df['Sex']=='male') & (df['Pclass']==i)])

#df['Age'].hist()
#p.show()
#df['Age'].dropna().hist(bins=16, range=(0,80), alpha=.3)
#p.show()

df['Gender'] = 4
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(np.int)

# build reference table of mediate age on gender/pclass
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender']==i)&(df['Pclass']==j+1)]['Age'].dropna().median()

df['AgeFill'] = df['Age']
df[df['Age'].isnull()][['Gender','Pclass','AgeFill']].head(10)

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull())&(df.Gender==i)&(df.Pclass==j+1),'AgeFill'] = median_ages[i,j]






