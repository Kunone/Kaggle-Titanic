# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 14:50:00 2015

@author: kunwang
"""

import csv as csv
import numpy as np

csv_file_object = csv.reader(open('../data/train.csv', 'rb'))
header = csv_file_object.next()

data=[]
for row in csv_file_object:
    data.append(row)
data = np.array(data)

# data[0::,4] take 5th column
# csv.reader treat all as string by default
number_passengers = np.size(data[0::,1])
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survived = number_survived/number_passengers

women_only_stat = data[0::,4]=='female'
man_only_stat = data[0::,4] != 'female'
women_onboard = data[women_only_stat,1].astype(np.float)
man_onboard = data[man_only_stat,1].astype(np.float)
proportion_women_survived = np.sum(women_onboard)/np.size(women_onboard)
proportion_man_survived = np.sum(man_onboard)/np.size(man_onboard)

# for test
test_file = open('../data/test.csv','rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()
predict_file = open('genderbasemodel.csv','wb')
predict_file_object = csv.writer(predict_file)

predict_file_object.writerow(['PassengerId','Survived'])
for row in test_file_object:
    if(row[3]=='female'):
        predict_file_object.writerow([row[0],'1'])
    else:
        predict_file_object.writerow([row[0],'0'])        

test_file.close()
predict_file.close()
