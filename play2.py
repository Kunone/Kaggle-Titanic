# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:50:07 2015

@author: kunwang
"""

# import
import csv as csv
import numpy as np

# read train set
train_file = open('../data/train.csv','rb')
train_file_object = csv.reader(train_file)
train_header = train_file_object.next()
data = []
for row in train_file_object:
    data.append(row)
data = np.array(data)

# to generate the mode
fare_ceiling = 40
data[data[0::,9].astype(np.float)>=fare_ceiling,9] = fare_ceiling-1
# np.max(data[0::,9].astype(np.float))
fare_bracket_size = 10;
number_price_brackets = fare_ceiling/fare_bracket_size
number_classes = len(np.unique(data[0::,2]))
survived_table = np.zeros((2, number_classes, number_price_brackets))

for i in xrange(number_classes):
    for j in xrange(number_price_brackets):
        women_survived_state = data[(data[0::,4]=='female')
                                    & (data[0::,9].astype(np.float)<fare_bracket_size*(j+1))
                                    & (data[0::,9].astype(np.float)>=fare_bracket_size*j)
                                    & (data[0::,2].astype(np.int)==i+1),1]
        man_survived_state = data[(data[0::,4]=='male')
                                    & (data[0::,9].astype(np.float)<fare_bracket_size*(j+1))
                                    & (data[0::,9].astype(np.float)>=fare_bracket_size*j)
                                    & (data[0::,2].astype(np.int)==i+1),1]
        survived_table[0,i,j] = np.mean(women_survived_state.astype(np.float))
        survived_table[1,i,j] = np.mean(man_survived_state.astype(np.float))
survived_table[survived_table!=survived_table] = 0
survived_table[survived_table<0.5] = 0
survived_table[survived_table>=0.5] = 1

# apply the mode to test set
test_file = open('../data/test.csv','rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()
predict_file = open('second.csv','wb')
p = csv.writer(predict_file)
p.writerow(["PassengerId", "Survived"])

for row in test_file_object:
    for j in xrange(number_price_brackets):
        try:
            row[8] = float(row[8])
        except:
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling:
            bin_fare = number_price_brackets - 1
            break
        if row[8]>=j*fare_bracket_size and row[8]<(j+1)*fare_bracket_size:
            bin_fare = j
            break
    #if row[3] == 'female':
    #    p.writerow([row[0], "%d" % int(survived_table[0,float(row[1]-1),bin_fare])])
    #else:
    #    p.writerow([row[0], "%d" % int(survived_table[1,float(row[1]-1),bin_fare])])

    if row[3] == 'female':                             #If the passenger is female
        p.writerow([row[0], "%d" % \
                   int(survived_table[0, float(row[1])-1, bin_fare])])
    else:                                          #else if male
        p.writerow([row[0], "%d" % \
                   int(survived_table[1, float(row[1])-1, bin_fare])])

predict_file.close()
test_file.close()
train_file.close()
