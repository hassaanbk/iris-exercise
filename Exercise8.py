# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing the data set iris
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import os
path = "/Users/Hassaan/Desktop/Exercise 8"
filename = 'iris1.csv'
fullpath = os.path.join(path, filename)
data_hassaan_i = pd.read_csv(fullpath, sep=',')
print(data_hassaan_i)
print(data_hassaan_i.columns.values)
print(data_hassaan_i.shape)
print(data_hassaan_i.describe())
print(data_hassaan_i.dtypes)
print(data_hassaan_i.head(5))
print(data_hassaan_i['Species'].unique())


# Splitting the predictor and target variables
colnames = data_hassaan_i.columns.values.tolist()
predictors = colnames[:4]
target = colnames[4]
print(target)

# splitting the dataset into train and test variables
data_hassaan_i['is_train'] = np.random.uniform(0, 1, len(data_hassaan_i)) <= .75
print(data_hassaan_i.head(5))

# Create two new dataframes, one with the training rows, one with the test rows
train, test = data_hassaan_i[data_hassaan_i['is_train'] ==
                          True], data_hassaan_i[data_hassaan_i['is_train'] == False]
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:', len(test))


dt_hassaan = DecisionTreeClassifier(
    criterion='entropy', min_samples_split=20, random_state=99)
dt_hassaan.fit(train[predictors], train[target])


preds = dt_hassaan.predict(test[predictors])
pd.crosstab(test['Species'], preds, rownames=[
            'Actual'], colnames=['Predictions'])

X = data_hassaan_i[predictors]
Y = data_hassaan_i[target]
# split the data sklearn module
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)


dt1_hassaan = DecisionTreeClassifier(
    criterion='entropy', max_depth=5, min_samples_split=20, random_state=99)
dt1_hassaan.fit(trainX, trainY)
# 10 fold cross validation using sklearn and all the data i.e validate the data
# help(KFold)
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
score = np.mean(cross_val_score(dt1_hassaan, trainX, trainY,
                scoring='accuracy', cv=crossvalidation, n_jobs=1))
score


# Test the model using the testing data
testY_predict = dt1_hassaan.predict(testX)
testY_predict.dtype
# Import scikit-learn metrics module for accuracy calculation
labels = Y.unique()
print(labels)
print("Accuracy:", metrics.accuracy_score(testY, testY_predict))
# Let us print the confusion matrix
print("Confusion matrix \n", confusion_matrix(testY, testY_predict))


cm = confusion_matrix(testY, testY_predict)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax)  # annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['setosa', 'versicolor', 'virginica'])
ax.yaxis.set_ticklabels(['setosa', 'versicolor', 'virginica'])
plt.show()

#Pruning

import numpy as np
num = np.array([1,2,3,4,5,6,7,8,9,10], dtype=int)
list_of_values = []
for x in num:
    dt1_hassaan = DecisionTreeClassifier(criterion='entropy',max_depth=x, min_samples_split=20, random_state=99)
    dt1_hassaan.fit(trainX,trainY)
    list_of_values.append(dt1_hassaan)
pd.crosstab(list_of_values['Values'], num, rownames=[
            'Data'], colnames=['Score'])