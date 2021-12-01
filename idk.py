import os
import numpy as np
from pandas import read_csv
import category_encoders as ce
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#df = pd.read_csv('adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
#                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
#                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
#df2 = pd.read_csv('adult.test', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
#                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
#                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])


def loadData(filename):
    d = read_csv(filename, names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
    ds = d.values
    #split into input, output
    X = ds[:, :-1]#all columns except last
    y = ds[:,-1]#only last column, ie 'income'
    X = X.astype(str)

    return X, y

X_train, y_train = loadData('adult.data')
X_test, y_test = loadData('adult.test')
#print('Train', X_train.shape, y_train.shape)
#print('Test', X_test.shape, y_test.shape)

oe = OrdinalEncoder()
oe2 = OrdinalEncoder()
oe.fit(X_train)
oe2.fit(X_test)

X_train = oe.transform(X_train)
X_test = oe2.transform(X_test)

le = LabelEncoder()
le2 = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le2.fit_transform(y_test)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

model = LogisticRegression(solver='lbfgs',max_iter = 10000)
model.fit(X_train, y_train)
y_hat = model.predict(X_test)
accuracy = accuracy_score(y_test, y_hat)
print('Logistic Regression Accuracy: %.2f' % (accuracy*100))

