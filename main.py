import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
import sklearn.preprocessing as pp
from sklearn import metrics
import category_encoders as ce
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate

if __name__ == '__main__':


    df = pd.read_csv('adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
    df2 = pd.read_csv('adult.test', names=['age', 'workclass', 'fnlwgt', 'education', 'education-number',
                                          'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                          'capital-loss', 'hours-per-week', 'native-country', 'income'])
    y = df['income']
    X = df.drop(labels='income',axis=1)
    y_test = df2['income']
    X_test = df2.drop(labels='income', axis=1)
    #X = X.drop([0])
    #X_test = X_test.drop([0])
    le = pp.LabelEncoder()
    # enc = pp.OneHotEncoder(handle_unknown='ignore')
    # enc2 = pp.OneHotEncoder(handle_unknown='ignore')

    enc = ce.count.CountEncoder()

    X = enc.fit_transform(X,y)

    X_test = enc.fit_transform(X_test, y_test)
    y = le.fit_transform(y)
    y_test = le.fit_transform(y_test)
    X = SelectKBest(chi2, k=14).fit_transform(X,y)
    X_test = SelectKBest(chi2, k=14).fit_transform(X_test,y_test)

    #X_2 = enc.fit(X)
    #X_2 = enc.fit_transform(X).toarray()
    #X2_test = enc2.fit(X_test)
    #X2_test = enc2.fit_transform(X_test).toarray()
    gnb = GaussianNB()
    #gnb.fit(X_2, y)

    gnb.fit(X, y)

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X,y)

    bnb = BernoulliNB()
    clf = ComplementNB()

    bnb.fit(X, y)
    clf.fit(X, y)

    y_pred = gnb.predict(X_test)
    y_pred2 = bnb.predict(X_test)
    y_pred3 = clf.predict(X_test)
    y_pred4 = neigh.predict(X_test)
    # cross validation
    scores = cross_val_score(gnb, X, y, cv=5)
    print(scores)
    # accuracy tests
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)
    print("Bernoulli Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred2) * 100)
    print("Complement Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred3) * 100)
    print("KNN model accuracy(in %):", metrics.accuracy_score(y_test, y_pred4) * 100)
    



