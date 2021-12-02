import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
import sklearn.preprocessing as pp
from sklearn import metrics
import category_encoders as ce
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import time

if __name__ == '__main__':

    # load data into numpy array
    # data = np.loadtxt("adult.data", delimiter= ',', dtype=str)
    #
    # # classification data is last column of data
    # y = np.array(data[:,14])
    #
    # # remove last column from data
    # X = np.delete(data,14,1)
    #
    # clf = GaussianNB()
    # # prints for reference
    # print(X[0])
    # print(X[:,0])
    #
    # # convert age to float
    # for x in X[:,0]:
    #     x = float(x)
    
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
    enc2 = ce.count.CountEncoder()

    X = enc.fit_transform(X,y)
    X_test = enc2.fit_transform(X_test, y_test)
    y = le.fit_transform(y)
    y_test = le.fit_transform(y_test)

    
    
    t_start = time.perf_counter()
    gnb = GaussianNB()
    gnb.fit(X, y)
    t_end = time.perf_counter()
    gnb_train_time = t_end - t_start
    
    t_start = time.perf_counter()
    bnb = BernoulliNB()
    bnb.fit(X, y)
    t_end = time.perf_counter()
    bnb_train_time = t_end - t_start

    t_start = time.perf_counter()
    clf = ComplementNB()
    clf.fit(X, y)
    t_end = time.perf_counter()
    clf_train_time = t_end - t_start

    t_start = time.perf_counter()
    lgr = LogisticRegression(solver='lbfgs',max_iter = 10000)
    lgr.fit(X, y)
    t_end = time.perf_counter()
    lgr_train_time = t_end - t_start

    t_start = time.perf_counter()
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X,y)
    t_end = time.perf_counter()
    neigh_train_time = t_end - t_start

    #svc = LinearSVC(max_iter = 1000000)
    #svc.fit(X,y)
    
    t_start = time.perf_counter()
    y_pred = gnb.predict(X_test)
    t_end = time.perf_counter()
    gnb_pred_time = t_end - t_start

    t_start = time.perf_counter()
    y_pred2 = bnb.predict(X_test)
    t_end = time.perf_counter()
    bnb_pred_time = t_end - t_start

    t_start = time.perf_counter()
    y_pred3 = clf.predict(X_test)
    t_end = time.perf_counter()
    clf_pred_time = t_end - t_start

    t_start = time.perf_counter()
    y_pred4 = lgr.predict(X_test)
    t_end = time.perf_counter()
    lgr_pred_time = t_end - t_start

    t_start = time.perf_counter()
    y_pred5 = neigh.predict(X_test)
    t_end = time.perf_counter()
    neigh_pred_time = t_end - t_start

    #y_pred6 = svc.predict(X_test)

    gnbAccuracy = metrics.accuracy_score(y_test, y_pred)*100
    bnbAccuracy = metrics.accuracy_score(y_test, y_pred2)*100
    clfAccuracy = metrics.accuracy_score(y_test, y_pred3)*100
    lgrAccuracy = metrics.accuracy_score(y_test, y_pred4)*100
    neighAccuracy = metrics.accuracy_score(y_test, y_pred5)*100
    #svcAccuracy = metrics.accuracy_score(y_test, y_pred6)*100
    
    print("Gaussian Naive Bayes model accuracy(in %):", gnbAccuracy)
    print("Bernoulli Naive Bayes model accuracy(in %):", bnbAccuracy)
    print("Complement Naive Bayes model accuracy(in %):", clfAccuracy)
    print("Logistic Regression model accuracy(in %):", lgrAccuracy)
    print("KNN model accuracy(in %):", neighAccuracy)
    #print("Support Vector Machine model accuracy(in %):", svcAccuracy)
    print("Gaussian Training time: ", gnb_train_time)
    print("Gaussian Trest time: ", gnb_pred_time)
    print("Bernoulli Training time: ", bnb_train_time)
    print("Bernoulli Trest time: ", bnb_pred_time)
    print("Complement Training time: ", clf_train_time)
    print("Complement Trest time: ", clf_pred_time)
    print("Logistic Regression Training time: ", lgr_train_time)
    print("Logistic Regression Trest time: ", lgr_pred_time)
    print("KNN Training time: ", neigh_train_time)
    print("KNN Trest time: ", neigh_pred_time)
    
    data = {'Gaussian':gnbAccuracy, 'Bernoulli':bnbAccuracy, 'Complement':clfAccuracy,
            'Log. Reg.':lgrAccuracy, 'KNN':neighAccuracy}
    names = list(data.keys())
    values = list(data.values())

    fig = plt.figure(figsize = (10,5))
    plt.ylim(70,85)
    for i in range(len(names)):
        plt.text(i, round(values[i],2), round(values[i],2), ha = 'center')
    plt.bar(names, values, color ='blue', width = 0.5)
    plt.ylabel("Perecent Accuracy")
    plt.title("Accuracy of Model Predictions")
    plt.show()

    x_plot = ['Gaussian', 'Bernoulli', 'Complement', 'Log. Reg.', 'KNN']
    y_train_time = [gnb_train_time, bnb_train_time, clf_train_time, lgr_train_time, neigh_train_time]
    y_pred_time = [gnb_pred_time, bnb_pred_time, clf_pred_time, lgr_pred_time, neigh_pred_time]

    x_axis = np.arange(len(x_plot))
    plt.bar(x_axis - 0.2, y_train_time, 0.4, label = 'Training Time')
    plt.bar(x_axis + 0.2, y_pred_time, 0.4, label = 'Prediction Time')
    plt.xticks(x_axis, x_plot)

    plt.ylabel("Time(Seconds)")
    plt.title("Training and Prediction Timings")
    plt.legend()
    plt.show()
    
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn2 = KNeighborsClassifier(n_neighbors=2)
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn4 = KNeighborsClassifier(n_neighbors=4)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn6 = KNeighborsClassifier(n_neighbors=6)
    knn7 = KNeighborsClassifier(n_neighbors=7)
    knn8 = KNeighborsClassifier(n_neighbors=8)

    knn1.fit(X,y)
    knn2.fit(X,y)
    knn3.fit(X,y)
    knn4.fit(X,y)
    knn5.fit(X,y)
    knn6.fit(X,y)
    knn7.fit(X,y)
    knn8.fit(X,y)

    y_knn1 = knn1.predict(X_test)
    y_knn2 = knn2.predict(X_test)
    y_knn3 = knn3.predict(X_test)
    y_knn4 = knn4.predict(X_test)
    y_knn5 = knn5.predict(X_test)
    y_knn6 = knn6.predict(X_test)
    y_knn7 = knn7.predict(X_test)
    y_knn8 = knn8.predict(X_test)

    knn1_accuracy = metrics.accuracy_score(y_test, y_knn1)*100
    knn2_accuracy = metrics.accuracy_score(y_test, y_knn2)*100
    knn3_accuracy = metrics.accuracy_score(y_test, y_knn3)*100
    knn4_accuracy = metrics.accuracy_score(y_test, y_knn4)*100
    knn5_accuracy = metrics.accuracy_score(y_test, y_knn5)*100
    knn6_accuracy = metrics.accuracy_score(y_test, y_knn6)*100
    knn7_accuracy = metrics.accuracy_score(y_test, y_knn7)*100
    knn8_accuracy = metrics.accuracy_score(y_test, y_knn8)*100

    data = {1:knn1_accuracy, 2:knn2_accuracy,3:knn3_accuracy,4:knn4_accuracy,
            5:knn5_accuracy,6:knn6_accuracy,7:knn7_accuracy,8:knn8_accuracy,}
    names = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize = (10,5))
    for i in range(len(names)):
        plt.text(i+1, round(values[i],2), round(values[i],2), ha = 'center')
    plt.plot( names, values, color ='r', ls = '-')
    plt.ylim(74,80)
    plt.ylabel("Perecent Accuracy")
    plt.xlabel("k-value")
    plt.title("Accuracy of KNN")
    plt.show()
