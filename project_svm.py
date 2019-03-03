#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 11:32:18 2019

@author: nugroho
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm 
from sklearn import datasets

#bcancer = datasets.load_breast_cancer()
bcancer = pd.read_csv('Project 5/breast-cancer.csv')

X = bcancer.iloc[:,:9]
y = bcancer.iloc[:,9]
print(X.head())
print(y.head())


# transform string ke numeric
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_0 = LabelEncoder()
X.iloc[:, 0] = labelencoder_X_0.fit_transform(X.iloc[:, 0])

labelencoder_X_1 = LabelEncoder()
X.iloc[:, 1] = labelencoder_X_1.fit_transform(X.iloc[:, 1])

labelencoder_X_2 = LabelEncoder()
X.iloc[:, 2] = labelencoder_X_2.fit_transform(X.iloc[:, 2])

labelencoder_X_3 = LabelEncoder()
X.iloc[:, 3] = labelencoder_X_3.fit_transform(X.iloc[:, 3])

labelencoder_X_4 = LabelEncoder()
X.iloc[:, 4] = labelencoder_X_4.fit_transform(X.iloc[:, 4])

labelencoder_X_5 = LabelEncoder()
X.iloc[:, 5] = labelencoder_X_5.fit_transform(X.iloc[:, 5])

labelencoder_X_6 = LabelEncoder()
X.iloc[:, 6] = labelencoder_X_6.fit_transform(X.iloc[:, 6])

labelencoder_X_7 = LabelEncoder()
X.iloc[:, 7] = labelencoder_X_7.fit_transform(X.iloc[:, 7])

labelencoder_X_8 = LabelEncoder()
X.iloc[:, 8] = labelencoder_X_8.fit_transform(X.iloc[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, :9]


# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

print(X)

# memisahkan antara data training dan testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# fitting & coba metode-metode kernel
from sklearn.svm import SVC

linear = SVC(kernel = 'linear', random_state = 0, C=100, gamma=0.2)
linear.fit(X_train, y_train)
print(linear.score(X_test, y_test))

poly = svm.SVC(kernel='poly', degree=3, C=10)
poly.fit(X,y)
print(poly.score(X_test, y_test))

rbf = svm.SVC(kernel='rbf', gamma=0.1, C=5000)
rbf.fit(X,y)
print(poly.score(X_test, y_test))


# corat coret ubah y ke numeric
#y_train
#def change_y_val(x):
#    if x == "no":
#        return 0
#    else:
#        return 1

#y_train_num = map(change_y_val, y_train.values)
#print(type(X_train))
#print(X_train)

#for i in range(0, len(y_train)):
#    if(str(y_train.iloc[i])=="no"):
#        y_train.iloc[i]=0
#    else:
#        y_train.iloc[i]=1


# Model Evaluation
predictions = poly.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# grid search mencari C dan gamma optimal
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

print(classification_report(y_test,grid_predictions))


# coba plot (masih gagal T.T)
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

#bcancer = pd.read_csv('Project 5/breast-cancer.csv')
#df = pd.DataFrame(bcancer)
#X = df.iloc[:, :9]
#y = df.iloc[:, 9]

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, y_train) 

# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(X=X_train, 
                      y=y_train,
                      clf=clf, 
                      legend=2,
                      FileExi)

value=1.5
width=0.75

fig = plot_decision_regions(X_train, y_train, clf=clf,
                  filler_feature_values={2: value, 3:value, 4:value, 5:value, 6:value, 7:value, 8:value},
                  filler_feature_ranges={2: width, 3: width, 4:width, 5:width, 6:width, 7:width, 8:width},
                  legend=2, ax=1)

# Update plot object with X/Y axis labels and Figure Title
plt.xlabel(X.columns[0], size=14)
plt.ylabel(X.columns[1], size=14)
plt.title('SVM Decision Region Boundary', size=16)