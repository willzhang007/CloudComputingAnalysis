__author__ = 'lichaozhang'

from numpy import genfromtxt




#load data
path = '/Users/lichaozhang/Desktop/ECE670Dataset/sorting/'
data = genfromtxt(path + 'runtime.csv', delimiter=',')

#extract features and targets

features = data[:,0:3]
targets = data[:,3]

print "the first 5 tuple of features>>>>",features[:5]
print "the first 5 tuple of targets>>>>>",targets[:5]

#shuffle data and split data into train_data and test_data

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features, targets, test_size=0.995, random_state=0)

print "the number of elements in train_data>>>>>>",X_train.shape, y_train.shape
print "the number of elements in test_data>>>>>>",X_test.shape, y_test.shape


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
###############################################################################
# Generate sample data
X = X_train[:,1]
X = X[:, np.newaxis]
y = y_train
print X
print y
###############################################################################


###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf',C=1e3)
#svr_lin = SVR(kernel='linear',C=1e3)
svr_poly = SVR(kernel='poly',C=1e3,degree=2)

rbf_model = svr_rbf.fit(X, y)
#lin_model = svr_lin.fit(X, y)
poly_model = svr_poly.fit(X, y)

y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
#save the trained model

from sklearn.externals import joblib

joblib.dump(rbf_model, 'rbf_model.pkl')
#joblib.dump(lin_model, 'lin_model.pkl')
joblib.dump(poly_model, 'poly_model.pkl')




#print "X_test is >>>>>>>",X_test

X_test = X_test[:,1]
X_test = X_test[:, np.newaxis]

svr_rbf_predict = rbf_model.predict(X_test)

#svr_lin_predict =lin_model.predict(X_test)
#svr_lin_predict = svr_lin.fit(X,y)

svr_poly_predict = poly_model.fit(X,y).predict(X_test)

svr_rbf_mean_squared_eror = mean_squared_error(y_test,svr_rbf_predict)
#vr_lin_mean_squared_eror = mean_squared_error(y_test,svr_lin_predict)
svr_poly_mean_squared_eror = mean_squared_error(y_test,svr_poly_predict)

print "the mean squared error of SVR_rbf is >>>>>\n",svr_rbf_mean_squared_eror
#print "the mean squared error of SVR_lin is >>>>>\n",svr_lin_mean_squared_eror
print "the mean squared error of SVR_ploy is >>>>>\n",svr_poly_mean_squared_eror

###############################################################################
# look at the results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
#plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


####################################################################

#load saved model

clf = joblib.load('rbf_model.pkl')