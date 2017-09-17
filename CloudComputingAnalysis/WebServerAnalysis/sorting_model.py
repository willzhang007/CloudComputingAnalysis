__author__ = 'lichaozhang'


import numpy as np
from numpy import genfromtxt
import pandas as pd



#load data
path = '/Users/lichaozhang/Desktop/ECE670Dataset/sorting/'
#data = genfromtxt(path + 'runtime1.csv', delimiter=',')
data = genfromtxt(path + 'runtime2.csv', delimiter=',')
#extract features and targets

features = data[:,0:3]
targets = data[:,7]

print pd.isnull(features)
print pd.isnull(targets)


#shuffle data and split data into train_data and test_data


from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features, targets, test_size=0.1, random_state=0)

print "the number of elements in train_data>>>>>>",X_train.shape, y_train.shape
print "the number of elements in test_data>>>>>>",X_test.shape, y_test.shape


from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

rfr_model = RandomForestRegressor(random_state=0)
x = X_train
y = y_train
rfr = rfr_model.fit(x,y_train)


rfr_predict = rfr.predict(X_test)

rfr_score = explained_variance_score(y_test, rfr_predict, multioutput='uniform_average')
rfr_mean_squared_eror = mean_squared_error(y_test,rfr_predict)

print "the explained_variance_score of RandomForestRegression is >>>>>\n",rfr_score
print "the mean squared error of RandomForestRegression is >>>>>\n",rfr_mean_squared_eror

print "the result that model predict is >>>>>",rfr_predict[:5]
print "the actual result is that>>>>>> ",y_test[:5]



'''
###########################################################################################
#plot relationship of cpu vs runtime

cpu = data[2]

runtime = data[7]


import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


#################################################

#mean_runtime_0 = np.mean(runtime_0)
#print "mean_runtime_0>>>>>>",mean_runtime_0

#cpu = np.array(0.5,1,2,4,8,10,12,18,20,22,24)
fig = plt.figure()
cpu = [0.5,1,2,4,8,18,24]

x = np.array(cpu)
X = x[:, np.newaxis]


runtime = [38,8.5,8.1,8,7.9,6.7,6.5]
y = runtime

#print y


#svr_poly = SVR(kernel='poly',C=1e3,degree=3)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.2)
#poly_model = svr_poly.fit(X, y)
#y_poly = svr_poly.fit(X, y).predict(X)
y_rbf = svr_rbf.fit(X, y).predict(X)

plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
#plt.plot(X, y_lin, c='r', label='Linear model')
#plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('cpu')
plt.ylabel('runtime')
plt.title('Sort 1M random numbers')
plt.legend()
#plt.show()

#################################################
#plot 100K random numbers
fig = plt.figure()

runtime = [2.77,0.72,0.68,0.67,0.61,0.59,0.56]
y = runtime
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.2)
#poly_model = svr_poly.fit(X, y)
#y_poly = svr_poly.fit(X, y).predict(X)
y_rbf = svr_rbf.fit(X, y).predict(X)

plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
#plt.plot(X, y_lin, c='r', label='Linear model')
#plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('cpu')
plt.ylabel('runtime')
plt.title('Sort 100K random numbers')
plt.legend()
#plt.show()

#################################################
#plot 10K random numbers
fig = plt.figure()

runtime = [0.21,0.058,0.051,0.050,0.046,0.044,0.043]
y = runtime
svr_rbf = SVR(kernel='rbf', C=1e3, gamma= 0.2)
#poly_model = svr_poly.fit(X, y)
#y_poly = svr_poly.fit(X, y).predict(X)
y_rbf = svr_rbf.fit(X, y).predict(X)

plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
#plt.plot(X, y_lin, c='r', label='Linear model')
#plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('cpu')
plt.ylabel('runtime')
plt.title('Sort 10K random numbers')
plt.legend()
plt.show()

'''