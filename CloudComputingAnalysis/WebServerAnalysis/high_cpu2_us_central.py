__author__ = 'lichaozhang'
from sklearn.datasets import load_files
import numpy as np

#import highcpu-16-us-central



from sklearn import linear_model, decomposition, datasets
import csv

#datasets = load_files(container_path = "~/Desktop/ECE670Dataset/CPU_test/highcpu-2-us-central.txt",encoding = 'utf-8')

#######################################################
#import data
path = '/Users/lichaozhang/Desktop/ECE670Dataset/CPU_test/'

data = np.loadtxt(path +'highcpu-2-us-central.txt', delimiter=",")
#data = np.loadtxt(path +'highcpu16-us-east1-b.txt',delimiter=",")
#data = np.loadtxt(path +'standard16-us-central1-b.txt',delimiter=",")
#data = np.loadtxt(path +'highcpu-4-us-central.txt',delimiter=",")
#data = np.loadtxt(path +'standard8-us-east1-b.txt',delimiter=",")
#data = np.loadtxt(path +'highmem-16-us-central.txt',delimiter=",")
#data = np.loadtxt(path +'highcpu16-us-east1-a.txt',delimiter=",")

data = np.delete(data,np.s_[1:5], 1)
print data

'''
data =np.insert(data,1,122583,axis=1)
data =np.insert(data,2,115105,axis=1)
data =np.insert(data,3,947,axis=1)
data =np.insert(data,4,3816,axis=1)
data =np.insert(data,5,1462,axis=1)
data =np.insert(data,6,21723.78211,axis=1)
data =np.insert(data,7,13,axis=1)

print data


features = data[:,0:8]
targets = data[:,8]

'''
features = data[:,0]
targets = data[:,1]
print "\nthe first 5 elements in features is >>>\n",features[:5]
print "\nthe first 5 elements in labels is >>>\n",targets[:5]

###############################################################################
# plot data
import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(features, targets, 'r.', markersize=6)

plt.title('data distribution')
plt.xlabel('The nth prime number', fontsize=18)
plt.ylabel('Time (s)', fontsize=16)
plt.show()







#print features
#########################################################
#shuffle data and split data into train_data and test_data


from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features, targets, test_size=0.4, random_state=0)

print "the number of elements in train_data>>>>>>",X_train.shape, y_train.shape
print "the number of elements in test_data>>>>>>",X_test.shape, y_test.shape

###########################################################

#compare  LinearRegression and IsotonicRegression
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


###############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

x = X_train
y = y_train

#y_ = ir.fit_transform(x, y)
#ir_model = ir.fit(x,y)
lr = LinearRegression()
lr_model = lr.fit(x, y)  # x needs to be 2d for LinearRegression


###############################################################################
# plot result

segments = [[[i, y[i]]] for i in range(len(x))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(len(x)))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=6)
#plt.plot(x, y_, 'g.-', markersize=6)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()



lr_predict = lr.predict(X_test[:, np.newaxis])
#ir_predict = ir_model.predict(X_test)

lr_score = explained_variance_score(y_test, lr_predict, multioutput='uniform_average')
#ir_score = explained_variance_score(y_test, ir_predict, multioutput='uniform_average')

#ir_mean_squared_eror = mean_squared_error(y_test,ir_predict)
lr_mean_squared_eror = mean_squared_error(y_test,lr_predict)

print "the explained_variance_score of LinearRegression is >>>>>\n",lr_score
#print "the explained_variance_score of IsotonicRegression is >>>>>\n",ir_score
#print "the mean squared error of IsotonicRegression is >>>>>\n",ir_mean_squared_eror
print "the mean squared error of LinearRegression is >>>>>\n",lr_mean_squared_eror
'''
#####################################################################################

#test linear regression>>>>>>>can predict values out of the training set,but does not seem to fit the data very well

import random
'''
features = features[:, np.newaxis]

lr = LinearRegression().fit(features,targets)

lr_predict = lr.predict(features)

lr_score = explained_variance_score(targets, lr_predict, multioutput='uniform_average')
lr_mean_squared_eror = mean_squared_error(targets,lr_predict)

print "the explained_variance_score of LinearRegression is >>>>>\n",lr_score
print "the mean squared error of LinearRegression is >>>>>\n",lr_mean_squared_eror
#a = random.sample(range(400000, 500000), 3)
a = np.arange(100, 1000000, 100)


#b = 200000
#b_predict = lr.predict(b)
#b_targets = targets[2000]



#print "b_predict is >>>>>>>",b_predict
#print "b_targets is >>>>>>>",b_targets
#a = map(float, a.split(','))
#print a
#a = ''.join(a)
a_1 = a[:, np.newaxis]
a_y = lr.predict(a_1)

#print a_y
################################################################################
#plot LinearRegression
segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(x))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(len(x)))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=6)
#plt.plot(a,a_y_highcpu16uscentral,'g.-')
#plt.plot(x, y_, 'g.-', markersize=6)
#plt.plot(a,highcpu-16-us-central.highcpu16uscentral(),'g.-')
plt.plot(a, a_y, 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Linear Fit'), loc='lower right')
plt.title('Linear regression')
#plt.show()
'''
###############################################################################
'''
#try RandomForestRegressor>>>>not good can not predict values out of training set

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


###############################################################################
# plot result

segments = [[[i, y[i]]] for i in range(len(x))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(len(x)))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=6)
#plt.plot(x, y_, 'g.-', markersize=6)
plt.plot(x, rfr.predict(x), 'b.-',markersize=6)
plt.gca().add_collection(lc)
plt.legend(('Data', 'RandomForest Fit'), loc='lower right')
plt.title('RandomForest regression')
plt.show()




import random
'''


'''
rfr = RandomForestRegressor(random_state=0).fit(features,targets)


#a = random.sample(range(400000, 500000), 3)
a = np.arange(400000, 500000, 100)
#a = map(float, a.split(','))
#print a
#a = ''.join(a)
a = a[:, np.newaxis]
a_y = rfr.predict(a)

print a_y

'''

############################################################
#hyperparameter selection (IsotonicRegression) >>>not good because it cant predict values out of the training rage -_-|||

'''
import random

ir = IsotonicRegression(y_min=0,out_of_bounds = 'nan').fit(features,targets)


a = random.sample(range(400000, 500000), 3)

print a
a_y = ir.predict(a)

print a_y

'''
#############################################################
#try svm >>>>>>takes too much time, can not work

'''
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

###############################################################################
# Generate sample data
X = X_train[:, np.newaxis]
y = y_train
#print X
###############################################################################


###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf',C=1e3)
svr_lin = SVR(kernel='linear',C=1e3)
svr_poly = SVR(kernel='poly',C=1e3,degree=2)


y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
#plt.show()

X_test = X_test[:, np.newaxis]
#print "X_test is >>>>>>>",X_test

svr_rbf_predict = svr_rbf.fit(X,y).predict(X_test)

svr_lin_predict = svr_lin.fit(X,y).predict(X_test)
#svr_lin_predict = svr_lin.fit(X,y)

svr_poly_predict = svr_poly.fit(X,y).predict(X_test)

svr_rbf_mean_squared_eror = mean_squared_error(y_test,svr_rbf_predict)
svr_lin_mean_squared_eror = mean_squared_error(y_test,svr_lin_predict)
svr_poly_mean_squared_eror = mean_squared_error(y_test,svr_poly_predict)

print "the mean squared error of SVR_rbf is >>>>>\n",svr_rbf_mean_squared_eror
print "the mean squared error of SVR_lin is >>>>>\n",svr_lin_mean_squared_eror
print "the mean squared error of SVR_ploy is >>>>>\n",svr_poly_mean_squared_eror



################################################################################

# try BayesianRidge,not good,it should be used for  ill-posed problem.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import BayesianRidge, LinearRegression

###############################################################################
# Generating simulated data with Gaussian weigthts
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)  # Create Gaussian data
# Create weigts with a precision lambda_ of 4.
lambda_ = 4.
w = np.zeros(n_features)
# Only keep 10 weights of interest
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# Create noise with a precision alpha of 50.
alpha_ = 50.
noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=n_samples)
# Create the target
y = np.dot(X, w) + noise

###############################################################################
# Fit the Bayesian Ridge Regression and an OLS for comparison
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)

ols = LinearRegression()
ols.fit(X, y)

###############################################################################
# Plot true weights, estimated weights and histogram of the weights
plt.figure(figsize=(6, 5))
plt.title("Weights of the model")
plt.plot(clf.coef_, 'b-', label="Bayesian Ridge estimate")
plt.plot(w, 'g-', label="Ground truth")
plt.plot(ols.coef_, 'r--', label="OLS estimate")
plt.xlabel("Features")
plt.ylabel("Values of the weights")
plt.legend(loc="best", prop=dict(size=12))

plt.figure(figsize=(6, 5))
plt.title("Histogram of the weights")
plt.hist(clf.coef_, bins=n_features, log=True)
plt.plot(clf.coef_[relevant_features], 5 * np.ones(len(relevant_features)),
         'ro', label="Relevant features")
plt.ylabel("Features")
plt.xlabel("Values of the weights")
plt.legend(loc="lower left")

plt.figure(figsize=(6, 5))
plt.title("Marginal log-likelihood")
plt.plot(clf.scores_)
plt.ylabel("Score")
plt.xlabel("Iterations")
plt.show()

'''
