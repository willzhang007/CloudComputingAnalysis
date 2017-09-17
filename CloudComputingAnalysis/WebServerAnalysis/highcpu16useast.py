__author__ = 'lichaozhang'
import numpy as np

path = '/Users/lichaozhang/Desktop/ECE670Dataset/CPU_test/'
data = np.loadtxt(path +'highcpu16-us-east1-b.txt',delimiter=",")

features = data[:,0]
targets = data[:,5]

print "\nthe first 5 elements in features is >>>\n",features[:5]
print "\nthe first 5 elements in labels is >>>\n",targets[:5]

from sklearn import cross_validation

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     features, targets, test_size=0.4, random_state=0)

print "the number of elements in train_data>>>>>>",X_train.shape, y_train.shape
print "the number of elements in test_data>>>>>>",y_test.shape, y_test.shape

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


ir = IsotonicRegression()

x = X_train
y = y_train

y_ = ir.fit_transform(x, y)
ir_model = ir.fit(x,y)
lr = LinearRegression()
lr_model = lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression


###############################################################################
# plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(len(x))]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(len(x)))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=6)
plt.plot(x, y_, 'g.-', markersize=6)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
#plt.show()

lr_predict = lr.predict(X_test[:, np.newaxis])
ir_predict = ir_model.predict(X_test)

#print np.isinf(ir_predict)
#print np.isnan(ir_predict)

print ir_predict[-5:]

lr_score = explained_variance_score(y_test, lr_predict, multioutput='uniform_average')
#ir_score = explained_variance_score(y_test, ir_predict, multioutput='uniform_average')

#ir_mean_squared_eror = mean_squared_error(y_test,ir_predict)
lr_mean_squared_eror = mean_squared_error(y_test,lr_predict)



print "the explained_variance_score of LinearRegression is >>>>>\n",lr_score
#print "the explained_variance_score of IsotonicRegression is >>>>>\n",ir_score
#print "the mean squared error of IsotonicRegression is >>>>>\n",ir_mean_squared_eror
print "the mean squared error of LinearRegression is >>>>>\n",lr_mean_squared_eror


features = features[:, np.newaxis]

lr = LinearRegression().fit(features,targets)

lr_predict = lr.predict(features)

lr_score = explained_variance_score(targets, lr_predict, multioutput='uniform_average')
lr_mean_squared_eror = mean_squared_error(targets,lr_predict)

print "the explained_variance_score of LinearRegression is >>>>>\n",lr_score
print "the mean squared error of LinearRegression is >>>>>\n",lr_mean_squared_eror
#a = random.sample(range(400000, 500000), 3)
a = np.arange(100, 1000000, 100)

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
plt.show()