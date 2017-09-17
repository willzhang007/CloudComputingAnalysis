__author__ = 'lichaozhang'

from numpy import genfromtxt
import numpy as np
#load data
path = '/Users/lichaozhang/Desktop/ECE670Dataset/sorting/'
#data = genfromtxt(path + 'runtime1.csv', delimiter=',')
data = genfromtxt(path + 'runtime2.csv', delimiter=',')


runtime_0 = data[data[:,1] == 1]
runtime_0_1000000 =runtime_0[runtime_0[:,0]==1000000]
runtime_0_1000000 = runtime_0_1000000[:,7]
#print "runtime_0_1000000>>>>>>>>",runtime_0[:5]

n = np.array(runtime_0_1000000)
print('with cpu=1,number=1000000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_500000 =runtime_0[runtime_0[:,0]==500000]
runtime_0_500000 = runtime_0_500000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_500000[:5]

n = np.array(runtime_0_500000)
print('with cpu=1,number=500000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_200000 =runtime_0[runtime_0[:,0]==200000]
runtime_0_200000 = runtime_0_200000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_200000[:5]

n = np.array(runtime_0_200000)
print('with cpu=1,number=200000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_100000 =runtime_0[runtime_0[:,0]==100000]
runtime_0_100000 = runtime_0_100000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_100000[:5]

n = np.array(runtime_0_100000)
print('with cpu=1,number=100000: mean %f, standard deviation %f' % (n.mean(), n.std()))


############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_50000 =runtime_0[runtime_0[:,0]==50000]
runtime_0_50000 = runtime_0_50000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_50000[:5]

n = np.array(runtime_0_50000)
print('with cpu=1,number=50000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_20000 =runtime_0[runtime_0[:,0]==20000]
runtime_0_20000 = runtime_0_20000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_20000[:5]

n = np.array(runtime_0_20000)
print('with cpu=1,number=20000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_10000 =runtime_0[runtime_0[:,0]==10000]
runtime_0_10000 = runtime_0_10000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_10000[:5]

n = np.array(runtime_0_10000)
print('with cpu=1,number=10000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_5000 =runtime_0[runtime_0[:,0]==5000]
runtime_0_5000 = runtime_0_5000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_5000[:5]

n = np.array(runtime_0_5000)
print('with cpu=1,number=5000: mean %f, standard deviation %f' % (n.mean(), n.std()))

############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_1000 =runtime_0[runtime_0[:,0]==1000]
runtime_0_1000 = runtime_0_1000[:,7]
#print "runtime_0>>>>>>>>",runtime_0_1000[:5]

n = np.array(runtime_0_1000)
print('with cpu=1,number=1000: mean %f, standard deviation %f' % (n.mean(), n.std()))
############################################################################################
runtime_0 = data[data[:,1] == 1]
runtime_0_100 =runtime_0[runtime_0[:,0]==1000]
runtime_0_100 = runtime_0_100[:,7]
#print "runtime_0>>>>>>>>",runtime_0_100[:5]

n = np.array(runtime_0_100)
print('with cpu=1,number=100: mean %f, standard deviation %f' % (n.mean(), n.std()))