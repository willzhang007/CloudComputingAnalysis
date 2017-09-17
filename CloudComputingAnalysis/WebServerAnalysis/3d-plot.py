__author__ = 'lichaozhang'


__author__ = 'lichaozhang'

import numpy as np

path = '/Users/lichaozhang/Desktop/ECE670Dataset/CPU_test/'

data1 = np.loadtxt(path +'highcpu-2-us-central.txt', delimiter=",")
data2 = np.loadtxt(path +'highcpu-4-us-central.txt', delimiter=",")
data3 = np.loadtxt(path +'highcpu-8-us-central.txt', delimiter=",")
data4 = np.loadtxt(path +'highcpu-16-us-central.txt', delimiter=",")
data5 = np.loadtxt(path +'highmem-2-us-central.txt', delimiter=",")
data6 = np.loadtxt(path +'highmem-4-us-central.txt', delimiter=",")
data7 = np.loadtxt(path +'highmem-8-us-central.txt', delimiter=",")
data8 = np.loadtxt(path +'highmem-16-us-central.txt', delimiter=",")
data9 = np.loadtxt(path +'standard1-us-central1-b.txt', delimiter=",")
data10 = np.loadtxt(path +'standard2-us-central1-b.txt', delimiter=",")
#data11 = np.loadtxt(path +'standard4-us-central1-b.txt', delimiter=",")
data12 = np.loadtxt(path +'standard8-us-central1-b.txt', delimiter=",")
data13 = np.loadtxt(path +'standard16-us-central1-b.txt', delimiter=",")
data14 = np.loadtxt(path +'small-us-central-b.txt', delimiter=",")
data15 = np.loadtxt(path +'micro-us-central1-b.txt', delimiter=",")

######################################################################
#generate data1 highcpu2_us_central
data1 = np.delete(data1,np.s_[1:5], 1)

#data1 =np.insert(data1,1,122583,axis=1)
#data1 =np.insert(data1,2,115105,axis=1)
#data1 =np.insert(data1,3,947,axis=1)
#data1 =np.insert(data1,4,3816,axis=1)
#data1 =np.insert(data1,5,1462,axis=1)
#data1 =np.insert(data1,6,21723.78211,axis=1)
data1 =np.insert(data1,1,1.8,axis=1)
data1 =np.insert(data1,2,2,axis=1)

print "data1 >>>>>>>\n",data1[:3]

####################################################################

#generate data2 highcpu4_us_central

data2 = np.delete(data2,np.s_[1:5], 1)

#data2 =np.insert(data2,1,121845,axis=1)
#data2 =np.insert(data2,2,156456,axis=1)
#data2 =np.insert(data2,3,822,axis=1)
#data2 =np.insert(data2,4,6653,axis=1)
#data2 =np.insert(data2,5,1602,axis=1)
#data2 =np.insert(data2,6,45246.3096,axis=1)
data2 =np.insert(data2,1,3.6,axis=1)
data2 =np.insert(data2,2,4,axis=1)

print "data2 >>>>>>>\n",data2[:3]

####################################################################

#generate data3 highcpu8_us_central

data3 = np.delete(data3,np.s_[1:5], 1)

#data3 =np.insert(data3,1,122283,axis=1)
#data3 =np.insert(data3,2,158057,axis=1)
#data3 =np.insert(data3,3,1263,axis=1)
#data3 =np.insert(data3,4,7070,axis=1)
#data3 =np.insert(data3,5,1607,axis=1)
#data3 =np.insert(data3,6,88627.928876,axis=1)
data3 =np.insert(data3,1,7.2,axis=1)
data3 =np.insert(data3,2,8,axis=1)
print "data3 >>>>>>>\n",data3[:3]
####################################################################

#generate data4 highcpu16_us_central

data4 = np.delete(data4,np.s_[1:5], 1)

#data4 =np.insert(data4,1,121912,axis=1)
#data4 =np.insert(data4,2,157142,axis=1)
#data4 =np.insert(data4,3,1359,axis=1)
#data4 =np.insert(data4,4,8080,axis=1)
#data4 =np.insert(data4,5,1592,axis=1)
#data4 =np.insert(data4,6,166241.4824,axis=1)
data4 =np.insert(data4,1,14.4,axis=1)
data4 =np.insert(data4,2,16,axis=1)
print "data4 >>>>>>>\n",data4[:3]


####################################################################

#generate data5 highmen2_us_central

data5 = np.delete(data5,np.s_[1:5], 1)

#data5 =np.insert(data5,1,121906,axis=1)
#data5 =np.insert(data5,2,113268,axis=1)
#data5 =np.insert(data5,3,1061,axis=1)
#data5 =np.insert(data5,4,3956,axis=1)
#data5 =np.insert(data5,5,1419,axis=1)
#data5 =np.insert(data5,6,21569.156107,axis=1)
data5 =np.insert(data5,1,13,axis=1)
data5 =np.insert(data5,2,2,axis=1)
print "data5 >>>>>>>\n",data5[:3]


####################################################################

#generate data6 highmen4_us_central

data6 = np.delete(data6,np.s_[1:5], 1)

#data6 =np.insert(data6,1,121940,axis=1)
#data6 =np.insert(data6,2,155208,axis=1)
#data6 =np.insert(data6,3,1432,axis=1)
#data6 =np.insert(data6,4,6773,axis=1)
#data6 =np.insert(data6,5,1436,axis=1)
#data6 =np.insert(data6,6,45374.624241,axis=1)
data6 =np.insert(data6,1,26,axis=1)
data6 =np.insert(data6,2,4,axis=1)
print "data6 >>>>>>>\n",data6[:3]

####################################################################

#generate data7 highmen8_us_central

data7 = np.delete(data7,np.s_[1:5], 1)

#data7 =np.insert(data7,1,12968,axis=1)
#data7 =np.insert(data7,2,81442,axis=1)
#data7 =np.insert(data7,3,1068,axis=1)
#data7 =np.insert(data7,4,6933,axis=1)
#data7 =np.insert(data7,5,1594,axis=1)
#data7 =np.insert(data7,6,89560.593339,axis=1)
data7 =np.insert(data7,1,52,axis=1)
data7 =np.insert(data7,2,8,axis=1)
print "data7 >>>>>>>\n",data7[:3]

####################################################################

#generate data8 highmen16_us_central

data8 = np.delete(data8,np.s_[1:5], 1)

#data8 =np.insert(data8,1,121950,axis=1)
#data8 =np.insert(data8,2,156804,axis=1)
#data8 =np.insert(data8,3,1325,axis=1)
#data8 =np.insert(data8,4,6870,axis=1)
#data8 =np.insert(data8,5,1588,axis=1)
#data8 =np.insert(data8,6,172424.35525,axis=1)
data8 =np.insert(data8,1,104,axis=1)
data8 =np.insert(data8,2,16,axis=1)
print "data8 >>>>>>>\n",data8[:3]


####################################################################

#generate data9 standard1_us_central

data9 = np.delete(data9,np.s_[1:5], 1)

#data9 =np.insert(data9,1,73817,axis=1)
#data9 =np.insert(data9,2,126362,axis=1)
#data9 =np.insert(data9,3,1171,axis=1)
#data9 =np.insert(data9,4,1979,axis=1)
#data9 =np.insert(data9,5,1494,axis=1)
#data9 =np.insert(data9,6,13775.053378,axis=1)
data9 =np.insert(data9,1,3.75,axis=1)
data9 =np.insert(data9,2,1,axis=1)
print "data9 >>>>>>>\n",data9[:3]

####################################################################

#generate data10 standard2_us_central

data10 = np.delete(data10,np.s_[1:5], 1)

#data10 =np.insert(data10,1,122005,axis=1)
#data10 =np.insert(data10,2,118995,axis=1)
#data10=np.insert(data10,3,1150,axis=1)
#data10=np.insert(data10,4,3936,axis=1)
#data10=np.insert(data10,5,1538,axis=1)
#data10=np.insert(data10,6,21659.085987,axis=1)
data10=np.insert(data10,1,7.5,axis=1)
data10=np.insert(data10,2,2,axis=1)
print "data10 >>>>>>>\n",data10[:3]

####################################################################

#generate data11 standard4_us_central

#data11 = np.delete(data11,np.s_[1:5], 1)

#data11 =np.insert(data11,1,122102,axis=1)
#data11 =np.insert(data11,2,154182,axis=1)
#data11 =np.insert(data11,3,1145,axis=1)
#data11 =np.insert(data11,4,6652,axis=1)
#data11 =np.insert(data11,5,1526,axis=1)
#data11 =np.insert(data11,6,45384.92086,axis=1)
#data11 =np.insert(data11,7,15,axis=1)
#data11 =np.insert(data11,8,4,axis=1)
#print "data11 >>>>>>>\n",data11[:3]


####################################################################

#generate data12 standard8_us_central

data12 = np.delete(data12,np.s_[1:5], 1)

#data12 = np.insert(data12,1,121539,axis=1)
#data12 = np.insert(data12,2,157132,axis=1)
#data12 = np.insert(data12,3,1192,axis=1)
#data12 = np.insert(data12,4,7884,axis=1)
#data12 = np.insert(data12,5,1544,axis=1)
#data12 = np.insert(data12,6,88569.056186,axis=1)
data12 = np.insert(data12,1,30,axis=1)
data12 = np.insert(data12,2,8,axis=1)
print "data12 >>>>>>>\n",data12[:3]


####################################################################

#generate data13 standard16_us_central

data13 = np.delete(data13,np.s_[1:5], 1)

#data13 = np.insert(data13,1,121952,axis=1)
#data13 = np.insert(data13,2,156728,axis=1)
#data13 = np.insert(data13,3,1184,axis=1)
#data13 = np.insert(data13,4,8450,axis=1)
#data13 = np.insert(data13,5,1602,axis=1)
#data13 = np.insert(data13,6,172592.49302,axis=1)
data13 = np.insert(data13,1,60,axis=1)
data13 = np.insert(data13,2,16,axis=1)
print "data13 >>>>>>>\n",data13[:3]


####################################################################

#generate data14 small_us_central

data14 = np.delete(data14,np.s_[1:5], 1)

#data14 = np.insert(data14,1,36479,axis=1)
#data14 = np.insert(data14,2,81296,axis=1)
#data14 = np.insert(data14,3,960,axis=1)
#data14 = np.insert(data14,4,996,axis=1)
#data14 = np.insert(data14,5,1574,axis=1)
#data14 = np.insert(data14,6,14552.863276,axis=1)
data14 = np.insert(data14,1,1.7,axis=1)
data14 = np.insert(data14,2,1,axis=1)
print "data14 >>>>>>>\n",data14[:3]


####################################################################

#generate data15 micro_us_central

data15 = np.delete(data15,np.s_[1:5], 1)

#data15 = np.insert(data15,1,16643,axis=1)
#data15 = np.insert(data15,2,19483,axis=1)
#data15 = np.insert(data15,3,212,axis=1)
#data15 = np.insert(data15,4,950,axis=1)
#data15 = np.insert(data15,5,307,axis=1)
#data15 = np.insert(data15,6,4149.980288,axis=1)
data15 = np.insert(data15,1,0.6,axis=1)
data15 = np.insert(data15,2,1,axis=1)
print "data15 >>>>>>>\n",data15[:3]




####################################################################

#generate combine data
data = np.concatenate((data1,data2),axis=0)
data = np.concatenate((data,data3),axis=0)
data = np.concatenate((data,data4),axis=0)
data = np.concatenate((data,data5),axis=0)
data = np.concatenate((data,data6),axis=0)
data = np.concatenate((data,data7),axis=0)
data = np.concatenate((data,data8),axis=0)
data = np.concatenate((data,data9),axis=0)
data = np.concatenate((data,data10),axis=0)
data = np.concatenate((data,data12),axis=0)
data = np.concatenate((data,data13),axis=0)
data = np.concatenate((data,data14),axis=0)
data = np.concatenate((data,data15),axis=0)
print "data combine >>>>>>>\n",data[:3]

features = data[:,0:3]
targets = data[:,3]

print "\nthe first 3 elements in features is >>>\n",features[:3]
print "\nthe first 3 elements in labels is >>>\n",targets[:3]

####################################################################







####################################################################
#plot 3d figures
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

x = data[:,0]
y = data[:,1]
z = data[:,3]

ax.scatter(x,y,z,c ='g',marker ='o')

ax.set_xlabel('x axis')
ax.set_xlabel('y axis')
ax.set_xlabel('z axis')

plt.show()



