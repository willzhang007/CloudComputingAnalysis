__author__ = 'lichaozhang'

import numpy as np
from numpy import genfromtxt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



#load data
path = '/Users/lichaozhang/Desktop/ECE670Dataset/sorting/'
data = genfromtxt(path + 'runtime1.csv', delimiter=',')

#extract features and targets

features = data[:,0:7]
targets = data[:,7]
print features.shape
print features
print targets



####################################################################
#plot 3d random_100




features_100 = features[features[:, 0] == 100]
features_100 = np.delete(features_100,np.s_[0], 1)

print features_100[:5]
print targets[:5]


fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')

x = features_100[:,0]
y = features_100[:,1]
data_100 = data[data[:, 0] == 100]
z = data_100[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='b',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_100')

#######################################################################
#plot random_1000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_1000 = features[features[:, 0] == 1000]

x = features_1000[:,1]
y = features_1000[:,2]
data_1000 = data[data[:, 0] == 1000]
z = data_1000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='g',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_1000')

#######################################################################
#plot random_5000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_5000 = features[features[:, 0] == 5000]

x = features_5000[:,1]
y = features_5000[:,2]
data_5000 = data[data[:, 0] == 5000]
z = data_5000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='black',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_5000')


#######################################################################
#plot random_10000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_10000 = features[features[:, 0] == 10000]

x = features_10000[:,1]
y = features_10000[:,2]
data_10000 = data[data[:, 0] == 10000]
z = data_10000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='r',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_10000')


#######################################################################
#plot random_50000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_50000 = features[features[:, 0] == 50000]

x = features_50000[:,1]
y = features_50000[:,2]
data_50000 = data[data[:, 0] == 50000]
z = data_50000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='pink',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_50000')

#######################################################################
#plot random_100000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_100000 = features[features[:, 0] == 100000]

x = features_100000[:,1]
y = features_100000[:,2]
data_100 = data[data[:, 0] == 100000]
z = data_100[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='orange',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_100000')


#######################################################################
#plot random_500000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_500000 = features[features[:, 0] == 500000]

x = features_500000[:,1]
y = features_500000[:,2]
data_500000 = data[data[:, 0] == 500000]
z = data_500000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='purple',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')

plt.title('random_500000')


#######################################################################
#plot random_1000000

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')


features_1000000 = features[features[:, 0] == 1000000]

x = features_1000000[:,1]
y = features_1000000[:,2]
data_1000000 = data[data[:, 0] == 1000000]
z = data_1000000[:,7]

print x[:5]
print y[:5]
print z[:5]

ax.scatter(x,y,z,c ='y',marker ='o')

ax.set_xlabel('cpu number ')
ax.set_ylabel('memory size ')
ax.set_zlabel('runtime')
plt.title('random_1000000')
plt.show()
'''
#######################################################################
#plot figure of data with different features when random =100

fig = plt.figure()

features_100 = features[features[:, 0] == 100]

x = features_100[:,1]
y = data[:,7]


#print x
#print y
'''

'''
import pandas as pd
import matplotlib.pyplot as plt

# Due to an agreement with the ChessGames.com admin, I cannot make the data
# for this plot publicly available. This function reads in and parses the
# chess data set into a tabulated pandas DataFrame.
#chess_data = read_chess_data()

# You typically want your plot to be ~1.33x wider than tall.
# Common sizes: (10, 7.5) and (12, 9)
plt.figure(figsize=(12, 9))

# Remove the plot frame lines. They are unnecessary chartjunk.
ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ensure that the axis ticks only show up on the bottom and left of the plot.
# Ticks on the right and top of the plot are generally unnecessary chartjunk.
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Make sure your axis ticks are large enough to be easily read.
# You don't want your viewers squinting to read your plot.
plt.xticks(fontsize=14)
plt.yticks(range(0.0001, 0.0006, 0.0001), fontsize=14)

# Along the same vein, make sure your axis labels are large
# enough to be easily read as well. Make them slightly larger
# than your axis tick labels so they stand out.
plt.xlabel("Elo Rating", fontsize=16)
plt.ylabel("Count", fontsize=16)

# Plot the histogram. Note that all I'm passing here is a list of numbers.
# matplotlib automatically counts and bins the frequencies for us.
# "#3F5D7D" is the nice dark blue color.
# Make sure the data is sorted into enough bins so you can see the distribution.
plt.hist(list(chess_data.WhiteElo.values) + list(chess_data.BlackElo.values),
         color="#3F5D7D", bins=100)

# Always include your data source(s) and copyright notice! And for your
# data sources, tell your viewers exactly where the data came from,
# preferably with a direct link to the data. Just telling your viewers
# that you used data from the "U.S. Census Bureau" is completely useless:
# the U.S. Census Bureau provides all kinds of data, so how are your
# viewers supposed to know which data set you used?
plt.text(1300, -5000, "Data source: www.ChessGames.com | "
         "Author: Randy Olson (randalolson.com / @randal_olson)", fontsize=10)

# Finally, save the figure as a PNG.
# You can also save it as a PDF, JPEG, etc.
# Just change the file extension in this call.
# bbox_inches="tight" removes all the extra whitespace on the edges of your plot.
plt.savefig("chess-elo-rating-distribution.png", bbox_inches="tight");

'''

