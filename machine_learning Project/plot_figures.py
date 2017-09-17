__author__ = 'lichaozhang'
import matplotlib.pylab as plt
import os
import numpy as np







#####################################
fig, ax = plt.subplots()
x = (5,10,12,15,20)
y = (0.903970,0.9038188,0.904183,0.90438816,0.904481)
y_pos = np.arange(len(x))

for i, v in enumerate(y):
    ax.text(i, v , str(v), color='red', fontweight='bold')

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Value of RMSE')
plt.title('The selection of iteration')
plt.xlabel('the vlaue of iteration')

plt.ylim([0.9,0.905])
#plt.show()

plt.savefig(os.path.join('iteration.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
plt.clf()






#####################################
#plot the selection of lambda figure


fig, ax = plt.subplots()


#axes.set_xlim([xmin,xmax])


x = (0.01,0.1,0.15,0.2,0.3)
y = (1.0861133,0.914007851,0.90399210,0.903818816,0.928744)
y_pos = np.arange(len(x))

for i, v in enumerate(y):
    ax.text(i, v , str(v), color='red', fontweight='bold')

plt.bar(y_pos, y, align='center', alpha=0.5)
plt.xticks(y_pos, x)
plt.ylabel('Value of RMSE')
plt.title('The selection of lambda')
plt.xlabel('the vlaue of lambda')
#plt.show()
plt.ylim([0.9,1.1])
plt.savefig(os.path.join('lambda.png'), dpi=300, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
plt.clf()

