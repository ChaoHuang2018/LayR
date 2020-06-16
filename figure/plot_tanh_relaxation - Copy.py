import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
import numpy as np
import math

def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

def tanh(x):
    y = np.tanh(x)
    return y

def relu(x):
    y = x.copy()
    y[y<0]=0
    return y

def plot_box(A, B, C, D):
    xy = np.array([[A[0], A[1]], [B[0],B[1]], [C[0],C[1]], [D[0],D[1]]])
    return Polygon(xy, closed=True, color='r', fill=False, hatch='\\', label='LP relaxation for Max operation')

x = np.arange(-4.0,4.0,0.1)

x_list = [-3, -1, 1, 3]
y_list = [0.5, 1.1, 0.0, 0.8]

plt.figure()
plt.xticks([-3, -1, 1, 3], ['$x_0$', '$x_1$', '$x_2$', '$x_3$'])
plt.yticks([2.0, 2.2, 3.0, 3.4], ['$b_0$', '$b_1$', '$b_2$', '$b_3$'])
plt.scatter(x_list[0], y_list[0], c='cyan', s=10)
plt.scatter(x_list[1], y_list[1], c='b', s=10)
plt.scatter(x_list[2], y_list[2], c='g', s=10)
plt.scatter(x_list[3], y_list[3], c='k', s=10)
plt.plot([x_list[0],x_list[0]], [-1, y_list[0]], color='cyan', linestyle=':')
plt.plot([x_list[1],x_list[1]], [-1, y_list[1]], color='b', linestyle=':')
plt.plot([x_list[2],x_list[2]], [-1, y_list[2]], color='g', linestyle=':')
plt.plot([x_list[3],x_list[3]], [-1, y_list[3]], color='k', linestyle=':')
plt.plot([-5, 5], [2.0, 2.0], color='cyan', linestyle='--')
plt.plot([-5, 5], [2.2, 2.2], color='b', linestyle='--')
plt.plot([-5, 5], [3.0, 3.0], color='g', linestyle='--')
plt.plot([-5, 5], [3.4, 3.4], color='k', linestyle='--')
# LP
plt.gca().add_patch(plot_box([-4, 2.0], [4, 2.0], [4, 1.1], [-4, 1.1]))
# MILP
# plt.plot([-4, 4], [1.1, 1.1], color='r', linestyle='-', label='MILP representation for Max operation')
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('max_LP.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()

