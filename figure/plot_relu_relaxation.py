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

def plot_triangle(A, B, C):
    xy = np.array([[A[0], A[1]], [B[0],B[1]], [C[0],C[1]]])
    return Polygon(xy, closed=True, color='r', fill=False, hatch='\\', label="LP Relaxation for ReLU: $a \leq 0\leq b$")

x = np.arange(-4.0,4.0,0.1)
y_relu = relu(x)
y_sigmod = sigmod(x)
y_tanh = tanh(x)

relu_A = (-2., 0.)
relu_B = (0., 0.)
relu_C = (3., 3.)

fig = plt.figure()
plt.plot(x,y_relu,c='k',linestyle='--')
plt.yticks([0], ['0'])
plt.xticks([-2, 0, 3], ['$a$', '0', '$b$'])
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('relu.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()

#plt.plot(x,y_sigmod,c='g',label="Sigmod",linestyle='-.')
#plt.plot(x,y_tanh,c='b',label="Tanh")
plt.plot(x,y_relu,c='k',linestyle='--')
plt.gca().add_patch(plot_triangle(relu_A,relu_B,relu_C))
plt.plot([relu_A[0],relu_A[0]], [-1, relu_A[1]], color='cyan', linestyle=':')
plt.plot([relu_C[0],relu_C[0]], [-1, relu_C[1]], color='cyan', linestyle=':')
plt.yticks([0], ['0'])
plt.xticks([-2, 0, 3], ['$a$', '0', '$b$'])
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('relu_1.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()


relu_A = (-2., 0.)
relu_C = (-0.5, 0.)
plt.plot(x,y_relu,c='k',linestyle='--')
plt.plot([relu_A[0],relu_C[0]], [relu_A[1],relu_C[1]], color='r', label="LP Relaxation for ReLU: $b \leq 0$")
plt.plot([relu_A[0],relu_A[0]], [-1, relu_A[1]], color='cyan', linestyle=':')
plt.plot([relu_C[0],relu_C[0]], [-1, relu_C[1]], color='cyan', linestyle=':')
plt.yticks([0], ['0'])
plt.xticks([-2, 0, -0.5], ['$a$', '0', '$b$'])
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('relu_2.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()


relu_A = (1., 1.)
relu_C = (3., 3.)
plt.plot(x,y_relu,c='k',linestyle='--')
plt.plot([relu_A[0],relu_C[0]], [relu_A[1],relu_C[1]], color='r', label="LP Relaxation for ReLU: $0 \leq a$")
plt.plot([relu_A[0],relu_A[0]], [-1, relu_A[1]], color='cyan', linestyle=':')
plt.plot([relu_C[0],relu_C[0]], [-1, relu_C[1]], color='cyan', linestyle=':')
plt.yticks([0], ['0'])
plt.xticks([1, 0, 3], ['$a$', '0', '$b$'])
plt.ylim([-1,4])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('relu_3.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()