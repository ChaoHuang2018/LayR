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

def tanh_de(x):
    return 1 - (tanh(x)) ** 2

def relu(x):
    y = x.copy()
    y[y<0]=0
    return y

def plot_triangle(a, c):
    B_0 = (tanh(c)-tanh(a)-(tanh_de(c)*c-tanh_de(a)*a))/(tanh_de(a)-tanh_de(c))
    B_1 = tanh_de(a)*(B_0-a) + tanh(a)
    xy = np.array([[a, tanh(a)], [B_0,B_1], [c,tanh(c)]])
    if c <= 0:
        name = "LP Relaxation for Tanh: $b \leq 0$"
    if a >= 0:
        name = "LP Relaxation for Tanh: $0 \leq a$"
    return Polygon(xy, closed=True, color='r', fill=False, hatch='\\', label=name)

def plot_poly(a, c):
    points = [[a, tanh(a)]]
    der = (tanh(c) - tanh(a))/(c-a)
    name = ''
    if (der < tanh_de(a)):
        print('1')
        points.append([c, tanh(c)])
        name = name + ' and $tanh\'_{[a,b]} < tanh\'(a)$'
    else:
        print('2')
        neg = (tanh_de(a) * (0 - a) + tanh(a))
        points.append([0, neg])
        points.append([c, tanh(c)])
        name = name + ' and $tanh\'_{[a,b]} \geq tanh\'(a)$'
    if (der > tanh_de(c)):
        print('3')
        pos = (tanh_de(c) * (0 - c) + tanh(c))
        points.append([0, pos])
        name = name + ' and $tanh\'_{[a,b]} \geq tanh\'(c)$'
    else:
        name = name + ' and $tanh\'_{[a,b]} < tanh\'(c)$'
    return Polygon(np.array(points), closed=True, color='r', fill=False, hatch='\\', label='LP Relaxation for Tanh: \n $a < 0 < b$' + name)


x = np.arange(-4.0,4.0,0.1)
y_relu = relu(x)
y_sigmod = sigmod(x)
y_tanh = tanh(x)

tanh_A = (-2., tanh(-2.))
tanh_C = (-0.5, tanh(-0.5))

# plt.figure()
# plt.plot(x,y_tanh,c='k',linestyle='--')
# plt.yticks([0], ['0'])
# plt.xticks([tanh_A[0], 0, tanh_C[0]], ['$a$', '0', '$b$'])
# plt.gca().add_patch(plot_triangle(tanh_A[0],tanh_C[0]))
# plt.plot([tanh_A[0],tanh_A[0]], [-2, tanh_A[1]], color='cyan', linestyle=':')
# plt.plot([tanh_C[0],tanh_C[0]], [-2, tanh_C[1]], color='cyan', linestyle=':')
# plt.ylim([-2,2])
# plt.xlim([-4,4])
# plt.legend(loc=2)
# plt.savefig('tanh1.pdf',bbox_inches='tight',pad_inches=0.0)
# plt.close()
#
# tanh_A = (0.3, tanh(0.3))
# tanh_C = (3.0, tanh(3.0))
# plt.figure()
# plt.plot(x,y_tanh,c='k',linestyle='--')
# plt.yticks([0], ['0'])
# plt.xticks([tanh_A[0], 0, tanh_C[0]], ['$a$', '0', '$b$'])
# plt.gca().add_patch(plot_triangle(tanh_A[0],tanh_C[0]))
# plt.plot([tanh_A[0],tanh_A[0]], [-2, tanh_A[1]], color='cyan', linestyle=':')
# plt.plot([tanh_C[0],tanh_C[0]], [-2, tanh_C[1]], color='cyan', linestyle=':')
# plt.ylim([-2,2])
# plt.xlim([-4,4])
# plt.legend(loc=2)
# plt.savefig('tanh3.pdf',bbox_inches='tight',pad_inches=0.0)
# plt.close()


tanh_A = (-2.3, tanh(-2.3))
tanh_C = (0.2, tanh(0.2))
plt.figure()
plt.plot(x,y_tanh,c='k',linestyle='--')
plt.yticks([0], ['0'])
plt.xticks([tanh_A[0], 0, tanh_C[0]], ['$a$', '0', '$b$'])
plt.gca().add_patch(plot_poly(tanh_A[0],tanh_C[0]))
plt.plot([tanh_A[0],tanh_A[0]], [-2, tanh_A[1]], color='cyan', linestyle=':')
plt.plot([tanh_C[0],tanh_C[0]], [-2, tanh_C[1]], color='cyan', linestyle=':')
plt.ylim([-2,2])
plt.xlim([-4,4])
plt.legend(loc=2)
plt.savefig('tanh2+.pdf',bbox_inches='tight',pad_inches=0.0)
plt.close()
