#from .NN_Tracking.code.neuralnetwork import NN
#import outputCNN as oc
import numpy as np
import sympy as sp
import time
from network_parser import nn_controller, nn_controller_details
from numpy import pi, tanh, array, dot
import cvxpy as cp
from gurobipy import *
#import controller_approximation_lib as cal



# test new approach for estimating sigmoid network's output range
##f = nn_controller('nn_17_sigmoid', 'sigmoid')
##NN_controller = nn_controller_details('nn_17_sigmoid', 'sigmoid')
##x = sp.symbols('x:'+ str(NN_controller.num_of_inputs))
##output_l, output_u = bp.output_range_MILP(NN_controller, [[0.3, 0.4], [0.45, 0.57], [0.2, 0.3]], 0)


# test cvxpy
def c(x,y):
    return [x+y<=1]

x = cp.Variable()
y = cp.Variable()


objective = cp.Maximize(2*x+y)
constraints = [0 <= x, 0 <= y]
con = c(x,y)
constraints += con

prob = cp.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
print(x.value)



