import numpy as np;
from network  import NetWork;


inputs = np.array([[0,0]]);
target = np.array([0]);
weight = np.array([[0] ,[0]]);
bia    = np.array([[0]]);

n = NetWork(inputs,weight,10,bia,target);
n.treinar(inputs , target);


# in_predition = np.array([[0,0]]);
# valor =  n.testar(in_predition);

#
# print(valor);


