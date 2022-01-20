import numpy as np ;



# funcao de activacao ;
def sigmod(x):
    return  (1 /  (1 +  np.exp(-x)));


# predicao do valores
def preditive(inputs ,weights , bias):
  layer1 = np.dot(inputs, weights ) +  bias;
  layer2 = sigmod(layer1);
  return  layer2;


def Error(predict ,target):
    # formula do erro ,mais com  aplicacao da derivada;
    # derivate =  np.square(predict - target);
    derivate   =  2 * (predict - target);





inputs = [1, 0];
weights = [0.2,0.5];
bias = np.array([0.2]);



print(preditive(inputs ,weights ,  bias));


# Port And
# [0 , 0 ] => [ 0]
# [0 , 1] => [ 0]
# [1 , 0 ] => [ 0]
# [1 , 1 ] => [ 1]


