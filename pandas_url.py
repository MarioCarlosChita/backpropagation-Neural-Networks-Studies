import pandas as pd ;
import numpy as np  ;
# from scatter import matplotlib ;
import matplotlib as plb;


dataframe = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv');
# print(dataframe.describe().all());
# series_data =  dataframe['Age'];

#
# print(series_data);

#
# Aprendendo Inteligencia  Artificial

import numpy as np ;

def  sigmod(x):
     return  (1 /  (1 +  np.exp(-x)));


def  prediction(inputs,  weights ,bias):
      layer1 =np.dot(inputs , weights) +  bias;
      layer2 = sigmod(layer1);
      return layer2;



inputs =   [1.5 , 1.1];
weights1 =  [0.1, 0.3];
bias =  np.array([0.0])

pred =  prediction(inputs , weights1 , bias);


# pegando o error
target  = 0;
mse  =np.square(pred - target);

print(mse);


