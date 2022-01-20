import numpy as np


class MultiPerceptron:
      inputs = [];
      weights =  [];
      bias = [];

      def __init__(self ,inputs, bias, weights):
         self.bias =  bias;
         self.weights =  weights;
         self.inputs  = inputs;

      def MSE(self ,predicao, alvo):
          mse = np.square(predicao-alvo);
          return mse;

      def treinar(self):
          layer1 = np.dot(self.inputs , self.weights) +  self.bias;
          layer2  = self.sigmod(layer1);
          return layer2;

      def sigmod(self,x):
          return (1 / (1 + np.exp(-x)));

      def direcaoRecta(self ,predicao , alvo):
          return 2 *  (predicao - alvo);

      def updates(self, derivada):
         self.weights = self.weights -derivada;

