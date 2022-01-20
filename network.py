import numpy as np;


class NetWork:

    def __init__(self, inputs,weight , interacao ,bia , target):
        self.inputs = inputs;
        self.weights = weight;
        self.target =  target;
        self.interacao =  interacao;
        self.bia = bia;
        self.learning_rate = 0.1;

    def  _sigmoid(self ,x):
         return (1 / (1 / np.exp(-x)));

    def _sigmoid_deriv(self ,x):
        return self._sigmoid(x) *(1 -self._sigmoid(x));

    def  testar(self,input):
          camada1 = np.dot(input,self.weights) + self.bia;
          camada2 = self._sigmoid(camada1);
          return camada2 ;

    def _gradiente_descente(self , input , alvo):
          camada1 = np.dot(input , self.weights) +self.bia;
          camada2 =  self._sigmoid(camada1);
          predicao = camada2;
          dcusto_dpredicao =  2  *  (predicao - alvo);
          dpredicao_dcamada1 = self._sigmoid_deriv(camada1);
          dbia_dcamada1  =  1;
          dweight_dcamada1 = input;
          derro_dbia  = (
             dcusto_dpredicao *  dpredicao_dcamada1 *  dbia_dcamada1
          )
          derro_dweight = (
              dcusto_dpredicao * dcusto_dpredicao *  dweight_dcamada1
          )
          return derro_dbia ,derro_dweight


    def update(self,derro_dbia ,derro_dweight):
        self.bia =  self.bia  -  self.learning_rate * derro_dbia;
        self.weights = self.weights - self.learning_rate * derro_dweight;

    def treinar(self, input , targets):
        for index  in range(self.interacao):
            # gradiente descendente
            derro_dbia ,derro_dweight =   self._gradiente_descente(self.inputs ,self.target);
            # update weights and bias
            self.update( derro_dbia,derro_dweight);






