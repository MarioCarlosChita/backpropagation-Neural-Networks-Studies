# inputs = np.array([2, 1.5]);
# weights_1 = np.array([1.45, -0.66]) ;
# bias = np.array([0.0]);
#
#
# multi = MultiPerceptron(inputs, bias , weights_1);
#
# target = 0
# predicao  = multi.treinar();
#
#
#
# ## Buscando o Erro
# Error1 = multi.MSE(predicao , target);
# print("Error1 de :" + str(Error1[0]));
# ## Direcao da  Recta
#
# direcao =multi.direcaoRecta(predicao ,target);
#
#
# multi.updates(direcao);
# predicao =  multi.treinar();
#
# Error2 = multi.MSE(predicao , target);
# print("Error2 de :"+str(Error2[0]));
#
