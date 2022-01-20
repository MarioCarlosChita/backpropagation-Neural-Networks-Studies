import   numpy as  np  ;


learning_rate =  0.1;
derror_bias =();
derror_weights = ();

# inputs de analise
bias = [];
weights =[];
targets = [];
inputs = [];

def sigmod(x):
    return  (1 /  (1 +  np.exp(-x)));


def compute_gradiente(bias , weights, inputs ,target):
    layer1 =  np.dot(inputs ,  weights) + bias;
    layer2 =  sigmod(layer1);
    predicao =  layer2;
    derror_dpredicao =  2 *  (predicao , target);
    dpredicao_dlayer1  = sigmod_derivate(layer1);
    dbias_dlayer1 =  1;
    dweight_dlayer1  = inputs;

    derror_weights = (
        derror_dpredicao * dpredicao_dlayer1 * dweight_dlayer1
    );

    derror_bias = (
            derror_dpredicao * dpredicao_dlayer1 * dbias_dlayer1
    )
    return derror_weights , derror_bias ;

def sigmod_derivate(x) :
    return sigmod(x)  *  (1 - sigmod(x));


def update(weights , bias , derror_bias , derror_weights):
    bias =  bias +  (learning_rate* derror_bias);
    weights = weights +(learning_rate* derror_weights);
    return  bias, weights;

