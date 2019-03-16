# test functions for the neural network class
from nn import utils
import numpy as np
import pytest

# test overall structure (aka that weight arrays are the right dimensions)
@pytest.mark.parametrize("structure,input,output,layer_sizes", [([5,3,7,10,1],
[0,0,0,0,0],[0],[(3,5),(7,3),(10,7),(1,10)])])
#the output and input values are irrelevant here, just the shapes

def test_structure(structure, input, output, layer_sizes):
    nn = utils.NeuralNetwork(structure,input,output)
    for i,layer in enumerate(nn.layers):
        assert layer.wts.shape == layer_sizes[i]


# test layer feedforward output (1D array of len(N_nodes))
@pytest.mark.parametrize("n_nodes,n_inputs,x,weights,raw_answer",
[(2,2,np.array([10,20]),np.array([[1,1],[0,2]]),np.array([30,40]))])

def test_layer_feedforward(n_nodes, n_inputs, x, weights, raw_answer):
    l = utils.Layer(n_nodes,n_inputs)
    l.set_weights_arr(weights)
    ff_output = l.feedforward_layer(x)
    for i in range(len(ff_output)):
        assert round(ff_output[i],8) == round(l.f(raw_answer)[i],8)


#Numbers in examples courtesy of
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
ff = {
'h1_net':0.3775,
'h1_out':0.59327,
'h2_out':0.596884378,
'o1_net':1.105905967,
'o1_out':0.75136507,
'o2_out':0.772928465,
'o1_err':0.274811083,
'o2_err':0.023560026
}

backprop = {
'new_L1':np.array([[0.35891648,0.408666186],[0.511301270,0.561370121]]),
'new_L0':np.array([[0.149780716,0.19956143],[0.24975114,0.29950229]])
}


@pytest.mark.parametrize("layer_sizes,inputs,outputs,weights_0,weights_1,\
b0,b1,alpha,ff",
[([2,2,2],np.array([0.05,0.10]),np.array([0.01,0.99]),np.array([[0.15,0.20],[0.25,0.30]]),
np.array([[0.4,0.45],[0.5,0.55]]),0.35,0.60,0.5,ff)])

def test_feedforward(layer_sizes,inputs,outputs,weights_0,weights_1,b0,b1,alpha,ff):
    test_net = utils.NeuralNetwork(layer_sizes,inputs,outputs,alpha=alpha,wd=0)
    test_net.layers[0].set_weights_arr(weights_0)
    test_net.layers[1].set_weights_arr(weights_1)
    test_net.layers[0].set_bias(np.array([b0,b0]))
    test_net.layers[1].set_bias(np.array([b1,b1]))
    ff_output,ff_activations = test_net.feedforward(return_activations=True)
    assert np.round(ff['h1_out'],6) == np.round(ff_activations[0][1][0],6)
    assert np.round(ff['h2_out'],6) == np.round(ff_activations[0][1][1],6)
    assert np.round(ff['o1_out'],6) == np.round(ff_activations[0][2][0],6)
    assert np.round(ff['o2_out'],6) == np.round(ff_activations[0][2][1],6)
    # squared error! fix this
    #assert np.round(ff['o1_err'],6) == np.round(outputs[0] - ff_output[0][0],6)
    #assert np.round(ff['o2_err'],6) == np.round(ff_output[0][1] - outputs[1],6)


@pytest.mark.parametrize("layer_sizes,inputs,outputs,weights_0,weights_1,\
b0,b1,alpha,backprop",
[([2,2,2],np.array([0.05,0.10]),np.array([0.01,0.99]),np.array([[0.15,0.20],[0.25,0.30]]),
np.array([[0.4,0.45],[0.5,0.55]]),0.35,0.60,0.50,backprop)])

def test_backpropagation(layer_sizes,inputs,outputs,weights_0,weights_1,b0,b1,alpha,backprop):
    test_net = utils.NeuralNetwork(layer_sizes,inputs,outputs,alpha=alpha,wd=0)
    test_net.layers[0].set_weights_arr(weights_0)
    test_net.layers[1].set_weights_arr(weights_1)
    test_net.layers[0].set_bias(np.array([b0,b0]))
    test_net.layers[1].set_bias(np.array([b1,b1]))
    test_net.backpropagate()
    # these are good to 4 decimal places but not after...
    assert round(backprop['new_L0'][0][0],6) == round(test_net.layers[0].wts[0][0],6)
    assert round(backprop['new_L0'][0][1],6) == round(test_net.layers[0].wts[0][1],6)
    assert round(backprop['new_L0'][1][0],6) == round(test_net.layers[0].wts[1][0],6)
    assert round(backprop['new_L0'][1][1],6) == round(test_net.layers[0].wts[1][1],6)

    # these have more error because they compound the error from above...
    assert round(backprop['new_L1'][0][0],6) == round(test_net.layers[1].wts[0][0],6)
    assert round(backprop['new_L1'][0][1],6) == round(test_net.layers[1].wts[0][1],6)
    assert round(backprop['new_L1'][1][0],6) == round(test_net.layers[1].wts[1][0],6)
    assert round(backprop['new_L1'][1][1],6) == round(test_net.layers[1].wts[1][1],6)
