# implement command line stuff for the basic neural network tests
import numpy as np

from .utils import NeuralNetwork

# BASIC STRUCTURE TESTS

# # test_net = NeuralNetwork([8,3,8],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8])
# # print(test_net)
# # print()
# sig = lambda z: 1/(1+np.exp(-z))
# input = np.array([[1,1,0],[1,1,1],[0,0,1],[0,0,0]])
# output = np.array([1,0,1,0]).reshape(4,1)
# print('OUTPUT',output)
# mini_net = NeuralNetwork([3,3,2,1],input,output)
# # for l in mini_net.layers:
# #     print(l.wts)
# weights_0 = np.array([[0.6,0.7,0.2],[0.2,0.9,0.4],[0.7,0.5,0.8]])
# weights_1 = np.array([[0.4,0.6,0.1],[0.9,0.1,0.3]])
# weights_2 = np.array([[0.45,0.55]])
# print(mini_net.layers[0])
# print(mini_net.layers[1])
# print(mini_net.layers[2])
# print()
# mini_net.layers[0].set_weights_arr(weights_0)
# mini_net.layers[1].set_weights_arr(weights_1)
# mini_net.layers[2].set_weights_arr(weights_2)
#
# for i in range(1,101):
#     mini_net.backpropagate()
#     results = mini_net.feedforward()
#     if i%10 == 0:
#         for r in results:
#             print([ round(n,3) for n in r])
#         print()


# BEST PARAMS TEST
# RESULTS AS FOUND BY THIS GRID SEARCH:
# alpha = 1, wd= 0.0001 or 0.00001, starting bias is 0.1 is fine
# alpha = 3, wd = 0.00001 converges REALLY fast

encoder_inputs = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
        [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1] ]
encoder_outputs = encoder_inputs
for alpha in [0.5,1,3]:
    for wd in [0.0001,0.00001]:
            print("\tParams: alpha, wd ",alpha,wd)
            test_net = NeuralNetwork([8,3,8],encoder_inputs,encoder_outputs,
            alpha=alpha,wd=wd)
            for i in range(1,10001):
                test_net.backpropagate()
                if i%5000 == 0:
                    print('**ITERATION',i,"**")
                    results = test_net.feedforward(new_x=encoder_inputs)
                    for r in results:
                        print(r.round(3))
                    print()
                    print()
