# classes and general utility functions for implementing a neural network
import numpy as np

## SINGLE LAYER

class Layer:
    """
    Represents a single layer of a neural network.
    Layer is a 2D numpy array of the given dimensions with xavier-normalized
    random weights and a sigmoid activation function.

    - Alter someday to support non-sigmoidal nodes...

    Example array:
            Input1, Input2, Input3
    Node1     0.3    0.15    2.7
    Node2     0.8     1.6    5.6
    Node3     4.1     0.9    0.1
    Node4     3.2     2.4    0.03
    """

    def __init__(self, n_nodes, n_inputs,
                    layer_bias=None,
                    activation_function=None,
                    act_fn_name=None):
        #the activation function for each neuron in the layer
        sigmoid = lambda z: 1 / (1+np.exp(-z))
        self.f = activation_function if activation_function is not None else sigmoid
        # name for the activation function for sensible printing
        self.f_name = act_fn_name if act_fn_name is not None else "sigmoid"
        #the bias value to use for each neuron in the layer. Use default 0.1 b/c of TA conversation.
        self.b = np.ones(n_nodes)*layer_bias if layer_bias is not None else np.ones(n_nodes)*0.1
        #2D array of weights, nodes(r) x inputs(c), use xavier initialization:
        # https://www.quora.com/What-are-good-initial-weights-in-a-neural-network
        # and https://stackoverflow.com/questions/48641192/xavier-and-he-normal-initialization-difference
        arr = np.random.randn(n_nodes,n_inputs) * np.sqrt(1/n_inputs)
        self.wts = arr

    # Overload the __repr__ operator to make printing simpler.
    def __repr__(self):
        lines = [] #make list so "array()" not printed!
        for node in self.wts:
            lines.append( str([ round(e,5) for e in node ]) )
        lines.append("Bias: %s" % [b for b in self.b])
        return "\n".join(lines)

    def set_bias(self,new_bias,change_dimensions_okay=False):
        """Replaces the layer's bias term with the input value"""
        if new_bias.shape == self.b.shape or change_dimensions_okay == True:
            self.b = new_bias
        else:
            raise ValueError("Dimensions Mismatch with self.b disallowed: \
                    %s self vs %s input" % (self.b.shape,new_bias.shape))

    def set_weights_arr(self,array,change_dimensions_okay=False):
        """Replaces the weights array with the given array. Does not allow
        the new array to be a different shape unless the optional arg is True"""
        if array.shape == self.wts.shape or change_dimensions_okay == True:
            self.wts = array
        else:
            raise ValueError("Dimensions Mismatch with self.wts disallowed: \
                    %s self vs %s input" % (self.wts.shape,array.shape))


    def feedforward_layer(self,input_values):
        """
        Calculates the vector of outputs from the whole layer for a given
        input vector:

        activation_function(W0x0 + W1x1 + W2x2 + ...)  for each node in layer

        Input: array-like of values, same length as the layer's number of cols
        Output: np.array of the activation values from each neuron, given the input
        """
        node_outputs = np.matmul(self.wts,input_values) + self.b
        return self.f(node_outputs) #apply AF to z



def array_sanitize(input):
    """helper function to ensure inputs behave well for single and multiple test sets,
    enforces [ [x,y,z] ] 2D array structure for 1D arrays so that transposition works."""
    return input if isinstance(input[0],(list,tuple,np.ndarray)) else np.array([input])


## NEURAL NETWORK

class NeuralNetwork:
    """
    Represents a full neural network with sigmoidal activation functions an
    a Euclidean squared distances loss function with regularization, starting
    biases at 0 and xavier normalization applied to the starting random weights.

    layer_sizes : array-like describing network structure, e.g. [8,3,8],
    inputs_x : array-like for the values of the initial inputs to the network
    true_results_y: array-like for the known answers (labels) to the given input
    alpha: the step size
    wd: the weight decay parameter
    """
    # TODO implement sparsity ^rho in cost function??
    def __init__(self,layer_sizes,inputs_x,true_results_y,alpha=1.0,wd=0.001):
        np.random.seed(314159) # set single random seed for Neural Network class! Thanks Miriam and Taylor
        # housekeeping for good behavior when testing multiple sets of inputs
        self.x = array_sanitize(inputs_x) #enforces 2D structure even for only one input set
        self.y = array_sanitize(true_results_y)
        # check that the inputs and the outputs match the layer dimensions
        if layer_sizes[0] != len(self.x[0]):
            raise ValueError("Dimension Mismatch in Length: %s input layer vs %s input set" %(len(layer_sizes[0]),len(self.x[0])) )
        if layer_sizes[-1] != len(self.y[0]):
            raise ValueError("Dimension Mismatch in Length: %s output layer vs %s answers set" %(len(layer_sizes[-1]),len(self.y[0])) )
        # set constants
        self.alpha = alpha
        self.wd = wd
        self.dim = layer_sizes
        # make Layers for every transition, including inputs->first layer
        layer_dims = zip(layer_sizes[:-1],layer_sizes[1:])
        self.layers = np.array([Layer(n,i) for i,n in layer_dims])

    def __repr__(self):
        #much easier to read printing version
        lines = []
        lines.append("x: %s" % self.x)
        lines.append("y: %s" % self.y)
        lines.append("Layers:")
        for i,layer in enumerate(self.layers):
            lines.append("%s -" %i)
            lines.append(layer.__repr__())
        return "\n".join(lines)


    def feedforward_single(self,input,return_activations=False):
        """
        implements feeding forward for a single set of inputs

        Input: 1D array of input values, optional flag to return all activations
        Output: 1D array of output values from final layer
                    -OR-
                Tuple of (array of outputs, array of activations for each layer)
                NOTE: the first array of activations is the inputs!
        """
        if return_activations == True:
            activation = np.array([np.zeros(n) for n in self.dim])
            activation[0] = input
            for i,layer in enumerate(self.layers):
                output = layer.feedforward_layer(input)
                activation[i+1] = output
                input = output
            return output,activation
        else:
            for layer in self.layers:
                output = layer.feedforward_layer(input)
                input = output
            return output


    def feedforward(self,new_x=None,return_activations=False):
        """
        implements feeding forward through the whole network

        Input: optional array of inputs, optional flag to return all activations
               (if no inputs given, use self.x)
        Output: Array of, for each set of inputs, the array of output values
                from the final layer
                        -OR- (if return_activations set to True)
                Tuple (outputs f.a. input sets, activations f.a. input sets)

                NOTE: the first array of each activations is the inputs!
        """
        #allow but don't require passing in new inputs, copy so we don't alter
        input = array_sanitize(np.copy(new_x)) if new_x is not None else np.copy(self.x)
        #initialize array of outputs
        results = np.array([ np.zeros(len(self.y[0])) for m in input])
        #make array of activation arrays, one for each layer & inputs
        if return_activations == True:
            single_blank_activation = np.array([np.zeros(n) for n in self.dim])
            activations = np.array([single_blank_activation for m in input])
            for m,input_set in enumerate(input): # do this for every test
                results[m],activations[m] = self.feedforward_single(input_set,return_activations=True)
            return results,activations

        #OR just want output, don't need to store activations
        # (for more efficient testing after training!)
        else:
            for m,input_set in enumerate(input):
                results[m] = self.feedforward_single(input_set)
            return results


    def backpropagate(self,new_y=None,new_x=None):
        """
        implements backpropagating error and updating weights through the whole
        network

        NOTE: Hadamard product is np.multiply(A,B)
        NOTE: for sigmoid, derivative s'(z) = s(z)*(1-s(z)), s(z)_i = output_i

        Inputs: optional array of true outputs and optional array of inputs
                (if not given, uses self.y and self.x respectively)
        Outputs: None (updates weights and biases in self.layers)
        """
        #allow but don't require passing in true output labels, copy for safety, ensure 2D
        true_ys = array_sanitize(np.copy(new_y)) if new_y is not None else np.copy(self.y)
        inputs = array_sanitize(np.copy(new_x)) if new_x is not None else np.copy(self.x)
        outputs,activations = self.feedforward(inputs,return_activations=True)

        # initialize arrays to hold partial derivatives of weights and biases
        grad_weights= np.array([ np.zeros(l.wts.shape) for l in self.layers ])
        grad_biases = np.array([ np.zeros(l.b.shape) for l in self.layers ])

        # initialize array to hold deltas for nodes in each layer
        deltas = np.array([ np.zeros(len(l.wts)) for l in self.layers ])

        # accumulate partial derivatives for every input set
        for output,activation,true_y in zip(outputs,activations,true_ys):

            # calculate delta and partial derivatives for output layer
            delta_outer = np.multiply( (output-true_y),(output*(1-output)) )
            grad_weights[-1] += np.array([activation[-2]])*np.array([delta_outer]).T #cast 1D as 2D to ensure proper multiplication
            grad_biases[-1] += delta_outer
            deltas[-1] = delta_outer

            # calcluate delta and partial derivatives for all inner layers, working backwards
            for i in range(len(self.layers)-2,-1,-1):
                l,next_l = self.layers[i],self.layers[i+1]
                a,prev_a = activation[i+1],activation[i] #activation[0] is input
                # calculate delta
                delta = np.multiply( np.matmul(next_l.wts.T,deltas[i+1]), (a*(1-a)) )
                deltas[i] = delta
                # calculate partial derivatives for weights and bias in the layer
                grad_weights[i] += np.atleast_2d([prev_a])*np.atleast_2d([delta]).T #cast both 1Ds as 2D to ensure proper matrix multiplication
                grad_biases[i] += delta

        # Iterate through all layers forward, updating weights
        for i,layer in enumerate(self.layers):
            weight_decay_term = self.wd*layer.wts
            error_term = (1/len(inputs)) * grad_weights[i]
            updated_wts = layer.wts - self.alpha*(error_term+weight_decay_term)
            updated_bias = layer.b - self.alpha*(1/len(inputs))*grad_biases[i]
            layer.set_weights_arr(updated_wts)
            layer.set_bias(updated_bias)
