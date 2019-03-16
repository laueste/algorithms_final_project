# the 8-3-8 encoder test
from .utils import NeuralNetwork

def train_encoder(encoder,iterations,verbose=False):
    """Backpropagates the input encoder for the input number of rounds.
    Does not return anything, prints at every 10% of the total number of
    iterations if verbose == True"""
    for i in range(1,iterations+1):
        encoder.backpropagate()
        if verbose == True:
            if i%(iterations/10) == 0:
                print('ITERATION',i)
                results = encoder.feedforward()
                for r in results:
                    print(r.round(3))
                print()


def make_838_encoder(iterations=10000,alpha=3,weight_decay=0.00001,verbose=False):
    """returns a trained 8-3-8 encoder that inputs in the form of one or more
    8-bit binary arrays. If trained with default parameters, will recognize
    and return the input to within 0.1 error."""
    encoder_inputs = [[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1] ]
    encoder = NeuralNetwork([8,3,8],encoder_inputs,encoder_inputs,
                alpha=alpha,wd=weight_decay)
    train_encoder(encoder,iterations,verbose=verbose)
    return encoder
