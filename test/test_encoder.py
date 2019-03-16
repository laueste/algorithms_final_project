# test functions for the 8-3-8 autoencoder puzzle
from nn import utils,encoder
import pytest

# test that 12345678 -> 12345678 within error of 0.5,0.1
@pytest.mark.parametrize("iterations,error_margin",
[(5000,0.5),(10000,0.1)]) #going to higher iterations takes longer than useful here

def test_encoder(iterations,error_margin):
    answers = [ [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1] ]
    test_net = encoder.make_838_encoder(iterations=iterations)
    results = test_net.feedforward()
    for i,r in enumerate(results):
        assert r[i] > (1.0 - error_margin) #check for high value for desired bit
        for j in range(8): #check all the rest of the bits for low values
            if j != i:
                assert r[j] < (0.0 + error_margin)
