# test the sequence featurization functions
import pytest
import numpy as np
from tf import utils
from tf import io

@pytest.mark.parametrize("sequences",
[('AA','AT','AC','AG','TT','TA','TC','TG','CC','CA','CT','CG','GG','GA','GT','GC'),
('ATCCATGC','TTTTTTTT','CCGTACGA','AAAAGGGG')])

# make sure that featurization works both forward and backwards
def test_featurize_unfeaturize(sequences):
    features_1 = np.array([ utils.featurize_sequence(s,word_len=1) for s in sequences])
    features_2 = np.array([ utils.featurize_sequence(s,word_len=2) for s in sequences])
    # after flattening, each set of features is (4**word_len)x(words_per_seq) long
    assert len(features_1[0]) == 4*(len(sequences[0])-1+1) #words_per_seq = len(seq) - (wordlen-1)
    assert len(features_2[0]) == 16*(len(sequences[0])-2+1)
    unfeatures_1 = np.array([ utils.interpret_features(s,word_len=1) for s in features_1 ])
    unfeatures_2 = np.array([ utils.interpret_features(s,word_len=2) for s in features_2 ])
    for i in range(len(sequences)):
        assert sequences[i] == unfeatures_1[i]
        assert sequences[i] == unfeatures_2[i]
