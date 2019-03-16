# functions and classes to assist in featurization and training
import numpy as np

def reverse_complement(seq):
    """returns the reverse complement of the input DNA sequence"""
    rc = ''
    for bp in seq.upper():
        if bp == 'A':
            rc = 'T'+rc
        if bp == 'T':
            rc = 'A'+rc
        if bp == 'G':
            rc = 'C'+rc
        if bp == 'C':
            rc = 'G'+rc
    return rc

def hamming_distance(seqA,seqB):
    A = np.array([ord(bp) for bp in seqA])
    B = np.array([ord(bp) for bp in seqB])
    return np.count_nonzero(A!=B)

def word_encode(word):
    """Turns a k-mer DNA word into a unique number by interpreting the sequence
    as a number in base 4
    """
    bases = ['A','T','G','C']
    index = 0
    for i,bp in enumerate(word[::-1]):
        index += (bases.index(bp) * (4**i))
    return index

def featurize_sequence(seq,word_len=1):
    """Implement a more interesting/sophisticated featurization:
    Try one-hot matrix featurization, decomposing the sequence into k-mer words
    (which can be length 1) and returning a 2D array as the featurization
    1 = 4
    2 = 16
    3 = 64
    """
    array = np.zeros((4**word_len,len(seq)-word_len+1))
    for i in range(0,len(seq)-word_len+1):
        word = seq[i:i+word_len]
        index = word_encode(word)
        array[index][i] = 1
    return array.flatten()

def interpret_features(flattened_features,word_len=1):
    """Turns a 1D array representing a featurized sequence into the sequence
    of base pairs itself. Requires the original word length as input"""
    bases = ['A','T','G','C']
    sequence = ''
    features = flattened_features.reshape(4**word_len,-1)
    print(features)
    for c in range(len(features[0])):
        for r in range(len(features)):
            f = features[r][c]
            if f > 0.9:
                row_index = str(np.base_repr(r,base=4)) #same digits as word_len
                #since k-mers overlap by k-1, after the first k-mer you only
                #ever need the LAST bp, aka last digit of row_index
                if sequence == '':
                    row_index = '0'*(word_len - len(row_index)) + row_index # add starting 0's as needed
                    sequence = ''.join([ bases[int(i)] for i in row_index])
                else:
                    sequence += bases[int(row_index[-1])]
    return sequence






# NOT USED BUT INTERESTING CONCEPT MAYBE
def basic_featurize_sequence(seq):
    """
    Implements super basic featurization where each sequence gets decomposed
    into an array of floats, one for each base in the sequence.

    T = 0.2
    C = 0.3
    A = 0.6
    G = 0.9

    This decomposition is just qualitatively determined; want the purines and
    pyrimidines to be close in value, want high GC content to look different
    than high AT content (it will here only when A/T and G/C equally likely...)

    Input: DNA sequence as a string
    Output: Array of floating point numbers, same length as input sequence
    """
    array = np.zeros(len(seq))
    mapping = {
        'A': 0.6,
        'T': 0.2,
        'C': 0.3,
        'G': 0.9
    }
    for i,bp in enumerate(seq.upper()): #TODO whatever to upper is in python
        array[i] = mapping[bp]
    return array


def scan_seq_sizes(window_size=8,start=0,stop=-1):
    """
    Builds a fast and not terribly accurate net and tests performance when fed
    in only a subset (window_size bases) of the original 17-bp sequence,
    and slides that window along the full sequence length to determine which
    window(s) contain the most pertinent information
    Ex. for window size 3:
    [A B C] D E F ... -> A [B C D] E F... -> A B [C D E] F... etc

    Inputs: window_size  size of the subsequence to test
            start  the position in the sequence to start the first window
            stop  the position in the sequence to end the final window
    """
    return
