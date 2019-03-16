import numpy as np
from .utils import word_encode,featurize_sequence,interpret_features,hamming_distance
from .io import read_seqs,write_seqs
from .prediction import undersample,oversample,underover,test_parameters
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# PREPARATORY FILE PARSING
np.random.seed(314159)
# Make the negative sequences file (just do this once, and delete the last newline)
# fasta = './data/yeast-upstream-1k-negative.fa'
# pos = './data/rap1-lieb-positives.txt'
# make_negatives_file(fasta,pos)
# pos_seqs = read_seqs('./data/rap1-lieb-positives.txt')
# neg_seqs = read_seqs('./data/rap1-lieb-constructed-negatives.txt')
# screen_by_distance(pos_seqs,neg_seqs)
# close_seqs = read_seqs('./data/rap1-lieb-close-negatives.txt')
# screen_by_distance_reverse(neg_seqs,close_seqs)

# READ IN DATA
print("Reading Files")
rap1_pos = read_seqs('./data/rap1-lieb-positives.txt')
rap1_far_neg = read_seqs('./data/rap1-lieb-far-negatives.txt')
rap1_close_neg = read_seqs('./data/rap1-lieb-close-negatives.txt')
rap1_test = read_seqs('./data/rap1-lieb-test.txt')

# INITIALIZE AND TRAIN MODELS TO ENSEMBLE

# MLP definitions were determined during model selection and written to
# best_estimator_parameters.txt

# Train the models to the datasets they were optimized for during selection
print("Building Classifiers")
# Undersampling
us,us_info = undersample(rap1_pos,rap1_close_neg,rap1_far_neg,
                                                neg_ratio=2,proportion_far=0.1)
X = np.array([ featurize_sequence(s,word_len=1) for s in us[:,0]])
y = np.ravel(label_binarize(us[:,1],classes=['0','1']))
mlp_us = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=10, learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_us.fit(X,y)

# Oversampling
os,os_info = oversample(rap1_pos,rap1_close_neg,rap1_far_neg,
                                            neg_ratio=5, total_samples=10000)
X = np.array([ featurize_sequence(s,word_len=1) for s in os[:,0]])
y = np.ravel(label_binarize(os[:,1],classes=['0','1']))
mlp_os = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 10), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_os.fit(X,y)

# Combined Sampling, 1:2 ratio
cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
                            neg_ratio=2, n_positives=500, proportion_far=0.1)
X = np.array([ featurize_sequence(s,word_len=1) for s in cs[:,0]])
y = np.ravel(label_binarize(cs[:,1],classes=['0','1']))
mlp_cs2 = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(100, 10), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_cs2.fit(X,y)

# Combined Sampling, 1:5 ratio
cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
                            neg_ratio=5, n_positives=500, proportion_far=0.1)
X = np.array([ featurize_sequence(s,word_len=1) for s in os[:,0]])
y = np.ravel(label_binarize(os[:,1],classes=['0','1']))
mlp_cs5 = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=10, learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='lbfgs', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_cs5.fit(X,y)

# Combined Sampling, 1:10 ratio
cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
                            neg_ratio=10, n_positives=500, proportion_far=0.01)
X = np.array([ featurize_sequence(s,word_len=1) for s in os[:,0]])
y = np.ravel(label_binarize(os[:,1],classes=['0','1']))
mlp_cs10 = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(200, 10), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=1000, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
mlp_cs10.fit(X,y)


# ENSEMBLE MODELS AND PREDICT TEST SEQUENCES
print("Ensemble Predictions")
estimators = [mlp_us,mlp_os,mlp_cs2,mlp_cs5,mlp_cs10]
test_X = np.array([ featurize_sequence(s,word_len=1) for s in rap1_test ])
predicted_ys = np.array([ np.zeros(len(test_X),dtype='float') for i in range(len(estimators)) ])
for i,e in enumerate(estimators):
    predicted_ys[i] = e.predict_proba(test_X)[:,1]
final_y = np.mean(predicted_ys[1:],axis=0) #average by col, decided to leave out the undersampling classifier! (first one)

np.set_printoptions(suppress=True) #don't print sci notation

# write individual predictions
individ_out = open('./data/individual_predictions.txt','w')
individ_out.write("Sequence         \t MLP US \t MLP OS \t CS 1:2 \t CS 1:5 \t CS 1:10\n")
for i,s in enumerate(rap1_test):
    individ_out.write("%s\t%8.6f\t%8.6f\t%8.6f\t%8.6f\t%8.6f\n" %
    (s,predicted_ys[0][i],predicted_ys[1][i],predicted_ys[2][i],predicted_ys[3][i],predicted_ys[4][i]) )

# write ensemble predictions
ensemble_out = open('./data/ensemble_predictions.txt','w')
for i,s in enumerate(rap1_test):
    ensemble_out.write("%s\t%6.4f\n" % (s,final_y[i]))











### Model Selection Process Conducted Below ##

# # UNDERSAMPLING
# print("Beginning Undersampling")
# # Make Dataset
# us,us_info = undersample(rap1_pos,rap1_close_neg,rap1_far_neg,
#                                                 neg_ratio=2,proportion_far=0.1)
# X = np.array([ featurize_sequence(s,word_len=1) for s in us[:,0]])
# y = np.ravel(label_binarize(us[:,1],classes=['0','1']))
# skf = StratifiedKFold(n_splits=3)
# us_datasets = list(skf.split(X,y)) #tuples of (train,test) INDICES
# # use list to catch generator contents, otherwise can only iterate once
#
# # Set parameters to test
# hidden_layer_sizes = [(10),(100,10),(100,5),(200,10),(200,5)] # everything except (10) has score 1.0
# k_neighbors = [3,5,10,20,50]
#
# # Test Parameters
# best_undersample = test_parameters(hidden_layer_sizes,k_neighbors,us_datasets,X,y,'undersample')
# print()
#
#
# # OVERSAMPLING
# print("Beginning Oversampling")
# # Make Dataset
# os,os_info = oversample(rap1_pos,rap1_close_neg,rap1_far_neg,
#                                             neg_ratio=5, total_samples=10000)
# X = np.array([ featurize_sequence(s,word_len=1) for s in os[:,0]])
# y = np.ravel(label_binarize(os[:,1],classes=['0','1']))
# skf = StratifiedKFold(n_splits=3)
# os_datasets = list(skf.split(X,y)) #tuples of (train,test) INDICES
# # use list to catch generator contents, otherwise can only iterate once
#
# # Set parameters to test
# hidden_layer_sizes = [(10),(100,10),(100,5),(200,10),(200,5)]
# k_neighbors = [3,5,10,20,50]
#
# # Test Parameters
# best_oversample = test_parameters(hidden_layer_sizes,k_neighbors,os_datasets,X,y,'oversample')
# print()
#
#
#
# # COMBINED SAMPLING
#
# # RATIO 1:2
# print("Beginning Under/Over-Sampling, 1:2")
# # Make Dataset
# cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
#                             neg_ratio=2, n_positives=500, proportion_far=0.1)
# X = np.array([ featurize_sequence(s,word_len=1) for s in cs[:,0]])
# y = np.ravel(label_binarize(cs[:,1],classes=['0','1']))
# skf = StratifiedKFold(n_splits=3)
# cs_datasets = list(skf.split(X,y)) #tuples of (train,test) INDICES
# # use list to catch generator contents, otherwise can only iterate once
#
# # Set parameters to test
# hidden_layer_sizes = [(10),(100,10),(100,5),(200,10),(200,5)]
# k_neighbors = [3,5,10,20]
#
# # Test Parameters
# best_underover2 = test_parameters(hidden_layer_sizes,k_neighbors,cs_datasets,X,y,'under/over 1:2')
# print()
#
#
# # RATIO 1:5
# print("Beginning Under/Over-Sampling, 1:5")
# # Make Dataset
# cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
#                             neg_ratio=5, n_positives=500, proportion_far=0.1)
# X = np.array([ featurize_sequence(s,word_len=1) for s in cs[:,0]])
# y = np.ravel(label_binarize(cs[:,1],classes=['0','1']))
# skf = StratifiedKFold(n_splits=3)
# cs_datasets = list(skf.split(X,y)) #tuples of (train,test) INDICES
# # use list to catch generator contents, otherwise can only iterate once
#
# # Set parameters to test
# hidden_layer_sizes = [(10),(100,10),(100,5),(200,10),(200,5)]
# k_neighbors = [3,5,10,20]
#
# # Test Parameters
# best_underover5 = test_parameters(hidden_layer_sizes,k_neighbors,cs_datasets,X,y,'under/over 1:5')
# print()
#
#
# # RATIO 1:10
# print("Beginning Under/Over-Sampling, 1:10")
# # Make Dataset
# cs,cs_info = underover(rap1_pos,rap1_close_neg,rap1_far_neg,
#                             neg_ratio=10, n_positives=500, proportion_far=0.01)
# X = np.array([ featurize_sequence(s,word_len=1) for s in cs[:,0]])
# y = np.ravel(label_binarize(cs[:,1],classes=['0','1']))
# skf = StratifiedKFold(n_splits=3)
# cs_datasets = list(skf.split(X,y)) #tuples of (train,test) INDICES
# # use list to catch generator contents, otherwise can only iterate once
#
# # Set parameters to test
# hidden_layer_sizes = [(10),(100,10),(100,5),(200,10),(200,5)]
# k_neighbors = [3,5,10,20]
#
# # Test Parameters
# best_underover10 = test_parameters(hidden_layer_sizes,k_neighbors,cs_datasets,X,y,'under/over 1:10')
# print()
#
#
# ## Write Best Parameters
# estimator_out = open('./tf/best_estimator_parameters.txt','w')
# estimators = [best_undersample,best_oversample,best_underover2,best_underover5,best_underover10]
# estimators = sorted(estimators,key=lambda d: 1-np.mean(d['fold_scores']))
# for e in estimators:
#     print(e['dataset'],e['method'],' mean',np.mean(e['fold_scores']),'std',np.std(e['fold_scores']))
#     estimator_out.write(">"+e['dataset']+"\n")
#     estimator_out.write(e['estimator'].__repr__()+"\n")




### GENERAL NOTES ###


# Strategies

# undersample - just do a, hm, 5:1 neg to pos ratio, just make 137*5 negs,
# one set randomly from all the negative seqs, and 4 sets from close

# oversample - keep same 5:1 ratio, just upsample from the 137 up to, say, 60k
# samples total, mostly the close negatives with some far negatives and the
# appropriate amount of upsampled positives to reach 60k

# combination of both - upsample positives to 500, then get 500 far negatives
# and 500*4 close negatives

# try all of the above with a 2:1 ratio as well

# use k-fold stratified cross-validation to learn hyperparameters in all cases






###  BASIC PRELIMINARY TESTS ###

# test first with 8-3-8 encoder

# mlp = MLPClassifier(max_iter=5000,solver='adam',learning_rate_init=0.01,
#                 hidden_layer_sizes=(3))
# data = np.matrix([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],
#         [0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
#         [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
# mlp.fit(data,data)
# print("Training set score: %f" % mlp.score(data, data))
# print("Training set loss: %f" % mlp.loss_)

# okay, that works!

# try with sequences
# mlp = MLPClassifier(max_iter=5000,solver='adam',learning_rate_init=0.01,
#                 hidden_layer_sizes=(6))
# seqs = ['AAA','ATG','GGG','CCG','CCC','ACG','CAG','GAC','TGC','TAA','TTC','AAG','TTG','GGA','CAT']
# inputs = np.array([ featurize_sequence(s,word_len=2) for s in seqs])
# print(seqs)
# print([ interpret_features(i,2) for i in inputs ])
# mlp.fit(inputs,inputs)
# print("Training set score: %f" % mlp.score(inputs,inputs))
# print("Training set loss: %f" % mlp.loss_)
# print(mlp.predict(featurize_sequence('TTT').reshape(1,-1)))
# print(featurize_sequence('TTT').reshape(1,-1))
# print(interpret_features(
#         mlp.predict(featurize_sequence('TTT').reshape(1,-1))[0],2))
# print('TTT')

# okay, well, for the given inputs, it's alright
