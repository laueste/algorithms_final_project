# implements data processing functions and model selection functions
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #to fix a weird bug that causes an NSException anytime plt.____ is called
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from .utils import hamming_distance
from .io import write_seqs

def screen_by_distance(pos_seqs,neg_seqs,threshold=7):
    """Write a file for the subset of negative sequences that are at most 5bp
    away (by hamming distance) from any positive sequence"""
    distances = np.zeros(len(neg_seqs))
    close_negatives = []
    for i,s in enumerate(neg_seqs):
        if i%10000 == 0:
            print(i)
        hd = 100
        for p in pos_seqs:
            hd = min(hd,hamming_distance(s,p))
        distances[i] = hd
        if hd <= threshold: #threshold chosen just by looking at the histogram
            close_negatives.append(s)
    fig,ax = plt.subplots()
    ax.hist(distances)
    ax.set_xlabel('Minimum Hamming Distance from Any Positive Seq')
    ax.set_ylabel('Negative Seq Counts')
    ax.set_title("Histogram of Hamming Distances for Negative Seqs, with Cutoff")
    plt.axvline(x=threshold,linestyle=":",color='r')
    plt.savefig('%s.png' % "HammingDistHistogram")
    print(np.mean(distances))
    write_seqs("./data/rap1-lieb-close-negatives.txt",close_negatives)

def screen_by_distance_reverse(all,close):
    """oops, forgot to make a file for the specifically far sequences..."""
    far_negatives = []
    for i,s in enumerate(all):
        if i%10000 == 0:
            print(i)
        if s not in close:
            far_negatives.append(s)
    write_seqs("./data/rap1-lieb-far-negatives.txt",far_negatives)

def make_dataset():
    return

# TODO save choices of sequences as INDEXES eventually?
def undersample(pos,close,far,neg_ratio=2,proportion_far=0.1):
    """
    Returns a dataset undersampled to the given ratio (pos:neg is 1:neg_ratio),
    with the given proportion of the negative samples taken from the far negatives

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'undersampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": proportion_far,
        "length": len(pos) + len(pos)*neg_ratio
    }
    total_n_neg = len(pos)*neg_ratio
    far_neg = np.random.choice(far,int(total_n_neg*proportion_far))
    close_neg = np.random.choice(close,total_n_neg-len(far_neg))
    all_neg = np.concatenate((close_neg,far_neg))
    all_pos = np.random.choice(pos,len(pos))
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos),dtype='int')))
    neg_data = np.column_stack((all_neg,np.zeros(total_n_neg,dtype='int')))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info

def oversample(pos,close,far,neg_ratio=5,total_samples=60000):
    """
    Returns a dataset where the positive samples are oversampled up to make the
    given ratio for the given total number of samples (should be greater than
    the number of close positives, which is around 46k)

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'oversampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": ((total_samples/(neg_ratio+1))*neg_ratio - len(close))/total_samples,
        "length": total_samples
    }
    total_n_neg = int((total_samples/(neg_ratio+1))*neg_ratio)
    far_neg = np.random.choice(far,max(0,total_n_neg-len(close)))
    close_neg = np.random.choice(close,total_n_neg-len(far_neg))
    all_neg = np.concatenate((close_neg,far_neg))
    all_pos = np.random.choice(pos,int(total_samples-total_n_neg))
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos),dtype='int')))
    neg_data = np.column_stack((all_neg,np.zeros(total_n_neg,dtype='int')))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info

def underover(pos,close,far,neg_ratio=2,n_positives=500,proportion_far=0.1):
    """
    Returns a dataset with the given number of positives (oversampled as needed)
    and the given proprortion of pos:neg sequences (undersampled as needed),
    with the given proprortion of far negatives within the negative samples

    Input: list of positive seqs, list of close negative seqs, list of far negative seqs
    Output: 2-column matrix of seq,binary_label, dictionary of description
    """
    info = {
        "type": 'under/over-sampling',
        "neg_ratio": neg_ratio,
        "proportion_far_neg": proportion_far,
        "length": n_positives + n_positives*neg_ratio
    }
    total_n_neg = int(n_positives*neg_ratio)
    far_neg = np.random.choice(far,int(total_n_neg*proportion_far))
    close_neg = np.random.choice(close,int(total_n_neg-len(far_neg)))
    all_neg = np.concatenate((close_neg,far_neg))
    all_pos = np.random.choice(pos,n_positives)
    pos_data = np.column_stack((all_pos,np.ones(len(all_pos),dtype='int')))
    neg_data = np.column_stack((all_neg,np.zeros(total_n_neg,dtype='int')))
    dataset = np.concatenate((pos_data,neg_data))
    #N by 2 array of data, 1rst col is sequences, 2nd col is labels
    np.random.shuffle(dataset) #shuffle the order of the rows
    return dataset, info


def test_parameters(hidden_layer_sizes_set,n_neighbors_set,datasets,X,y,dataset_name):
    """
    Essentially a sub-script to verbosely perform a basic gridsearch on KNNs
    versus MLPs, just changing either the hidden layer structure or the number
    of nearest neighbors.

    Input: list of tuples of hidden layer sizes, list of ints for n neighbors,
           list of tuples of (train,test) indices for the given dataset with
           one tuple per fold to be tested, the inputs X and the answers y
           of the dataset to be tested
    Output: dictionary that holds, for the winning combination of factors,
            the method name, the relevant parameter (either hidden layer sizes
            or number of neighbors), list of scores on all the dataset folds,
            and the trained estimator itself
    """
    best = {
        'dataset': dataset_name,
        'method': '',
        'params': [],
        'fold_scores': [0.0],
        'estimator': None
    }
    for params in zip(hidden_layer_sizes_set,n_neighbors_set):
        hls,kn = params
        print("Hidden Layer Sizes:",hls)
        print("K Nearest Neighbors:",kn)
        mlp_adam = MLPClassifier(hidden_layer_sizes=hls,solver='adam',max_iter=1000,learning_rate='adaptive')
        mlp_lbfgs = MLPClassifier(hidden_layer_sizes=hls,solver='lbfgs',max_iter=1000,learning_rate='adaptive')
        knn_uniform = KNeighborsClassifier(n_neighbors=kn,weights='uniform')
        knn_distance = KNeighborsClassifier(n_neighbors=kn,weights='distance')
        print()
        mlp_adam_results = []
        mlp_lbfgs_results = []
        knn_uniform_results = []
        knn_distance_results = []
        for train,test in datasets:
            train_X = X[train]
            train_y = y[train]
            test_X = X[test]
            test_y = y[test]
            mlp_adam.fit(train_X,train_y)
            mlp_lbfgs.fit(train_X,train_y)
            knn_uniform.fit(train_X,train_y)
            knn_distance.fit(train_X,train_y)
            # print("MLP, Adam")
            mlp_adam_p = mlp_adam.predict_proba(test_X) #predict_proba returns [prob(0),prob(1)]
            mlp_adam_y = mlp_adam_p[:,1] #probability of being 1 aka pos hit for rap1
            mlp_adam_score = roc_auc_score(test_y,mlp_adam_y)
            mlp_adam_results.append(mlp_adam_score)
            # print("\tROC-AUC:",mlp_adam_score)
            # print("MLP, LBFGS")
            mlp_lbfgs_p = mlp_lbfgs.predict_proba(test_X) #predict_proba returns [prob(0),prob(1)]
            mlp_lbfgs_y = mlp_lbfgs_p[:,1]
            mlp_lbfgs_score = roc_auc_score(test_y,mlp_lbfgs_y)
            mlp_lbfgs_results.append(mlp_lbfgs_score)
            # print("\tROC-AUC:",mlp_lbfgs_score)
            # print("KNN, uniform")
            knn_unif_p = knn_uniform.predict_proba(test_X) #predict_proba gives [prob(0),prob(1)]
            knn_unif_y = knn_unif_p[:,1]
            knn_unif_score = roc_auc_score(test_y,knn_unif_y)
            knn_uniform_results.append(knn_unif_score)
            # print("\tROC-AUC:",knn_unif_score)
            # print("KNN, distance")
            knn_dist_p = knn_distance.predict_proba(test_X) #predict_proba returns [prob(0),prob(1)]
            knn_dist_y = knn_dist_p[:,1]
            knn_dist_score = roc_auc_score(test_y,knn_dist_y)
            knn_distance_results.append(knn_dist_score)
            # print("\tROC-AUC:",knn_dist_score)
            # print('- - - -')
        # print("ROC-AUC Results for Given Params")
        # print("MLP Adam:",np.mean(mlp_adam_results),np.std(mlp_adam_results),' mean stdev')
        # print("MLP LBFGS:",np.mean(mlp_lbfgs_results),np.std(mlp_lbfgs_results),' mean stdev')
        # print("KNN Uniform:",np.mean(knn_uniform_results),np.std(knn_uniform_results),' mean stdev')
        # print("KNN Distance:",np.mean(knn_distance_results),np.std(knn_distance_results),' mean stdev')
        # print()
        # update 'best'
        for method,scores,estimator in zip(
                ['MLP Adam','MLP LBFGS','KNN Uniform','KNN Distance'],
                [mlp_adam_results,mlp_lbfgs_results,knn_uniform_results,knn_distance_results],
                [mlp_adam,mlp_lbfgs,knn_uniform,knn_distance]):
            if np.mean(best['fold_scores']) < np.mean(scores):
                best['method'] = method
                best['estimator'] = estimator
                best['fold_scores'] = scores
                best['params'] = params[0] if 'MLP' in method else params[1]
    print()
    print("BEST:")
    for k,v in best.items():
        print(k,v)
    print("Mean Score:", np.mean(best['fold_scores']) )
    return best

def roc_curve():
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    return
