import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def perf_evaluate(emb_outputs, true_pairs, false_pairs, num_users):
    user_embs = emb_outputs[:num_users, :]
    pred = sigmoid(user_embs.dot(user_embs.T))
    y_score = []
    y_true = []
    for u, v in true_pairs:
        y_score.append(pred[u, v])
        y_true.append(1)
    for u, v in false_pairs:
        y_score.append(pred[u, v])
        y_true.append(0)
    y_score = np.array(y_score, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int8)

    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    return roc_auc, pr_auc

def load_test_dataset(true_pair_filename, false_pair_filename):
    true_pairs = []
    with open(true_pair_filename, 'r') as f:
        for line in f:
            u, v = line.split()
            u = int(u)
            v = int(v)
            true_pairs.append((u, v))
    false_pairs = []
    with open(false_pair_filename, 'r') as f:
        for line in f:
            u, v = line.split()
            u = int(u)
            v = int(v)
            false_pairs.append((u, v))
    return true_pairs, false_pairs