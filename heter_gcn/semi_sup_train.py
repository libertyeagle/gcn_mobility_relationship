import tensorflow as tf
import numpy as np
import random

from gcn.heter_utils import *
from gcn.models import SemiSupRGCN
from gcn.evaluate import perf_evaluate, load_test_dataset

seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

num_supports = 4
features_dim = 512
# change num_users and num_pois accroding to the dataset
num_users = 7844
num_pois = 18382

flags = tf.app.flags 
FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 5000, 'Number of epochs to train.') 
flags.DEFINE_integer('hidden1', 256, 'Number of units in hidden layer 1.')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('emb_size', 128, 'Number of units in output layer.')
flags.DEFINE_integer("num_walks", 100, "Number of random walks per node.")
flags.DEFINE_integer("walk_len", 5, "Random walk length.")
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer("num_pos_samples", 80, "Number of postive samples for each node.")
flags.DEFINE_integer("neg_sample_size", 20, "Size of negative sampling each epoch.")
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('heter_weights', 0.1, 'Weight for heterogeneous postive sampling.')
flags.DEFINE_float('neg_sample_weights', 1., 'Weight for negative sampling.')
flags.DEFINE_float('semi_sup_weights', 2., 'Weight for partial social graph neighbors.')


placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=(None, features_dim)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'homo_samples': tf.placeholder(tf.int32, shape=(None, )),
    'heter_samples': tf.placeholder(tf.int32, shape=(None, )),
    'semi_sup_samples': tf.placeholder(tf.int32, shape=(None, )),
    'neg_samples': tf.placeholder(tf.int32, shape=(None, FLAGS.neg_sample_size))
}

# load dataset 
graph = load_graph("dataset/austin/austin_heter_graph.npz")
partial_social_graph = load_graph("dataset/austin/austin_partial_social_graph.npz")
node_features = load_features("dataset/austin/austin_user_embeddings.npy",
                              "dataset/austin/austin_poi_embeddings.npy")
true_pairs_val, false_pairs_val = load_test_dataset("dataset/austin/social_relations/semi_supervised/true_pairs_val.txt",
                          "dataset/austin/social_relations/semi_supervised/false_pairs_val.txt")
true_pairs_test, false_pairs_test = load_test_dataset("dataset/austin/social_relations/semi_supervised/true_pairs_test.txt",
                          "dataset/austin/social_relations/semi_supervised/false_pairs_test.txt")


supports = construct_semi_supervised_supports(graph, partial_social_graph, num_users, num_pois)

homo_samples, heter_samples = sample_context(
                                    graph,
                                    num_users,
                                    num_pois,
                                    FLAGS.walk_len,
                                    FLAGS.num_walks,
                                    FLAGS.num_pos_samples
                              )
semi_sup_samples = sample_semi_supervised_context(partial_social_graph, FLAGS.num_pos_samples)

model = SemiSupRGCN(placeholders, FLAGS.hidden1, FLAGS.emb_size, FLAGS.neg_sample_size, FLAGS.learning_rate, num_users)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(FLAGS.epochs):
    neg_samples_1 = np.random.randint(0, num_users,
                 (num_users + num_pois, FLAGS.neg_sample_size // 2), dtype=np.int32)
    neg_samples_2 = np.random.randint(num_users, num_users + num_pois, 
                 (num_users + num_pois, FLAGS.neg_sample_size // 2), dtype=np.int32)
    neg_samples = np.concatenate((neg_samples_1, neg_samples_2), axis=1)
    neg_samples = np.vstack((neg_samples, 
            (num_users + num_pois) * np.ones((1, FLAGS.neg_sample_size), dtype=np.int32)))
    feed_dict = {}
    feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
    feed_dict.update({placeholders['features']: node_features})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['homo_samples']: homo_samples[:, epoch % FLAGS.num_pos_samples]})
    feed_dict.update({placeholders['heter_samples']: heter_samples[:, epoch % FLAGS.num_pos_samples]})
    feed_dict.update({placeholders['neg_samples']: neg_samples})
    feed_dict.update({placeholders['semi_sup_samples']: semi_sup_samples[:, epoch % FLAGS.num_pos_samples]})

    _, cost = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
    print("epoch: {:d}, train_loss={:5f}".format(epoch, cost))

feed_dict = {}
feed_dict.update({placeholders['support'][i]: supports[i] for i in range(len(supports))})
feed_dict.update({placeholders['features']: node_features})
final_embeddings = sess.run(model.outputs, feed_dict=feed_dict)
np.save("dataset/austin/gcn_embeddings/austin_embeddings_semi_sup.npy", final_embeddings)
print("embeddings saved.")

roc_auc, pr_auc = perf_evaluate(final_embeddings, true_pairs_val, false_pairs_val, num_users)
print("evaluation on validation set:")
print("ROCAUC={:.5f}".format(roc_auc))
print("PRAUC={:.5f}".format(pr_auc))

roc_auc, pr_auc = perf_evaluate(final_embeddings, true_pairs_test, false_pairs_test, num_users)
print("evaluation on test set:")
print("ROCAUC={:.5f}".format(roc_auc))
print("PRAUC={:.5f}".format(pr_auc))