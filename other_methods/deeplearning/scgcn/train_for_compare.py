import argparse
import os
import pandas as pd
import sys
import time

import numpy
import numpy as np
import pickle as pkl
import tensorflow as tf
from utils import *
from tensorflow.python.saved_model import tag_constants
from models import scGCN
# sys.stdout = open("output_log.txt", "w")

import warnings

warnings.filterwarnings("ignore")
# ' del_all_flags(FLAGS)
tf.compat.v1.disable_eager_execution()
# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('proj', 'xxx', 'Project name')
flags.DEFINE_string('dataset', 'input', 'data dir')
flags.DEFINE_string('output', 'result', 'predicted results')
flags.DEFINE_bool('graph', False, 'if graph generated from R? ... select the optional graph.')
flags.DEFINE_string('model', 'scGCN', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
# flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, labels_binary_train, labels_binary_val, labels_binary_test, \
train_mask, pred_mask, val_mask, test_mask, new_label, true_label, \
index_guide, types = load_data(
    FLAGS.dataset, rgraph=FLAGS.graph)

support = [preprocess_adj(adj)]
num_supports = 1
model_func = scGCN


# Define placeholders
placeholders = {
    'support':
        [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features':
        tf.sparse_placeholder(tf.float32,
                              shape=tf.constant(features[2], dtype=tf.int64)),
    'labels':
        tf.placeholder(tf.float32, shape=(None, labels_binary_train.shape[1])),
    'labels_mask':
        tf.placeholder(tf.int32),
    'dropout':
        tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero':
        tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask,
                                        placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())

train_accuracy = []
train_loss = []
val_accuracy = []
val_loss = []
test_accuracy = []
test_loss = []

# Train model

# configurate checkpoint directory to save intermediate model training weights
saver = tf.train.Saver()
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')

for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, labels_binary_train,
                                    train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy],
                    feed_dict=feed_dict)
    train_accuracy.append(outs[2])
    train_loss.append(outs[1])
    # Validation
    cost, acc, duration = evaluate(features, support, labels_binary_val,
                                   val_mask, placeholders)
    val_loss.append(cost)
    val_accuracy.append(acc)
    test_cost, test_acc, test_duration = evaluate(features, support,
                                                  labels_binary_test,
                                                  test_mask, placeholders)
    test_accuracy.append(test_acc)
    test_loss.append(test_cost)
    saver.save(sess=sess, save_path=save_path)
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
          "{:.5f}".format(outs[1]), "train_acc=", "{:.5f}".format(outs[2]),
          "val_loss=", "{:.5f}".format(cost), "val_acc=", "{:.5f}".format(acc),
          "time=", "{:.5f}".format(time.time() - t))
    if epoch > FLAGS.early_stopping and val_loss[-1] > np.mean(
            val_loss[-(FLAGS.early_stopping + 1):-1]):
        print("Early stopping...")
        break

print("Finished Training....")

all_mask = np.array([True] * len(train_mask))
labels_binary_all = new_label

feed_dict_all = construct_feed_dict(features, support, labels_binary_all,
                                    all_mask, placeholders)
feed_dict_all.update({placeholders['dropout']: FLAGS.dropout})

# Embeddings , activation_output就是我们要的embedding
activation_output = sess.run(model.activations, feed_dict=feed_dict_all)[1]

predict_output = sess.run(model.outputs, feed_dict=feed_dict_all)
prob = sess.run(tf.nn.softmax(predict_output))

preds = sess.run(tf.argmax(prob, 1))
trues = sess.run(tf.argmax(labels_binary_all, 1))


'''
    保存结果
'''
from sklearn.metrics import silhouette_score, f1_score, adjusted_rand_score, accuracy_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
#这里要把pred和true做一个inverse，转到之前的类型上去
enc = LabelEncoder()
enc.fit(types)
preds = enc.inverse_transform(preds)
trues = enc.inverse_transform(trues)
acc = accuracy_score(trues[pred_mask], preds[pred_mask])
macrof1 = f1_score(trues[pred_mask], preds[pred_mask], average='macro')
sil = silhouette_score(activation_output[pred_mask], preds[pred_mask])
ari = adjusted_rand_score(trues[pred_mask], preds[pred_mask])
print("pred's shape is {:}".format(preds[pred_mask].shape))
print("acc is {:.3f}".format(acc))
print("f1 score is {:.3f}".format(macrof1))
print("silhoutte score is {:.3f}".format(sil))
print("ARI is {:.3f}".format(ari))

scgcn_res = pd.read_csv('scgcn_res.csv', index_col=0)
scgcn_res[FLAGS.proj]['acc'] = acc
scgcn_res[FLAGS.proj]['f1'] = macrof1
scgcn_res[FLAGS.proj]['sil'] = sil
scgcn_res[FLAGS.proj]['ari'] = ari
scgcn_res.to_csv('scgcn_res.csv')


query_trues = pd.DataFrame(data=trues[pred_mask], columns=['type'])
query_preds = pd.DataFrame(data=preds[pred_mask], columns=['type'])
query_prob = pd.DataFrame(data=prob[pred_mask].max(axis=1), columns=['prob'])
all_preds = pd.DataFrame(data=preds, columns=['cell_type'])

# embeddings_2d = pd.DataFrame(data=data_2d, columns=['x', 'y'])
# query_embeddings_2d = pd.DataFrame(data=data_2d[pred_mask], columns=['x', 'y'])

query_trues.to_csv(os.path.join(save_path, 'query_labels.csv'), index=False)
query_preds.to_csv(os.path.join(save_path, 'query_preds.csv'), index=False)
query_prob.to_csv(os.path.join(save_path, 'query_prob.csv'), index=False)
all_preds.to_csv(os.path.join(save_path, 'all_preds.csv'), index=False)

np.save('all_embeddings.npy', activation_output)
np.save('query_embeddings.npy', activation_output[pred_mask])

print("finish")

