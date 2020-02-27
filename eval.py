#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import os
from text_cnn import TextCNN
from tensorflow.contrib import learn
import sys
import time
sys.path.append('../..')
from lib.data_helpers import embed_dataset, batch_iter
from lib.evaluation import decode_evaluation, evaluate

results_folder = 'results'
data_folder = '../../data'
test_corrects_file_name = os.path.join(data_folder, 'corrections_test.txt')
test_sentences_file_name = os.path.join(data_folder, 'sentence_test.txt')


# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", os.path.join(data_folder, "glove_test.txt"), "Test data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# correction_targets = ['a', 'an', 'the']
correction_targets_glove_ids = [6, 39, 2]


# Load data
window_size = 4
print("Loading data...")
x_test, y_test = embed_dataset(FLAGS.data_file, correction_targets_glove_ids, window_size)

# Build vocabulary
glove_file_name = os.path.join(data_folder, 'glove_vectors.txt')

with open(glove_file_name) as json_data:
    vocabulary = json.load(json_data)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        proba = graph.get_operation_by_name("output/proba").outputs[0]


        # Generate batches for one epoch
        batches = batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_confidences = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            batch_proba = sess.run(proba, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            batch_proba = [max(x) for x in batch_proba]
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_confidences = np.concatenate([all_confidences, batch_proba])

labels_with_probabilities = list(zip(all_predictions, all_confidences))

with open(FLAGS.data_file) as json_data:
    test_data = json.load(json_data)

evaluation_results = decode_evaluation(test_data, labels_with_probabilities)

# save results file
timestamp = str(int(time.time()))
results_file_name = os.path.join(results_folder, 'evaluate_test_{}.txt'.format(timestamp))

with open(results_file_name, 'w') as f:
    json.dump(evaluation_results, f)

evaluate(test_sentences_file_name, test_corrects_file_name, results_file_name)
