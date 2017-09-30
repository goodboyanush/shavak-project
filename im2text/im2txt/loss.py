import tensorflow as tf
import numpy as np
import os
import time
import datetime
import txt_classifier.data_helpers as data_helpers
from txt_classifier.text_cnn import TextCNN
# from tensorflow.contrib import learn
import csv
from txt_classifier.VocabPreprocess import *

checkpoint_dir = "./txtclassifier/runs/14/checkpoints/model-"
batch_size = 128

def load_model(sess):
    with tf.variable_scope("classifier") as scope:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)


def getLoss(targets, actual, graph):

    # CHANGE THIS: Load data. Load your own data here
    with tf.variable_scope("classifier") as scope:
        # x_raw, y_test = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
        # y_test = np.argmax(y_test, axis=1)
        x_raw = actual
        y_test = targets
        # Map data into vocabulary
        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        vocab_processor = VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_raw)))

        print("\nEvaluating...\n")

        # Evaluation
        # ==================================================

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]

        # input_x = tf.get_variable(x_test, name='input_x')
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]


        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])