#train script
import argparse
import json
from DataSetReader import *
from vocabulary import *
from S2SAttentionModel import *
from PretrainedEmbeddings import *
import sys
import numpy as np


# def train(args, train_data, valid_data, id_to_word=None, vocabulary_size=None):
def train(args, train_data, id_to_word=None, vocabulary_size=None, word_to_id = None):
  step = 0

  # save the args and the vocabulary

  if not os.path.exists(args.save_dir): 
    os.makedirs(args.save_dir)  

  with open(os.path.join(args.save_dir,'args.json'),'w') as f:
    f.write(json.dumps(vars(args),indent=1))

  with open(os.path.join(args.save_dir,'word_to_id.json'),'w') as f:
    f.write(json.dumps(word_to_id,indent=1))

  with open(os.path.join(args.save_dir,'id_to_word.json'),'w') as f:
    f.write(json.dumps(id_to_word,indent=1))
  

  with tf.Graph().as_default():

    if bool(args.use_pretrained_embedding_matrix):
        print('----------------------------loading word2vec embeddings')
        embeddings = PretrainedEmbeddings(args.pretrained_embedding_filename)
        initial_embedding_matrix = embeddings.load_embedding_matrix(train_data.word_to_id)
        args.embedding_size = embeddings.embedding_size

    model = S2SAttentionModel(train_data.vocabulary_size,
                   rnn_size=args.rnn_size,
                   num_layers=args.num_layers,
                   max_gradient_norm=args.max_gradient_norm,
                   max_summary_seq_length=args.max_q,
                   model=args.model,
                   batch_size=args.batch_size,
                   embedding_size=args.embedding_size,
                   embedding_trainable = bool(args.train_embedding_matrix),
                   learning_rate=args.learning_rate,
                   optimizer=args.optimizer,
                   forward_only=False)  

    # generate placeholders for the inputs
    enc_inputs  = tf.placeholder(tf.int32, shape=(None, args.max_t), name="enc_inputs")
    # dec_inputs  = tf.placeholder(tf.int32, shape=(None, args.max_q+1), name="dec_inputs")
    # dec_weights  = tf.placeholder(tf.float32, shape=(None, args.max_q), name="dec_weights")

    # build a graph that computes predictions from the inference model
    logits_op = model.inference(enc_inputs, dec_inputs)
    #predictions_op = tf.nn.softmax(logits_op)
    predictions_op = model.prediction(logits_op)

    # add to the graph the ops for loss calculation
    loss_op = model.loss(logits_op, dec_inputs, dec_weights)

    # add to the graph the ops that calculate and apply gradients
    train_op = model.training(loss_op)

    # add the op to calculate perplexity
    perplexity_op = model.evaluation(logits_op)

    # directory to dump the intermediate models
    checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')  

    # location to dump the predictions for validation
    if not os.path.exists(os.path.join(args.save_dir, 'predictions')):
      os.makedirs(os.path.join(args.save_dir, 'predictions'))

    #summary directry
    summary_dir = os.path.abspath(os.path.join(args.save_dir, 'predicted_summaries'))
    if not os.path.exists(summary_dir):
      os.makedirs(summary_dir)


    saver = tf.train.Saver(tf.all_variables(),max_to_keep=max(5,args.early_stopping_batch_window+2))

    def write_batch_summary_to_file(predictions, summary_file, data_set, batch_counter, batch_size):
      output_summaries = []
      start_index = batch_counter * batch_size
      for pid in xrange(len(predictions)):
        token_list_id = start_index + pid
        words = []
        for l in xrange(args.max_q):
          tokenid = np.argmax(predictions[pid, l])
          words.append(id_to_word[tokenid])
        output_summaries.append(' '.join(words) + '\n')
      
      #print(output_summaries)
      summary_file.writelines(output_summaries)


    def evaluate_model(sess,
      data_set,
      enc_inputs,
      dec_inputs,
      dec_weights,
      perplexity_op,
      loss_op,
      batch_size,
      id_to_word,
      vocabulary_size,
      train_epoch_completed,
      print_sentences=False):
      """ evaluate on the entire data to track some metrics
      """
      batch_loss = 0.0
      out_file_name = data_set.name + '_' + str(train_epoch_completed) + '.outsentences'
      out_file_name = os.path.join(summary_dir, out_file_name)
      out_file = open(out_file_name, 'wa')
      initial_epoch = data_set.epochs_completed

      print('*************** Evaluating dataset:: ' + data_set.name + ' summary file name ' + out_file_name)

      #if print_sentences:
      #  predictions = np.array([]).reshape(0,args.max_q, data_set.vocabulary_size + args.max_t)

      if data_set.name == 'train':
        feed_prev_bool = False
      else:
        feed_prev_bool = True 

      batch_counter=0

      while True:
        print('Inside while of evaluate') 
        batch_enc_inputs,batch_enc_weights,batch_dec_inputs,batch_dec_weights = data_set.next_batch(batch_size)

        feed_dict = {enc_inputs:batch_enc_inputs,
        dec_inputs:batch_dec_inputs,
        dec_weights:batch_dec_weights}

        batch_loss += sess.run(loss_op,feed_dict=feed_dict)*np.shape(batch_dec_weights)[0]
        print(batch_counter)
        print('initial_epoch %d epochs completed %d batch counter %d'%(initial_epoch, data_set.epochs_completed, batch_counter))
        if print_sentences:
          batch_predictions = sess.run(predictions_op, feed_dict=feed_dict)
          print('pp')
          print(batch_predictions.shape)
          write_batch_summary_to_file(batch_predictions, out_file, data_set, batch_counter, batch_size)
          #predictions = np.concatenate([predictions, batch_predictions],axis=0)

        batch_counter += 1
        #printing only one batch for train
        if data_set.name == 'train':
          print_sentences = False

        if data_set.epochs_completed > initial_epoch:
            print 'breaking'
            break
      
      loss = batch_loss / data_set.num_examples
      perplexity = math.exp(float(batch_loss)) if batch_loss < 300 else float("inf")
      out_file.close()
      return perplexity,loss

    # evaluation function
    def evaluate_model_old(sess,
      data_set,
      enc_inputs,
      dec_inputs,
      dec_weights,
      perplexity_op,
      loss_op,
      batch_size,
      id_to_word,
      print_sentences=False):
      """ evaluate on the entire data to track some metrics
      """
      batch_loss = 0.0
      initial_epoch = data_set.epochs_completed
      if print_sentences:
        predictions = np.array([]).reshape(0,args.max_q, data_set.vocabulary_size)
      while True:
        batch_enc_inputs,batch_enc_weights,batch_dec_inputs,batch_dec_weights = data_set.next_batch(batch_size)
        feed_dict = {enc_inputs:batch_enc_inputs,
        dec_inputs:batch_dec_inputs,
        dec_weights:batch_dec_weights}
        batch_loss += sess.run(loss_op,feed_dict=feed_dict)*np.shape(batch_dec_weights)[0]
        if print_sentences:
          predictions = np.concatenate([predictions,sess.run(predictions_op, feed_dict=feed_dict)],axis=0)
        if data_set.epochs_completed > initial_epoch:
            break
      loss = batch_loss / data_set.num_examples
      perplexity = math.exp(float(batch_loss)) if batch_loss < 300 else float("inf")
      
      if print_sentences:
        output_summaries = []
        for id in xrange(len(predictions)):
          words = []
          for l in xrange(args.max_q):
            tokenid = np.argmax(predictions[id,l])
            words.append(id_to_word[tokenid])
          output_summaries.append(' '.join(words))

        print(output_summaries)
      # print 'perplexity:' + str(perplexity)
      # print 'loss:' + str(loss)
      return perplexity,loss


    session_conf = tf.ConfigProto(allow_soft_placement=True)
    session_conf.gpu_options.allow_growth=True
    with tf.Session(config=session_conf) as sess:
      # run the op to initialize the variables
      init = tf.initialize_all_variables()
      sess.run(init)    

      # use the pre-trained word2vec embeddings
      if bool(args.use_pretrained_embedding_matrix):
        sess.run(model.embedding_matrix.assign(initial_embedding_matrix))

      print('Trainable Variables')
      print '\n'.join([v.name for v in tf.trainable_variables()])

      # instantiate a SummaryWriter to output summaries and the graph
      #train_summary_writer = tf.train.SummaryWriter(train_summary_dir, graph_def=sess.graph_def)
      #valid_summary_writer = tf.train.SummaryWriter(valid_summary_dir, graph_def=sess.graph_def)

      step = 0
      previous_epoch = 0
      performance = {}
      performance['valid_loss'] = []
      performance['train_loss'] = []
      performance['test_loss']= []
      performance['valid_perplexity'] = []
      performance['train_perplexity'] = []
      performance['test_perplexity'] = []     
      while train_data.epochs_completed < args.max_epochs:

        step += 1
       
        start_time = time.time() 
        batch_enc_inputs,batch_enc_weights,batch_dec_inputs,batch_dec_weights = train_data.next_batch(args.batch_size)

        feed_dict = {enc_inputs:batch_enc_inputs,
        dec_inputs:batch_dec_inputs,
        dec_weights:batch_dec_weights}
        
        _, loss_val, perplexity_val = sess.run([train_op, loss_op, perplexity_op],feed_dict=feed_dict)
        duration = time.time() - start_time

        # print an overview 
        if True:
          print('epoch %d batch %d: loss = %.3f perplexity = %.2f (%.3f secs)' % (train_data.epochs_completed+1,
            step,
            loss_val,
            perplexity_val,
            duration))

        if train_data.epochs_completed > previous_epoch:
          previous_epoch = train_data.epochs_completed
          batch_in_epoch = 0
          # evaluate the model on the entire data    
          if train_data.epochs_completed % args.evaluate_every_epochs == 0 or train_data.epochs_completed == args.max_epochs-1:
            perplexity,loss = evaluate_model(sess,train_data,enc_inputs,dec_inputs,dec_weights,perplexity_op,loss_op,args.batch_size,
              id_to_word, vocabulary_size, train_data.epochs_completed, print_sentences=True)
            #print('-----------------------------------train not calculating')
            print('----------------------------------train perplexity : %0.03f loss : %0.03f' %(perplexity,loss))
            performance['train_loss'].append(loss)
            performance['train_perplexity'].append(perplexity)
            train_data.reset_batch(epochs_completed=previous_epoch)

            perplexity,loss = evaluate_model(sess,valid_data,enc_inputs,dec_inputs,dec_weights,perplexity_op,loss_op,args.batch_size,
              id_to_word, vocabulary_size, train_data.epochs_completed, print_sentences=True)
            #print('-----------------------------------valid not calculating')
            print('----------------------------------valid perplexity : %0.03f loss : %0.03f' %(perplexity,loss))
            performance['valid_loss'].append(loss)
            performance['valid_perplexity'].append(perplexity)
            valid_data.reset_batch()

            perplexity,loss = evaluate_model(sess,test_data,enc_inputs,dec_inputs,dec_weights,perplexity_op,loss_op,args.batch_size,
              id_to_word, vocabulary_size, train_data.epochs_completed, print_sentences=True)
            #print('-----------------------------------valid not calculating')
            print('----------------------------------test perplexity : %0.03f loss : %0.03f' %(perplexity,loss))
            performance['test_loss'].append(loss)
            performance['test_perplexity'].append(perplexity)
            test_data.reset_batch()
        
          if train_data.epochs_completed % args.save_every_epochs == 0 or train_data.epochs_completed == args.max_epochs-1: 
            path = saver.save(sess, checkpoint_prefix, global_step=train_data.epochs_completed)
            print("Saved model checkpoint to {}".format(path))
         
        print(train_data.epochs_completed)
        print('ec')
        sys.stdout.flush()


'''
  while train_data.epochs_completed < args.max_epochs:
    step += 1
    print("Running step %d and epoch %d"%(step,train_data.epochs_completed))
    encoder_inputs, encoder_weights, decoder_inputs, decoder_weights = train_data.next_batch(args.batch_size)
    x = word_ids_to_sentence(encoder_inputs[0], id_to_word)
    print(x.encode('utf-8'))
    print('Encoder Inputs***************************************')
    print(encoder_inputs)
    print('Encode weights***************************************')
    print(encoder_weights)
    print('Decoder Inputs****************************************')
    print(decoder_inputs)
    print('Decoder weights')
    print(decoder_weights)
'''



def load_summaries(file_name):
  print('Readind file ' + file_name)
  # with open(file_name) as file:
  #   lines = file.readlines()
  #   ques = []
  #   ans = []
  #   for line in lines:
  #     data = json.loads(line)
  #     ques.append(data['seq_text'])
  #     ans.append(data['dii_text'])
  data = np.load(file_name).item()
    # return ques, ans
  return data

#

def load_data(train_file, vocabulary_min_count):
  # train_q, train_a = load_qa(train_file)
  # test_q, test_a = load_qa(test_file)
  # valid_q, valid_a = load_qa(valid_file)
  data = load_summaries(train_file)
  print('Loaded data...building vocab')
  # vocabulary_sentences = list(data)
  vocabulary_sentences = []
  for sent in data.values():
      vocabulary_sentences.extend(sent)
# vocabulary_sentences.extend(test_a)
  # vocabulary_sentences.extend(valid_a)
  #Added question words also?
  # vocabulary_sentences.extend(train_q)
  # vocabulary_sentences.extend(test_q)
  # vocabulary_sentences.extend(valid_q)
  word_to_id, id_to_word = build_vocabulary(vocabulary_sentences, min_count = vocabulary_min_count, tokenizer='simple')
  with open('word_to_id.txt','w+') as wf:
    wf.write(str(word_to_id))

  vocabulary_size = len(word_to_id)
  # train_data = DataSetReader(train_q, train_a, word_to_id, max_q, max_t, 'train')
  # test_data = DataSetReader(test_q, test_a, word_to_id, max_q, max_t,'test')
  # valid_data = DataSetReader(valid_q, valid_a, word_to_id, max_q, max_t, 'valid')
  
  return data, id_to_word, word_to_id


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--encoder_embedding_size', type=int, default=20, help='size of the embeddings for records (<65)')
  parser.add_argument('--decoder_embedding_size', type=int, default=64, help='size of the embeddings for output words') # 100,200 32,64,128
  parser.add_argument('--bidirectional_encoder', type=int, default=0, help='use bidirectional rnn for encoder or not (1 or 0)')
  parser.add_argument('--rnn_size', type=int, default=64, help='size of RNN hidden state') # 100
  parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN (default: 1)') 
  parser.add_argument('--model', type=str, default='gru', help='rnn, gru, basic_lstm, or lstm (default: gru)')
  parser.add_argument('--rnn_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability when using multiple RNNs')
  parser.add_argument('--output_dropout_keep_prob', type=float, default=1.0, help='dropout keep probability for the output (i.e. before the softmax layer)')
  parser.add_argument('--train_embedding_matrix', type=int, default=1, help='if 0 does not train the embedding matrix and keeps it fixed')
  parser.add_argument('--use_pretrained_embedding_matrix', type=int, default=0, help='if 1 use the pre-trained word2vec for initializing the embedding matrix')
  parser.add_argument('--pretrained_embedding_filename', type=str, default='../resources/GoogleNews-vectors-negative300.bin', help='full path to the .bin file containing the pre-trained word vectors') 
  parser.add_argument('--embedding_size', type=int, default=100, help='size of embedding if pretrained embeddings are not used')


  # Training parameters
  parser.add_argument('--batch_size', type=int, default=64, help='batch size') # 16, 32, 64
  parser.add_argument('--max_epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('--optimizer', type=str, default='adam', help='gradient_descent, adam') #rmsprop, 0.01, 0.1
  parser.add_argument('--max_gradient_norm', type=float, default=1000.0, help='maximum gradient norm for clipping. currently unused.')
  parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
  parser.add_argument('--save_dir', type=str, default='exp_1', help='directory to save checkpointed models')
  parser.add_argument('--print_every', type=int, default=1, help='print some info after this many batches')
  parser.add_argument('--summary_every', type=int, default=10, help='dump summaries for tensorboard after this many batches')
  parser.add_argument('--save_every_epochs', type=int, default=1, help='save the model after this many epochs')
  parser.add_argument('--evaluate_every_epochs', type=int, default=1, help='evaluate the model on the entire data after this many epochs')

  parser.add_argument('--early_stopping', type=int, default=1, help='if 1 enables early stopping based on loss on the validation split')
  parser.add_argument('--early_stopping_batch_window', type=int, default=5, help='early stop if the validation loss is greater than the loss from these many previous steps')
  parser.add_argument('--early_stopping_threshold', type=float, default=0.05, help='threshold for early stopping')
  
  #Task specific
  parser.add_argument('--trainFilename', type=str, default=None, help='name of the train file')
  parser.add_argument('--validFilename', type=str, default=None, help='name of the valid file')
  parser.add_argument('--testFilename', type=str, default=None, help='name of the test file')
  parser.add_argument('--vocab_min_count', type=int, default=1, help='keep words whose count is >= vocab_min_count')
  parser.add_argument('--max_q', type=int, default=50, help='maximum question length')
  parser.add_argument('--max_t', type=int, default=50, help='maximum answer length')

  args = parser.parse_args()

  # train_data, test_data, valid_data, id_to_word = load_data(args.trainFilename, args.testFilename, args.validFilename, args.max_q, args.max_t, args.vocab_min_count)
  data, id_to_word, word_to_id = load_data(args.trainFilename, args.vocab_min_count)
  print(id_to_word)
  vocabulary_size = len(id_to_word)
  # train(args, train_data, valid_data, id_to_word=id_to_word, vocabulary_size=vocabulary_size)
  train(args, data, id_to_word=id_to_word, vocabulary_size=vocabulary_size, word_to_id=word_to_id)