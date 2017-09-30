#DataSetReader
import numpy as np
from vocabulary import *

__author__  = "Parag Jain"
__email__   = "pajain06@in.ibm.com"

class DataSetReader(object):
  
  def __init__(self, questions, texts, word_to_id, max_q, max_t, name):
    print('Inside dataset reader object')
    print('Question length %d, text length %d'%( len(questions), len(texts) ) )
    self._questions = questions
    self._texts = texts
    self._num_q = len(questions)
    self._num_t = len(texts)
    self._num_examples = self._num_q

    self._max_q = max_q
    self._max_t = max_t
    self.name = name
    
    
    self._text_input = np.zeros([self._num_examples, self._max_t], dtype=np.int32)
    
    if questions == None:
      self._mode = 'test'
    else:
      assert self._num_t == self._num_q, "Number of questions and number of text does not match" 
      self._mode = 'train'
      self._max_q_lenghts = []
      self._max_t_lenghts = []
      self._qdata = []
      self._tdata = []
      self._question_input = np.zeros([self._num_examples, self._max_q], dtype=np.int32)


      for idx in xrange(self._num_examples):
        q = self._questions[idx]
        t = self._texts[idx]
        q_tokens = sentence_to_word_ids(q, word_to_id, max_sequence_length = None, tokenizer='simple', prependGO = True)
        t_tokens = sentence_to_word_ids(t, word_to_id, max_sequence_length = None, tokenizer='simple', prependGO = False)
        self._qdata.append(q_tokens)
        self._tdata.append(t_tokens)
        self._max_q_lenghts.append(len(q_tokens))
        self._max_t_lenghts.append(len(t_tokens))

      #this loop can be merged actually
      for dataid in xrange(self._num_examples):
        q_data = self._qdata[dataid]
        t_data = self._tdata[dataid]
        if len(q_data) > self._max_q:
          q_data = q_data[:self._max_q]
        if len(t_data) > self._max_t:
          t_data = t_data[:self._max_t]

        #print('^^^^^^^ len t_data %d'%len(t_data))
        self._question_input[dataid,:len(q_data)] = np.asarray(q_data)
        self._text_input[dataid,:len(t_data)] = np.asarray(t_data)

      self._word_to_id = word_to_id
      self._id_to_word = dict(zip(word_to_id.values(),word_to_id.keys()))
      self._vocabulary_size = len(self._word_to_id)

      self._epochs_completed = 0
      self._index_in_epoch = 0


  def reset_batch(self, epochs_completed=0):
    self._index_in_epoch = 0
    self._epochs_completed = epochs_completed

  def next_batch(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    end = self._index_in_epoch

    encoder_inputs = self._text_input[start:end]
    encoder_weights = np.ones([batch_size, self._max_t], dtype=np.float32)
    t_seq_lengths = self._max_t_lenghts[start:end]
    for i in range(len(t_seq_lengths)):
      length = t_seq_lengths[i]
      #looping only for length - 1 so that <EOS> is not considered????
      for j in range(length-1,self._max_t):
        encoder_weights[i,j] = 0.0

    batch_size = encoder_inputs.shape[0]
    decoder_inputs = np.zeros([batch_size, self._max_q+1], dtype=np.int32)

    if self._mode == 'train':
      decoder_inputs[:,:-1] = self._question_input[start:end]
      decoder_weights = np.ones([batch_size, self._max_q], dtype=np.float32)
      q_seq_lengths = self._max_q_lenghts[start:end]
      for i in range(len(q_seq_lengths)):
        length = q_seq_lengths[i]
        #looping only for length - 1 so that <EOS> is not considered????
        for j in range(length-1,self._max_q):
          decoder_weights[i,j] = 0.0
    else:
      decoder_inputs[:,0] = np.asarray([self._word_to_id['<GO>']] * batch_size, dtype=np.int32)
      decoder_weights = None

    if self._index_in_epoch >= self._num_examples:
      #print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Shuff')
      self._epochs_completed += 1
      perm = list(np.arange(self._num_examples))
      np.random.shuffle(perm)
      np.take(self._text_input,perm,axis=0,out=self._text_input)
      if self._mode == 'train':
        np.take(self._question_input,perm,axis=0,out=self._question_input)
      self._index_in_epoch = 0

    return encoder_inputs, encoder_weights, decoder_inputs, decoder_weights

  @property
  def vocabulary_size(self):
    return self._vocabulary_size

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def inputs(self):
    return self._inputs

  @property
  def summaries(self):
    return self._summaries

  @property
  def records(self):
    return self._records

  @property
  def summary_sequence_lengths(self):
    return self._summary_seq_lengths

  @property
  def word_to_id(self):
    return self._word_to_id

  @property
  def id_to_word(self):
    return self._id_to_word

  @property
  def max_summary_seq_length(self):
    return self._max_summary_seq_length

  @property
  def max_record_seq_length(self):
    return self._max_record_seq_length

  @property
  def record_initial_size(self):
    return self._record_initial_size

  @property
  def epochs_completed(self):
    return self._epochs_completed














