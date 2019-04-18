# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib import seq2seq
class Sequence_Label(object):
    """
    Extractive Method
    Sequence_labeling.
    """
    def __init__(self,mode,vocab_size,target_vocab_size,emb_dim,encoder_num_units,encoder_num_layers,decoder_num_units,
                 decoder_num_layers,dropout_emb,dropout_hidden,tgt_sos_id,tgt_eos_id,learning_rate,
                 clip_norm,attention_option,beam_size,optimizer,maximum_iterations):

        # inputs
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
        self.labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
        self.lengths = tf.placeholder(tf.int32, shape=[None], name='lengths')

        self.vocab_size = vocab_size
        self.target_vocab_size = target_vocab_size
        self.masks = tf.sequence_mask(self.lengths,dtype=tf.float32,name='masks')

        # cell
        def cell(num_units):
            cell = rnn.BasicLSTMCell(num_units=num_units)
            if mode == 'train':
                cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_hidden)
            return cell

        # embeddings
        self.embeddings = tf.get_variable("embeddings", shape=[vocab_size, emb_dim],dtype=tf.float32)

        x = tf.nn.embedding_lookup(self.embeddings,self.inputs)
        if mode == 'train':
            x = tf.nn.dropout(x, 1 - dropout_emb)

        fw_cell = rnn.MultiRNNCell([cell(encoder_num_units) for _ in range(encoder_num_layers)])
        bw_cell = rnn.MultiRNNCell([cell(encoder_num_units) for _ in range(encoder_num_layers)])
        (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=x,
            sequence_length=self.lengths,
            dtype=tf.float32
        )
        outputs = tf.concat([fw_outputs, bw_outputs], axis=2)
        self.logits = tf.layers.dense(outputs,units=2,activation=None,use_bias=True)
        self.predicts = tf.argmax(self.logits,dimension=2)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels) * self.masks
        self.loss = tf.reduce_sum(loss) / tf.to_float(tf.shape(self.inputs)[0])
        tf.summary.scalar('loss', self.loss)
        # summaries
        self.merged = tf.summary.merge_all()

        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, dtype=tf.int32)
            self.learning_rate = tf.Variable(learning_rate, trainable=False)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            tvars=tf.trainable_variables()
            grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,tvars),5)
            self.train_op=self.optimizer.apply_gradients(zip(grads,tvars),self.global_step)


    def train_step(self, sess, batch_data):
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
        labels = np.zeros_like(encoder_inputs)
        batch_size=len(encoder_inputs)
        max_seqlen=len(encoder_inputs[0])
        for i in range(batch_size):
            for j in range(max_seqlen):
                if encoder_inputs[i][j]!=0 and encoder_inputs[i][j] in decoder_inputs[i]:
                    labels[i][j]=1
        feed_dict = {
            self.inputs:encoder_inputs,
            self.labels:labels,
            self.lengths:encoder_lengths
        }
        _, loss, summaries, global_step = sess.run([self.train_op, self.loss, self.merged, self.global_step], feed_dict)
        return loss, summaries, global_step

    def eval_step(self, sess, batch_data):
        # encoder_inputs,encoder_lengths = batch_data
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
        decoder_outputs = np.array(decoder_inputs)[:,1:]

        query_id = []
        for item,length in zip(encoder_inputs,encoder_lengths):
            query_id.append(item[:length])

        golden_id = []
        for item,length in zip(decoder_outputs,decoder_lengths):
            golden_id.append(item[:length-1])

        feed_dict = {
            self.inputs:encoder_inputs,
            self.lengths:encoder_lengths
        }

        predicts = sess.run(self.predicts,feed_dict)
        predict_id=[]
        for encoder_input,predict,length in zip(encoder_inputs,predicts,encoder_lengths):
            predict=predict[:length]
            temp=[]
            for i in range(len(predict)):
                if predict[i]==1:
                    temp.append(encoder_input[i])
            predict_id.append(temp)
        return query_id,golden_id,predict_id

if __name__ == '__main__':
    pass
