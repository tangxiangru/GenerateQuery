# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers import core as layers_core

class Seq2seq_Attention(object):
    """
    Generative Method
    Seq2seq + Attention
    """

    def __init__(self,mode,vocab_size,target_vocab_size,emb_dim,encoder_num_units,encoder_num_layers,
                 decoder_num_units, decoder_num_layers,dropout_emb,dropout_hidden,tgt_sos_id,tgt_eos_id,
                 learning_rate,clip_norm,attention_option,beam_size,optimizer,maximum_iterations):

        assert mode in ["train", "infer"], "invalid mode!"
        assert encoder_num_units==decoder_num_units, "encoder num_units **must** match decoder num_units"
        self.target_vocab_size=target_vocab_size

        # inputs
        self.encoder_inputs = tf.placeholder(tf.int32,shape=[None,None],name='encoder_inputs')
        self.decoder_inputs = tf.placeholder(tf.int32,shape=[None,None],name='decoder_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32,shape=[None,None],name='decoder_outputs')
        self.encoder_lengths = tf.placeholder(tf.int32,shape=[None],name='encoder_lengths')
        self.decoder_lengths = tf.placeholder(tf.int32,shape=[None],name='decoder_lengths')

        # cell
        def cell(num_units):
            cell = rnn.BasicLSTMCell(num_units=num_units)
            if mode == 'train':
                cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=1 - dropout_hidden)
            return cell

        # embeddings
        self.embeddings = tf.get_variable('embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32)

        # Encoder
        with tf.variable_scope('encoder'):
            # embeddings
            encoder_inputs_emb = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            if mode == 'train':
                encoder_inputs_emb = tf.nn.dropout(encoder_inputs_emb, 1 - dropout_emb)

            # encoder_rnn_cell
            fw_encoder_cell = cell(encoder_num_units)
            bw_encoder_cell = cell(encoder_num_units)

            # bi_lstm encoder
            (encoder_outputs_fw, encoder_outputs_bw),(encoder_state_fw,encoder_state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_encoder_cell,
                cell_bw=bw_encoder_cell,
                inputs=encoder_inputs_emb,
                sequence_length=self.encoder_lengths,
                dtype=tf.float32
            )
            encoder_outputs = tf.concat([encoder_outputs_fw,encoder_outputs_bw], 2)

            # A linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder.
            # This is needed because the encoder is bidirectional but the decoder is not.
            encoder_states_c = tf.layers.dense(inputs=tf.concat([encoder_state_fw.c,encoder_state_bw.c],axis=-1),
                                             units=encoder_num_units,activation=None,use_bias=False)
            encoder_states_h = tf.layers.dense(inputs=tf.concat([encoder_state_fw.h, encoder_state_bw.h], axis=-1),
                                             units=encoder_num_units, activation=None, use_bias=False)
            encoder_states = rnn.LSTMStateTuple(encoder_states_c,encoder_states_h)

            encoder_lengths = self.encoder_lengths

        # Decoder
        with tf.variable_scope('decoder'):
            decoder_inputs_emb = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            if mode == 'train':
                decoder_inputs_emb = tf.nn.dropout(decoder_inputs_emb, 1 - dropout_emb)
            # decoder_rnn_cell
            decoder_cell = cell(decoder_num_units)
            with tf.variable_scope('attention_mechanism'):
                if attention_option == "luong":
                    attention_mechanism = seq2seq.LuongAttention(
                        num_units=decoder_num_units, memory=encoder_outputs, memory_sequence_length=encoder_lengths)
                    cell_input_fn = lambda inputs, attention: inputs
                    output_attention = True
                elif attention_option == "scaled_luong":
                    attention_mechanism = seq2seq.LuongAttention(
                        num_units=decoder_num_units, memory=encoder_outputs, memory_sequence_length=encoder_lengths, scale=True)
                    cell_input_fn = lambda inputs, attention: inputs
                    output_attention = True
                elif attention_option == "bahdanau":
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=decoder_num_units, memory=encoder_outputs, memory_sequence_length=encoder_lengths)
                    cell_input_fn = lambda inputs, attention: tf.concat([inputs,attention],-1)
                    output_attention = False
                elif attention_option == "normed_bahdanau":
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=decoder_num_units, memory=encoder_outputs, memory_sequence_length=encoder_lengths, normalize=True)
                    cell_input_fn = lambda inputs, attention: tf.concat([inputs, attention],-1)
                    output_attention = False
                else:
                    raise ValueError("Unknown attention option %s" % attention_option)
            # # Only generate alignment in greedy INFER mode.
            # alignment_history = (mode == 'infer' and beam_size==0)
            alignment_history = False
            decoder_cell = seq2seq.AttentionWrapper(
                cell = decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=decoder_num_units,
                alignment_history=alignment_history,
                cell_input_fn=cell_input_fn,
                output_attention=output_attention
            )

            batch_size = tf.shape(self.encoder_inputs)[0]
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_states)

            projection_layer = layers_core.Dense(units=target_vocab_size, use_bias=False)

            # train/infer
            if mode=='train':
                # helper
                helper = seq2seq.TrainingHelper(
                    inputs=decoder_inputs_emb,
                    sequence_length=self.decoder_lengths
                )
                # decoder
                decoder = seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=projection_layer
                )
                # dynamic decoding
                self.final_outputs, self.final_state, self.final_sequence_lengths = seq2seq.dynamic_decode(
                    decoder=decoder,
                    swap_memory=True
                )
            else:
                start_tokens = tf.fill([batch_size], tgt_sos_id)
                end_token = tgt_eos_id

                # helper
                helper = seq2seq.GreedyEmbeddingHelper(
                    embedding=self.embeddings,
                    start_tokens=start_tokens,
                    end_token=end_token
                )
                # decoder
                decoder = seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=helper,
                    initial_state=decoder_initial_state,
                    output_layer=projection_layer
                )

                # dynamic decoding
                self.final_outputs, self.final_state, self.final_sequence_lengths = seq2seq.dynamic_decode(
                    decoder=decoder,
                    maximum_iterations=maximum_iterations,
                    swap_memory=True
                )

            self.logits = self.final_outputs.rnn_output
            self.sample_id = self.final_outputs.sample_id

        if mode=='train':
            # loss
            with tf.variable_scope('loss'):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.decoder_outputs, logits=self.logits)
                masks = tf.sequence_mask(
                    lengths=self.decoder_lengths,dtype=tf.float32)
                self.loss = tf.reduce_sum(cross_entropy * masks) / tf.to_float(batch_size)
                tf.summary.scalar('loss', self.loss)

            # summaries
            self.merged = tf.summary.merge_all()

            # train_op
            self.learning_rate = tf.Variable(learning_rate, trainable=False)
            self.global_step = tf.Variable(0, dtype=tf.int32)
            tvars = tf.trainable_variables()
            clipped_gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=clip_norm)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(clipped_gradients, tvars), global_step=self.global_step)

    def train_step(self,sess,batch_data):
        encoder_inputs,decoder_inputs,encoder_lengths,decoder_lengths=batch_data
        if self.target_vocab_size>0:
            for i in range(len(decoder_inputs)):
                for j in range(len(decoder_inputs[i])):
                    if decoder_inputs[i][j]>=self.target_vocab_size:
                        decoder_inputs[i][j]=1 #UNK
        decoder_inputs = np.array(decoder_inputs)
        decoder_outputs = decoder_inputs[:,1:]

        feed_dict={
            self.encoder_inputs:encoder_inputs,
            self.decoder_inputs:decoder_inputs,
            self.decoder_outputs:decoder_outputs,
            self.encoder_lengths:encoder_lengths,
            self.decoder_lengths:decoder_lengths,
        }
        _,loss,summaries,global_step = sess.run([self.train_op,self.loss,self.merged,self.global_step], feed_dict)
        return loss,summaries,global_step
    def eval_step(self,sess,batch_data):
        # encoder_inputs,encoder_lengths = batch_data
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
        decoder_outputs = np.array(decoder_inputs)[:,1:]

        query_id = []
        for item,length in zip(encoder_inputs,encoder_lengths):
            query_id.append(item[:length])

        golden_id = []
        for item,length in zip(decoder_outputs,decoder_lengths):
            golden_id.append(item[:length-1])

        feed_dict={
            self.encoder_inputs:encoder_inputs,
            self.encoder_lengths:encoder_lengths
        }
        sample_id,lengths = sess.run([self.sample_id,self.final_sequence_lengths],feed_dict)
        predict_id = []
        for item,length in zip(sample_id,lengths):
            predict_id.append(item[:length-1])
        return query_id,golden_id,predict_id

if __name__ == '__main__':
    pass













