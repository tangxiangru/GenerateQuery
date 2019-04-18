# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest
from tensorflow.python.layers import core as layers_core

"""
Extractive+Generative Method
Pointer decoder + Generator decoder
Our novel encoder-decoder model
"""
class TrainHelper(seq2seq.Helper):
    """
    Decoder Helper for Train.
    """
    def __init__(self,decoder_inputs,decoder_sequence_length):
        self.decoder_inputs = decoder_inputs
        self.decoder_sequence_length = decoder_sequence_length

    @property
    def batch_size(self):
        return tf.shape(self.decoder_inputs)[0]
    @property
    def sample_ids_shape(self):
        pass
    @property
    def sample_ids_dtype(self):
        pass

    def initialize(self, name=None):
        finished = tf.equal(0, self.decoder_sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(all_finished,
                              lambda: tf.zeros_like(self.decoder_inputs[:,0,:]),
                              lambda: self.decoder_inputs[:,0,:]) #[batch_size,emb_dim]
        return (finished, next_inputs)

    def sample(self, time, outputs, name=None, **unused_kwargs):
        with tf.name_scope(name, "TrainingHelperSample"):
            sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
            return sample_ids #[batch_size]

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        next_time = time + 1
        finished = (next_time >= self.decoder_sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(all_finished,
                              lambda: tf.zeros_like(self.decoder_inputs[:,0,:]),
                              lambda: self.decoder_inputs[:,next_time,:]) #[batch_size,emb_dim]
        return (finished, next_inputs, state)

class GreedyEmbeddingHelper(seq2seq.Helper):
    """
    Decoder Helper for infer.
    """
    def __init__(self,embeddings, start_tokens, end_token):
        self.embeddings=embeddings
        self.start_tokens = start_tokens
        self.end_token = end_token
        self.start_inputs = self._embedding_fn(start_tokens)

    @property
    def batch_size(self):
        return tf.shape(self.start_tokens)[0]
    @property
    def sample_ids_shape(self):
        pass
    @property
    def sample_ids_dtype(self):
        pass

    def _embedding_fn(self,sample_ids):
        return tf.nn.embedding_lookup(self.embeddings,sample_ids)

    def initialize(self, name=None):
        finished = tf.tile([False], [self.batch_size])
        return (finished, self.start_inputs)

    def sample(self, time, outputs, name=None):
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, outputs  # unused by next_inputs_fn
        finished = tf.equal(sample_ids, self.end_token)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(all_finished,
                              # If we're finished, the next_inputs value doesn't matter
                              lambda: self.start_inputs,
                              lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)

class PointerDecoder(seq2seq.Decoder):
    def __init__(self, cell, helper, initial_state,
                 encoder_inputs_ids, encoder_sequence_length, encoder_outputs,
                 vocab_size, target_vocab_size, num_units, emb_dim):
        self.cell=cell
        self.helper=helper
        self.initial_state=initial_state
        self.encoder_inputs_ids=encoder_inputs_ids
        self.encoder_sequence_length=encoder_sequence_length
        self.encoder_outputs=encoder_outputs
        self.vocab_size=vocab_size
        self.target_vocab_size=target_vocab_size

        #attention weights
        self.Wh=layers_core.Dense(num_units, use_bias=False, activation=None)
        self.Ws=tf.get_variable(name='Ws',shape=[num_units,num_units])
        self.v = layers_core.Dense(1, use_bias=False, activation=None)

    @property
    def batch_size(self):
        return tf.shape(self.encoder_sequence_length)[0]

    @property
    def output_size(self):
        return seq2seq.BasicDecoderOutput(rnn_output=self.vocab_size, sample_id=tf.TensorShape([]))

    @property
    def output_dtype(self):
        return seq2seq.BasicDecoderOutput(rnn_output=tf.float32, sample_id=tf.int32)

    def initialize(self, name=None):
        """
        Initialize the decoder.
        """
        return self.helper.initialize() + (self.initial_state,) #(finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self.cell(inputs, state) #cell_outputs [batch_size,num_units]

        #attention
        attn1=self.Wh(self.encoder_outputs)+tf.expand_dims(tf.matmul(cell_outputs,self.Ws),1) #[batch_size,enc_seq,num_units]
        attn2=tf.squeeze(self.v(tf.tanh(attn1)),axis=[2]) #[batch_size,enc_seq]
        encoded_mask = (tf.sequence_mask(self.encoder_sequence_length, dtype=tf.float32,name='encoded_mask') - 1) * 1e6
        attention_weight=tf.nn.softmax(attn2+encoded_mask) #[batch_size,enc_seq]
        context=tf.reduce_sum(self.encoder_outputs*tf.expand_dims(attention_weight,2),1) #[batch_size,num_units]

        #add p_copy to p_mix
        p_copy = attention_weight  # [batch_size,enc_seq]
        expand_p_copy = tf.expand_dims(p_copy,2) #[batch_size,enc_seq,1]
        encoder_inputs_mask = tf.one_hot(self.encoder_inputs_ids, self.vocab_size) #[batch_size,enc_seq,vocab_size]
        p_copy_to_mix = tf.reduce_sum(encoder_inputs_mask*expand_p_copy, 1) #[batch_size,vocab_size]

        p_mix = p_copy_to_mix
        p_mix = tf.maximum(p_mix, tf.zeros_like(p_mix, tf.float32) + 1e-12)

        sample_ids = self.helper.sample(time=time, outputs=p_mix)
        (finished, next_inputs, next_state) = self.helper.next_inputs(
            time=time,
            outputs=p_mix,
            state=cell_state,
            sample_ids=sample_ids)

        outputs = seq2seq.BasicDecoderOutput(tf.concat([p_mix,cell_outputs],-1), sample_ids)
        return (outputs, next_state, next_inputs, finished)

class GeneratorDecoder(seq2seq.Decoder):
    def __init__(self, cell, helper, initial_state,
                 encoder_sequence_length, encoder_outputs,
                 decoder_sequence_length, decoder_outputs,
                 vocab_size, target_vocab_size, num_units, emb_dim):
        self.cell=cell
        self.helper=helper
        self.initial_state=initial_state
        self.encoder_sequence_length=encoder_sequence_length
        self.encoder_outputs=encoder_outputs
        self.decoder_sequence_length=decoder_sequence_length
        self.decoder_outputs=decoder_outputs
        self.vocab_size=vocab_size
        self.target_vocab_size=target_vocab_size

        #attention weights
        self.Wh=layers_core.Dense(num_units, use_bias=False, activation=None)
        self.Ws=tf.get_variable(name='Ws',shape=[num_units,num_units])
        self.v = layers_core.Dense(1, use_bias=False, activation=None)

        self.Wh2=layers_core.Dense(num_units, use_bias=False, activation=None)
        self.Ws2=tf.get_variable(name='Ws2',shape=[num_units,num_units])
        self.v2 = layers_core.Dense(1, use_bias=False, activation=None)

        self.V1 = tf.get_variable(name='V1', shape=[num_units * 4, target_vocab_size])
        self.b1 = tf.get_variable(name='b1', shape=[target_vocab_size], initializer=tf.zeros_initializer)

    @property
    def batch_size(self):
        return tf.shape(self.encoder_sequence_length)[0]

    @property
    def output_size(self):
        return seq2seq.BasicDecoderOutput(rnn_output=self.vocab_size, sample_id=tf.TensorShape([]))

    @property
    def output_dtype(self):
        return seq2seq.BasicDecoderOutput(rnn_output=tf.float32, sample_id=tf.int32)

    def initialize(self, name=None):
        """
        Initialize the decoder.
        """
        return self.helper.initialize() + (self.initial_state,) #(finished, first_inputs, initial_state)

    def step(self, time, inputs, state, name=None):
        cell_outputs, cell_state = self.cell(inputs, state) #cell_outputs [batch_size,num_units]

        #attention
        attn1=self.Wh(self.encoder_outputs)+tf.expand_dims(tf.matmul(cell_outputs,self.Ws),1) #[batch_size,enc_seq,num_units]
        attn2=tf.squeeze(self.v(tf.tanh(attn1)),axis=[2]) #[batch_size,enc_seq]
        encoded_mask = (tf.sequence_mask(self.encoder_sequence_length, dtype=tf.float32,name='encoded_mask') - 1) * 1e6
        attention_weight=tf.nn.softmax(attn2+encoded_mask) #[batch_size,enc_seq]
        context=tf.reduce_sum(self.encoder_outputs*tf.expand_dims(attention_weight,2),1) #[batch_size,num_units]

        attn3=self.Wh2(self.decoder_outputs)+tf.expand_dims(tf.matmul(cell_outputs,self.Ws2),1) #[batch_size,enc_seq,num_units]
        attn4=tf.squeeze(self.v2(tf.tanh(attn3)),axis=[2]) #[batch_size,enc_seq]
        decoded_mask = (tf.sequence_mask(self.decoder_sequence_length, dtype=tf.float32,name='decoded_mask') - 1) * 1e6
        attention_weight2=tf.nn.softmax(attn4+decoded_mask) #[batch_size,enc_seq]
        context2=tf.reduce_sum(self.decoder_outputs*tf.expand_dims(attention_weight2,2),1) #[batch_size,num_units]

        p_vocab = tf.matmul(tf.concat([cell_outputs, context, context2], axis=-1),
                            self.V1) + self.b1  # [batch_size,target_vocab_size]
        p_vocab = tf.nn.softmax(p_vocab)
        p_vocab_to_mix = tf.concat([p_vocab, tf.zeros(shape=[self.batch_size, self.vocab_size - self.target_vocab_size])], 1)
        p_mix = p_vocab_to_mix
        p_mix = tf.maximum(p_mix, tf.zeros_like(p_mix, tf.float32) + 1e-12)

        sample_ids = self.helper.sample(time=time, outputs=p_mix)
        (finished, next_inputs, next_state) = self.helper.next_inputs(
            time=time,
            outputs=p_mix,
            state=cell_state,
            sample_ids=sample_ids)

        outputs = seq2seq.BasicDecoderOutput(p_mix, sample_ids)
        return (outputs, next_state, next_inputs, finished)

class New_Pointer_Generator(object):
    """
    Extractive+Generative Method
    Pointer decoder + Generator decoder
    Our novel encoder-decoder model
    """
    def __init__(self,mode,vocab_size,target_vocab_size,emb_dim,encoder_num_units,encoder_num_layers,
                 decoder_num_units, decoder_num_layers,dropout_emb,dropout_hidden,tgt_sos_id,tgt_eos_id,
                 learning_rate,clip_norm,attention_option,beam_size,optimizer,maximum_iterations):

        assert mode in ["train", "infer"], "invalid mode!"

        # inputs
        self.encoder_inputs = tf.placeholder(tf.int32,shape=[None,None],name='encoder_inputs')
        self.encoder_lengths = tf.placeholder(tf.int32,shape=[None],name='encoder_lengths')

        self.decoder_inputs = tf.placeholder(tf.int32,shape=[None,None],name='decoder_inputs')
        self.decoder_outputs = tf.placeholder(tf.int32,shape=[None,None],name='decoder_outputs')
        self.decoder_lengths = tf.placeholder(tf.int32,shape=[None],name='decoder_lengths')

        self.decoder_inputs2 = tf.placeholder(tf.int32,shape=[None,None],name='decoder_inputs2')
        self.decoder_outputs2 = tf.placeholder(tf.int32,shape=[None,None],name='decoder_outputs2')
        self.decoder_lengths2 = tf.placeholder(tf.int32,shape=[None],name='decoder_lengths2')

        self.batch_size=tf.shape(self.encoder_lengths)[0]
        self.vocab_size=vocab_size
        self.target_vocab_size=target_vocab_size

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
            encoder_outputs = tf.concat([encoder_outputs_fw,encoder_outputs_bw], 2) #[batch_size,enc_seq,lstm_dim*2]

            # A linear layer to reduce the encoder's final FW and BW state into a single initial state for the decoder.
            # This is needed because the encoder is bidirectional but the decoder is not.
            encoder_states_c = tf.layers.dense(inputs=tf.concat([encoder_state_fw.c,encoder_state_bw.c],axis=-1),
                                             units=encoder_num_units,activation=None,use_bias=False)
            encoder_states_h = tf.layers.dense(inputs=tf.concat([encoder_state_fw.h, encoder_state_bw.h], axis=-1),
                                             units=encoder_num_units, activation=None, use_bias=False)
            encoder_states = rnn.LSTMStateTuple(encoder_states_c,encoder_states_h)

        # Decoder
        with tf.variable_scope('pointer_decoder'):
            decoder_inputs_emb = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs)
            if mode == 'train':
                decoder_inputs_emb = tf.nn.dropout(decoder_inputs_emb, 1 - dropout_emb)

            # decoder_rnn_cell
            decoder_cell = cell(decoder_num_units)
            decoder_initial_state = encoder_states

            # train/infer
            if mode=='train':
                # helper
                helper = TrainHelper(decoder_inputs=decoder_inputs_emb,
                                     decoder_sequence_length=self.decoder_lengths)
                maximum_iterations=None
            else:
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id
                helper = GreedyEmbeddingHelper(embeddings=self.embeddings,
                                               start_tokens=start_tokens,
                                               end_token=end_token)
            pointer_decoder=PointerDecoder(cell=decoder_cell,
                                           helper=helper,
                                           initial_state=decoder_initial_state,
                                           encoder_inputs_ids=self.encoder_inputs,
                                           encoder_sequence_length=self.encoder_lengths,
                                           encoder_outputs=encoder_outputs,
                                           vocab_size=vocab_size,
                                           target_vocab_size=target_vocab_size,
                                           num_units=encoder_num_units,
                                           emb_dim=emb_dim)

            # dynamic decoding
            self.final_outputs, self.final_state, self.final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=pointer_decoder,
                maximum_iterations=maximum_iterations,
                swap_memory=True
            )

            self.probs = self.final_outputs.rnn_output #Note:prob_distribution!!!
            self.sample_id = self.final_outputs.sample_id
            self.probs1 = self.probs[:,:,:-decoder_num_units]
            decoder_outputs=self.probs[:,:,-decoder_num_units:]
            #decoder_outputs = tf.nn.embedding_lookup(self.embeddings,self.final_outputs.sample_id)


        with tf.variable_scope('generator_decoder'):
            decoder_inputs_emb2 = tf.nn.embedding_lookup(self.embeddings, self.decoder_inputs2)
            if mode == 'train':
                decoder_inputs_emb2 = tf.nn.dropout(decoder_inputs_emb2, 1 - dropout_emb)

            # decoder_rnn_cell
            decoder_cell2 = cell(decoder_num_units)
            decoder_initial_state2 = encoder_states

            # train/infer
            if mode == 'train':
                # helper
                helper2 = TrainHelper(decoder_inputs=decoder_inputs_emb2,
                                     decoder_sequence_length=self.decoder_lengths2)
                maximum_iterations = None
                decoder_sequence_length=self.decoder_lengths
            else:
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id
                helper2 = GreedyEmbeddingHelper(embeddings=self.embeddings,
                                               start_tokens=start_tokens,
                                               end_token=end_token)
                decoder_sequence_length=self.final_sequence_lengths
            generator_decoder=GeneratorDecoder(cell=decoder_cell2,
                                           helper=helper2,
                                           initial_state=decoder_initial_state,
                                           encoder_sequence_length=self.encoder_lengths,
                                           encoder_outputs=encoder_outputs,
                                           decoder_sequence_length=decoder_sequence_length,
                                           decoder_outputs=decoder_outputs,
                                           vocab_size=vocab_size,
                                           target_vocab_size=target_vocab_size,
                                           num_units=encoder_num_units,
                                           emb_dim=emb_dim)

            # dynamic decoding
            self.final_outputs2, self.final_state2, self.final_sequence_lengths2 = seq2seq.dynamic_decode(
                decoder=generator_decoder,
                maximum_iterations=maximum_iterations,
                swap_memory=True
            )

            self.probs2 = self.final_outputs2.rnn_output #Note:prob_distribution!!!
            self.sample_id2 = self.final_outputs2.sample_id


        if mode=='train':
            # loss
            with tf.variable_scope('loss'):
                labels = tf.one_hot(self.decoder_outputs,vocab_size) #[batch_size,dec_seq,vocab_size]
                cross_entropy = tf.reduce_sum(-labels*tf.log(self.probs1),-1) #[batch_size,dec_seq]
                masks = tf.sequence_mask(lengths=self.decoder_lengths,dtype=tf.float32)
                self.loss1 = tf.reduce_sum(cross_entropy * masks) / tf.to_float(self.batch_size)
                labels2 = tf.one_hot(self.decoder_outputs2, vocab_size)  # [batch_size,dec_seq,vocab_size]
                cross_entropy2 = tf.reduce_sum(-labels2 * tf.log(self.probs2), -1)  # [batch_size,dec_seq]
                masks2 = tf.sequence_mask(lengths=self.decoder_lengths2, dtype=tf.float32)
                self.loss2 = tf.reduce_sum(cross_entropy2 * masks2) / tf.to_float(self.batch_size)
                self.loss = self.loss1*0.5+self.loss2*0.5
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

    def train_step(self, sess, batch_data):
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
        if self.target_vocab_size > 0:
            for i in range(len(decoder_inputs)):
                for j in range(len(decoder_inputs[i])):
                    if decoder_inputs[i][j]>=self.target_vocab_size and decoder_inputs[i][j] not in encoder_inputs[i]:
                        decoder_inputs[i][j]=1 #UNK

        new_encoder_inputs=[]
        new_encoder_lengths=[]
        for item,length in zip(encoder_inputs,encoder_lengths):
            new_encoder_inputs.append(item[:length]+[3]+item[length:])
            new_encoder_lengths.append(length+1)

        new_decoder_inputs1=[]
        new_decoder_inputs2=[]
        decoder_inputs_lengths1= []
        decoder_inputs_lengths2 =[]
        for i in range(len(decoder_inputs)):
            temp1=[2]
            temp2=[2]
            for j in range(1,decoder_lengths[i]):
                if decoder_inputs[i][j] in encoder_inputs[i]:
                    temp1.append(decoder_inputs[i][j])
                elif decoder_inputs[i][j]<self.target_vocab_size:
                    temp2.append(decoder_inputs[i][j])
                else:
                    temp2.append(1)
            temp1.append(3)
            temp2.append(3)
            new_decoder_inputs1.append(temp1)
            new_decoder_inputs2.append(temp2)
            decoder_inputs_lengths1.append(len(temp1)-1)
            decoder_inputs_lengths2.append(len(temp2)-1)

        max_decoder1=max(decoder_inputs_lengths1)+1
        max_decoder2=max(decoder_inputs_lengths2)+1

        pad_new_decoder_inputs1=[]
        pad_new_decoder_inputs2=[]

        for item1,item2 in zip(new_decoder_inputs1,new_decoder_inputs2):
            pad_item1=[0]*(max_decoder1-len(item1))
            pad_item2=[0]*(max_decoder2-len(item2))
            pad_new_decoder_inputs1.append(item1+pad_item1)
            pad_new_decoder_inputs2.append(item2+pad_item2)

        decoder_inputs1 = np.array(pad_new_decoder_inputs1)
        decoder_outputs1 = decoder_inputs1[:, 1:]
        decoder_inputs2 = np.array(pad_new_decoder_inputs2)
        decoder_outputs2 = decoder_inputs2[:, 1:]

        feed_dict = {
            self.encoder_inputs: new_encoder_inputs,
            self.encoder_lengths: new_encoder_lengths,
            self.decoder_inputs:decoder_inputs1,
            self.decoder_outputs:decoder_outputs1,
            self.decoder_lengths:decoder_inputs_lengths1,
            self.decoder_inputs2:decoder_inputs2,
            self.decoder_outputs2: decoder_outputs2,
            self.decoder_lengths2:decoder_inputs_lengths2
        }
        _,loss, summaries, global_step = sess.run([self.train_op,self.loss, self.merged, self.global_step], feed_dict)
        return loss, summaries, global_step

    def eval_step(self, sess, batch_data):
        encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
        decoder_inputs = np.array(decoder_inputs)
        decoder_outputs = decoder_inputs[:,1:]

        query_id = []
        for item,length in zip(encoder_inputs,encoder_lengths):
            query_id.append(item[:length])

        golden_id = []
        for item,length in zip(decoder_outputs,decoder_lengths):
            golden_id.append(item[:length-1])

        new_encoder_inputs=[]
        new_encoder_lengths=[]
        for item,length in zip(encoder_inputs,encoder_lengths):
            new_encoder_inputs.append(item[:length]+[3]+item[length:])
            new_encoder_lengths.append(length+1)

        feed_dict={
            self.encoder_inputs:new_encoder_inputs,
            self.encoder_lengths:new_encoder_lengths
        }

        sample_id,lengths = sess.run([self.sample_id,self.final_sequence_lengths],feed_dict)
        sample_id2, lengths2 = sess.run([self.sample_id2, self.final_sequence_lengths2], feed_dict)
        predict_id = []
        for item,item2,length,length2 in zip(sample_id,sample_id2,lengths,lengths2):
            if length2>=2 and length>=2:
                temp=[]
                temp.extend(item[:length-1])
                temp.extend(item2[:length2-1])
                predict_id.append(temp)
            elif length>=2:
                predict_id.append(item[:length - 1])
            elif length2>=2:
                predict_id.append(item2[:length2 - 1])
            else:
                predict_id.append([])
        return query_id,golden_id,predict_id

if __name__ == '__main__':
    pass