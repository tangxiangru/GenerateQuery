# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import utils
import random
import models
from progressbar import ProgressBar

#data path
tf.flags.DEFINE_string('dataset_path','data/train.txt','dataset_path')
tf.flags.DEFINE_string('mappings_path','data/mappings.pkl','mappings_path')
tf.flags.DEFINE_string('pre_trained_emb_path','data/Word60.model','pre_trained_emb_path')
tf.flags.DEFINE_string('out_dir','outputs','out_dir')
#model parameters
tf.flags.DEFINE_integer('encoder_num_units',128,'encoder_num_units')
tf.flags.DEFINE_integer('encoder_num_layers',1,'encoder_num_layers')
tf.flags.DEFINE_integer('decoder_num_units',128,'decoder_num_units')
tf.flags.DEFINE_integer('decoder_num_layers',1,'decoder_num_layers')
tf.flags.DEFINE_integer('source_vocab_size',-1,'source_vocab_size')
tf.flags.DEFINE_integer('target_vocab_size',-1,'target_vocab_size')
tf.flags.DEFINE_integer('emb_dim',60,'emb_dim')
tf.flags.DEFINE_float('dropout_emb',0.4,'dropout_emb')
tf.flags.DEFINE_float('dropout_hidden',0.4,'dropout_hidden')
tf.flags.DEFINE_float('learning_rate',0.001,'learning_rate')
tf.flags.DEFINE_float('clip_norm',5,'clip_norm')
tf.flags.DEFINE_string('attention_option','scaled_luong','attention_option')
tf.flags.DEFINE_integer('beam_size',0,'beam_size')
tf.flags.DEFINE_string('optimizer','adam','optimizer')
tf.flags.DEFINE_integer('maximum_iterations',6,'maximum_iterations')
#train parameters
tf.flags.DEFINE_string('model_name','New_Pointer_Generator','model_name')
tf.flags.DEFINE_integer('batch_size',64,'batch_size')
tf.flags.DEFINE_float('memory_usage',1.0,'memory_usage')
tf.flags.DEFINE_integer('max_epoch',200,'max_epoch')
tf.flags.DEFINE_integer('max_no_improve',10,'max_no_improve')
tf.flags.DEFINE_boolean('clean',True,'clean')
tf.flags.DEFINE_integer('seed',24,'seed')
tf.flags.DEFINE_boolean('restore',False,'restore')

FLAGS=tf.flags.FLAGS

#Clean all previous data!
if FLAGS.clean and FLAGS.restore is False:
    print "Clean all previous data!"
    os.system("rm -rf %s" % FLAGS.mappings_path)
    os.system("rm -rf %s" % FLAGS.out_dir)


def train():
    query_list, keywords_list = utils.read_data(FLAGS.dataset_path)
    # shuffle data
    data = zip(query_list,keywords_list)
    random.seed(FLAGS.seed)
    random.shuffle(data)
    query_list,keywords_list = zip(*data)

    item2id, id2item, target_vocab_size = utils.load_mappings(query_list, keywords_list,mappings_path=FLAGS.mappings_path,
                                                                       source_vocab_size=FLAGS.source_vocab_size,
                                                                       target_vocab_size=FLAGS.target_vocab_size)
    vocab_size = len(item2id)
    print vocab_size,target_vocab_size

    query,keywords = utils.prepare_dataset(query_list,keywords_list,item2id)

    data=zip(query,keywords)
    train_data,dev_data,test_data=data[:int(len(data)*0.8)],data[int(len(data)*0.8):int(len(data)*0.9)],data[int(len(data)*0.9):]
    train_manager = utils.BatchManager(train_data,FLAGS.batch_size,shuffle=True)
    dev_manager = utils.BatchManager(dev_data,FLAGS.batch_size,shuffle=False)
    test_manager = utils.BatchManager(test_data,FLAGS.batch_size,shuffle=False)


    dev_goldens = keywords_list[int(len(data)*0.8):int(len(data)*0.9)]
    test_goldens = keywords_list[int(len(data)*0.9):]

    model_dir = os.path.join(FLAGS.out_dir,'models')
    summary_dir = os.path.join(FLAGS.out_dir,'summries')
    if not os.path.exists(FLAGS.out_dir):
        os.mkdir(FLAGS.out_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory_usage
    with tf.Session(config=tfConfig) as sess:
        with tf.variable_scope("root"):
            train_model=getattr(models,FLAGS.model_name)('train',vocab_size,target_vocab_size,FLAGS.emb_dim,
                        FLAGS.encoder_num_units,FLAGS.encoder_num_layers,FLAGS.decoder_num_units,FLAGS.decoder_num_layers,
                        FLAGS.dropout_emb,FLAGS.dropout_hidden,item2id[utils.START],item2id[utils.END],
                        FLAGS.learning_rate,FLAGS.clip_norm,FLAGS.attention_option,FLAGS.beam_size,FLAGS.optimizer,FLAGS.maximum_iterations)
        with tf.variable_scope("root",reuse=True):
            dev_model=getattr(models,FLAGS.model_name)('infer',vocab_size,target_vocab_size,FLAGS.emb_dim,
                        FLAGS.encoder_num_units,FLAGS.encoder_num_layers,FLAGS.decoder_num_units,FLAGS.decoder_num_layers,
                        FLAGS.dropout_emb,FLAGS.dropout_hidden,item2id[utils.START],item2id[utils.END],
                        FLAGS.learning_rate,FLAGS.clip_norm,FLAGS.attention_option,FLAGS.beam_size,FLAGS.optimizer,FLAGS.maximum_iterations)
            dev_f1_value = tf.placeholder(dtype=tf.float32,name='dev_f1')
            dev_f1_summary = tf.summary.scalar(name='dev_f1',tensor=dev_f1_value)
        with tf.variable_scope('root',reuse=True):
            test_model=getattr(models,FLAGS.model_name)('infer',vocab_size,target_vocab_size,FLAGS.emb_dim,
                        FLAGS.encoder_num_units,FLAGS.encoder_num_layers,FLAGS.decoder_num_units,FLAGS.decoder_num_layers,
                        FLAGS.dropout_emb,FLAGS.dropout_hidden,item2id[utils.START],item2id[utils.END],
                        FLAGS.learning_rate,FLAGS.clip_norm,FLAGS.attention_option,FLAGS.beam_size,FLAGS.optimizer,FLAGS.maximum_iterations)
            test_f1_value = tf.placeholder(dtype=tf.float32,name='test_f1')
            test_f1_summary = tf.summary.scalar(name='test_F1',tensor=test_f1_value)
        with tf.variable_scope('training_procedure'):
            best_epoch = tf.get_variable('best_epoch',shape=[],initializer=tf.zeros_initializer(),trainable=False,dtype=tf.int32)
            best_dev_score = tf.get_variable('best_dev_score',shape=[],initializer=tf.zeros_initializer(),trainable=False,dtype=tf.float32)
            best_test_score = tf.get_variable('best_test_score',shape=[],initializer=tf.zeros_initializer(),trainable=False,dtype=tf.float32)
        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(summary_dir)
        # try:
        if FLAGS.restore is not False:
            checkpoint = tf.train.latest_checkpoint(model_dir)
            saver.restore(sess, checkpoint)
            print 'Restore model from %s.' % checkpoint

        else:
            sess.run(tf.global_variables_initializer())
            if os.path.exists(FLAGS.pre_trained_emb_path):
                print "Loading pre_trained word embeddings from %s" % FLAGS.pre_trained_emb_path
                pre_embeddings = utils.load_emb(item2id,FLAGS.pre_trained_emb_path,FLAGS.emb_dim)
                sess.run(train_model.embeddings.assign(pre_embeddings))
                del pre_embeddings

        start_epoch,best_dev_f1, best_test_f1 = sess.run([best_epoch,best_dev_score,best_test_score])
        no_improve = 0
        print 'Train start!'
        sess.run(train_model.learning_rate.assign(FLAGS.learning_rate))
        for epoch in range(start_epoch+1,FLAGS.max_epoch):
            # train
            train_loss = []
            bar = ProgressBar(max_value=train_manager.num_batch)
            for batch_data in bar(train_manager.iter_batch()):
                batch_loss,summaries,global_step=train_model.train_step(sess,batch_data)
                #add summaries to tensorboard
                summary_writer.add_summary(summaries,global_step)
                train_loss.append(batch_loss)
            print "Epoch %d finished. Loss: %.4f" % (epoch,np.mean(train_loss))

            # dev
            querys = []
            predicts = []
            goldens = []
            for batch_data in dev_manager.iter_batch():
                encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
                query_id,golden_id,predict_id=dev_model.eval_step(sess,batch_data)
                querys.extend(query_id)
                goldens.extend(golden_id)
                predicts.extend(predict_id)
            # dev_p,dev_r,dev_f1 = utils.evaluate_scores(querys,predicts,goldens,id2item,FLAGS.out_dir)
            dev_p, dev_r, dev_f1 = utils.evaluate_scores2(querys, predicts, dev_goldens,goldens, id2item, FLAGS.out_dir)
            print "Dev   precision / recall / f1 score: %.2f / %.2f / %.2f" % (dev_p,dev_r,dev_f1)

            # test
            querys = []
            predicts = []
            goldens = []
            for batch_data in test_manager.iter_batch():
                encoder_inputs, decoder_inputs, encoder_lengths, decoder_lengths = batch_data
                query_id,golden_id,predict_id=test_model.eval_step(sess,batch_data)
                querys.extend(query_id)
                goldens.extend(golden_id)
                predicts.extend(predict_id)

            test_p,test_r,test_f1 = utils.evaluate_scores(querys,predicts,goldens,id2item,FLAGS.out_dir)
            #test_p, test_r, test_f1 = utils.evaluate_scores2(querys, predicts, test_goldens,goldens, id2item, FLAGS.out_dir)
            print "Test   precision / recall / f1 score: %.2f / %.2f / %.2f" % (test_p,test_r,test_f1)
            print "Best dev f1: %.2f    test f1: %.2f   epoch:%d\n" % (best_dev_f1, best_test_f1,int(sess.run(best_epoch)))

            summary_writer.add_summary(sess.run(dev_f1_summary,feed_dict={dev_f1_value:dev_f1}),global_step=epoch+1)
            summary_writer.add_summary(sess.run(test_f1_summary, feed_dict={test_f1_value: dev_f1}),global_step=epoch + 1)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                sess.run(best_epoch.assign(epoch))
                sess.run(best_dev_score.assign(best_dev_f1))
                sess.run(best_test_score.assign(best_test_f1))
                saver.save(sess,os.path.join(model_dir,FLAGS.model_name))
                no_improve =0
            else:
                no_improve = no_improve+1
                if no_improve >= FLAGS.max_no_improve:
                    break
        print "Best dev f1: %.2f    test f1: %.2f   epoch:%d" % (best_dev_f1,best_test_f1,int(best_epoch.eval()))

if __name__ == '__main__':
    train()
