#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:
#   Sining Sun (Northwestern Polytechnical University, China)
#   Chao Peng, EECS, Peking University, Beijing, China.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

sys.path.append('.')

from io_funcs.signal_processing import audiowrite, stft, istft
import io_funcs.kaldi_io as kio
from model.blstm import LSTM
from io_funcs.tfrecords_io import get_padded_batch
from local.utils import pp, show_all_variables
from io_funcs.signal_processing import *

FLAGS = None


def read_list_file(name, batch_size):
    file_name = os.path.join(FLAGS.lists_dir, name + ".lst")
    if not os.path.exists(file_name):
        tf.logging.fatal("File doesn't exist %s", file_name)
        sys.exit(-1) 
    config_file = open(file_name)
    tfrecords_lst = []

    for line in config_file:
        utt_id = line.strip().split()[0] 
        tfrecords_name = utt_id

        if not os.path.exists(tfrecords_name):
            tf.logging.fatal("TFRecords doesn't exist %s", tfrecords_name)
            sys.exit(-1)
        tfrecords_lst.append(tfrecords_name)

    num_batches = int(len(tfrecords_lst) / batch_size + 0.5)
    return tfrecords_lst, num_batches


def decode():
    """Decoding the inputs using current model."""
    tfrecords_lst, num_batches = read_list_file('tt_tf', FLAGS.batch_size)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                tt_mixed, tt_labels, tt_genders, tt_lengths = get_padded_batch(
                    tfrecords_lst, FLAGS.batch_size, FLAGS.input_size * 2,
                    FLAGS.output_size * 2, num_enqueuing_threads=1, num_epochs=1, shuffle=False)
                tt_inputs = tf.slice(tt_mixed, [0, 0, 0], [-1, -1, FLAGS.input_size])
                tt_angles = tf.slice(tt_mixed, [0, 0, FLAGS.input_size], [-1, -1, -1])
                # Create two models with train_input and val_input individually.
        with tf.name_scope('model'):
            model = LSTM(FLAGS, tt_inputs, tt_labels, tt_lengths, tt_genders, infer=True)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session()
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(FLAGS.TF_save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            tf.logging.info("Restore from " + ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.fatal("checkpoint not found.")
            sys.exit(-1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # cmvn_filename = os.path.join(FLAGS.date_dir, "/train_cmvn.npz")
        # if os.path.isfile(cmvn_filename):
        #    cmvn = np.load(cmvn_filename)
        # else:
        #    tf.logging.fatal("%s not exist, exit now." % cmvn_filename)
        #    sys.exit(-1)

    separated_dir = FLAGS.separated_dir
    if not os.path.exists(separated_dir):
        os.makedirs(separated_dir)
    processed = 0
    try:
        for batch in range(num_batches):
            if coord.should_stop():
                break
            # if FLAGS.assign == 'def':
            #    cleaned1, cleaned2,angles, lengths = sess.run([model._cleaned1, model._cleaned2,tt_angles, tt_lengths])
            # else:
            #    x1, x2  = model.get_opt_output()
            #    cleaned1, cleaned2, angles, lengths = sess.run([x1, x2, tt_angles, tt_lenghts])  # 原来的代码中单词打错
            cleaned1, cleaned2, angles, lengths = sess.run([model._cleaned1, model._cleaned2, tt_angles, tt_lengths])
            spec1 = cleaned1 * np.exp(angles * 1j)
            spec2 = cleaned2 * np.exp(angles * 1j)
            # sequence = activations * cmvn['stddev_labels'] + cmvn['mean_labels']
            for i in range(0, FLAGS.batch_size):
                tffilename = tfrecords_lst[i + processed]
                (_, name) = os.path.split(tffilename)  # ('c:\\csv', 'test.csv')
                (partname, _) = os.path.splitext(name)
                wav_name1 = separated_dir + '/' + partname + '_1.wav'
                wav_name2 = separated_dir + '/' + partname + '_2.wav'
                wav1 = istft(spec1[i, 0:lengths[i], :], size=FLAGS.window_size, shift=FLAGS.window_shift)
                wav2 = istft(spec2[i, 0:lengths[i], :], size=FLAGS.window_size, shift=FLAGS.window_shift)
                audiowrite(wav1, wav_name1, FLAGS.sample_rate, True, True)
                audiowrite(wav2, wav_name2, FLAGS.sample_rate, True, True)
            processed = processed + FLAGS.batch_size

            if (batch + 1) % 10 == 0:
                print((batch + 1) * FLAGS.batch_size)

    except Exception as e:
        # Report exceptions to the coordinator.
        coord.request_stop(e)
    finally:
        # Terminate as usual.  It is innocuous to request stop twice.
        coord.request_stop()
        coord.join(threads)

    tf.logging.info("Done decoding.")
    sess.close()


def train_one_epoch(sess, coord, tr_model, tr_num_batches):
    """Runs the model one epoch on given data."""
    tr_loss = 0
    for batch in range(tr_num_batches):
        if coord.should_stop():
            break
        _, loss = sess.run([tr_model.train_op, tr_model.loss])
        tr_loss += loss

        if (batch + 1) % 500 == 0:
            lr = sess.run(tr_model.lr)
            print("MINIBATCH %d: TRAIN AVG.LOSS %f, (learning rate %e)" % (
                  batch + 1, tr_loss / (batch + 1) / FLAGS.batch_size, lr))
            sys.stdout.flush()
    tr_loss /= (tr_num_batches * FLAGS.batch_size)

    return tr_loss


def eval_one_epoch(sess, coord, val_model, val_num_batches):
    """Cross validate the model on given data."""
    val_loss = 0
    for batch in range(val_num_batches):
        if coord.should_stop():
            break
        loss = sess.run(val_model._loss)
        val_loss += loss
    val_loss /= (val_num_batches * FLAGS.batch_size)

    return val_loss


def train():
    tr_tfrecords_lst,  tr_num_batches  = read_list_file("tr_tf", FLAGS.batch_size)
    val_tfrecords_lst, val_num_batches = read_list_file("cv_tf", FLAGS.batch_size)

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            with tf.name_scope('input'):
                # Tensor(25,?,258), (25,?,258), (25,1,2), (25,)
                tr_mixed, tr_labels, tr_genders, tr_lengths = get_padded_batch(
                    tr_tfrecords_lst, FLAGS.batch_size, FLAGS.input_size * 2, FLAGS.output_size * 2,
                    num_enqueuing_threads=FLAGS.num_threads, num_epochs=FLAGS.max_epochs)

                # Tensor(25,?,258), (25,?,258), (25,1,2), (25,)
                val_mixed, val_labels, val_genders, val_lengths = get_padded_batch(
                    val_tfrecords_lst, FLAGS.batch_size, FLAGS.input_size * 2, FLAGS.output_size * 2,
                    num_enqueuing_threads=FLAGS.num_threads, num_epochs=FLAGS.max_epochs + 1)

                tr_inputs = tf.slice(tr_mixed, [0, 0, 0], [-1, -1, FLAGS.input_size])
                val_inputs = tf.slice(val_mixed, [0, 0, 0], [-1, -1, FLAGS.input_size])

        with tf.name_scope('model'):
            tr_model = LSTM(FLAGS, tr_inputs, tr_labels, tr_lengths, tr_genders)
            # tr_model and val_model should share variables
            tf.get_variable_scope().reuse_variables()
            val_model = LSTM(FLAGS, val_inputs, val_labels, val_lengths, val_genders)
        show_all_variables()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Prevent exhausting all the gpu memories.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.99
        config.allow_soft_placement = True 
        # sess = tf.InteractiveSession(config=config)
        sess = tf.Session(config=config)
        sess.run(init)

        start_epoch = 0
        if FLAGS.resume_training.lower() == 'true':
            ckpt = tf.train.get_checkpoint_state(FLAGS.TF_save_dir)
            if ckpt and ckpt.model_checkpoint_path:
                tf.logging.info("restore from " + ckpt.model_checkpoint_path)
                tr_model.saver.restore(sess, ckpt.model_checkpoint_path)
                best_path = ckpt.model_checkpoint_path
                file_name = ckpt.model_checkpoint_path.split('/')[-1]
                iter_str = file_name.split('_')[1][4:]
                start_epoch = int(iter_str)
            else:
                tf.logging.fatal("checkpoint not found")

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # Cross validation before training.
            loss_prev = eval_one_epoch(sess, coord, val_model, val_num_batches)
            tf.logging.info("Crossval prerun avg.loss %.4F" % loss_prev)

            sess.run(tf.assign(tr_model.lr, FLAGS.learning_rate))
            for epoch in range(start_epoch, FLAGS.max_epochs):
                start_time = time.time()

                # Training
                tf.logging.info('Begin Training')
                tr_loss = train_one_epoch(sess, coord, tr_model, tr_num_batches)

                # Validation
                tf.logging.info('Begin Evaluation on Validation Set')
                val_loss = eval_one_epoch(sess, coord, val_model, val_num_batches)

                end_time = time.time()
                # Determine checkpoint path
                ckpt_name = "iter%d_lrate%e_tr%.4f_cv%.4f" % (
                            epoch + 1, FLAGS.learning_rate, tr_loss, val_loss)
                ckpt_dir = FLAGS.TF_save_dir
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                # Relative loss between previous and current val_loss
                rel_impr = (loss_prev - val_loss) / loss_prev
                # Accept or reject new parameters
                if val_loss < loss_prev:
                    tr_model.saver.save(sess, ckpt_path)
                    # Logging train loss along with validation loss
                    loss_prev = val_loss
                    best_path = ckpt_path
                    tf.logging.info("ITERATION %d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
                        " AVG.LOSS %.4f, %s (%s), TIME USED: %.2fs" % (
                        epoch + 1, tr_loss, FLAGS.learning_rate, val_loss,
                        "nnet accepted", ckpt_name, (end_time - start_time) / 1))
                else:
                    tr_model.saver.restore(sess, best_path)
                    tf.logging.info("ITERATION %d: TRAIN AVG.LOSS %.4f, (lrate%e) CROSSVAL"
                        " AVG.LOSS %.4f, %s, (%s), TIME USED: %.2fs" % (
                        epoch + 1, tr_loss, FLAGS.learning_rate, val_loss,
                        "nnet rejected", ckpt_name, (end_time - start_time) / 1))

                # Start halving when improvement is low
                if rel_impr < FLAGS.start_halving_impr:
                    FLAGS.learning_rate *= FLAGS.halving_factor
                    sess.run(tf.assign(tr_model.lr, FLAGS.learning_rate))

                # Stopping criterion
                if rel_impr < FLAGS.end_halving_impr:
                    if epoch < FLAGS.min_epochs:
                        tf.logging.info("we were supposed to finish, but we continue as "
                        "min_epochs : %s" % FLAGS.min_epochs)
                        continue
                    else:
                        tf.logging.info("finished, too small rel. improvement %g" % rel_impr)
                        break

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            # Terminate as usual.  It is innocuous to request stop twice.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)

        tf.logging.info("Done training!")
        sess.close()


def main(_):  # 里边参数不能省略
    if not os.path.exists(FLAGS.TF_save_dir):
        os.makedirs(FLAGS.TF_save_dir)
    if FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--separated_dir',
        type=str,
        default='storage/separated/PIT_BLSTM_3_496_2speaker_8KHz_0dB_def',
        help="Directory of train, val and test data."
    )
    parser.add_argument(
        '--lists_dir',
        type=str,
        default='storage/lists',
        help="Directory to load train, val and test data."
    )
    parser.add_argument(
        '--TF_save_dir', 
        type=str,
        default='storage/TFCheckpoint/PIT_BLSTM_3_496_2speaker_8KHz_0dB', 
        help="Directory to put the train result."
    )
    parser.add_argument(
        '--decode',
        type=int,
        default=0,
        # action='store_true',
        help="Flag indicating decoding or training."
    )
    parser.add_argument(
        '--resume_training',
        type=str,
        default='true', 
        help="Flag indicating whether to resume training from cptk."
    )

    parser.add_argument(
        '--input_size',
        type=int,
        default=129,
        help="The dimension of input."
    )
    parser.add_argument(
        '--output_size',
        type=int,
        default=129,
        help="The dimension of output."
    )
    parser.add_argument(
        '--czt_dim',
        type=int,
        default=0,
        help="chrip-z transform feats dimension. it should be 0 if you just use fft spectrum feats"
    )
    parser.add_argument(
        '--rnn_size',
        type=int,
        default=496, 
        help="Number of rnn units to use."
    )
    parser.add_argument(
        '--rnn_num_layers',
        type=int,
        default=3,
        help="Number of layer of rnn model."
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=25,
        help="Mini-batch size."
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help="Initial learning rate."
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=10, 
        help="Min number of epochs to run trainer without halving."
    )
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=50,
        help="Max number of epochs to run trainer totally."
    )
    parser.add_argument(
        '--halving_factor',
        type=float,
        default=0.5,
        help="Factor for halving."
    )
    parser.add_argument(
        '--start_halving_impr',
        type=float,
        default=0.003,
        help="Halving when ralative loss is lower than start_halving_impr."
    )
    parser.add_argument(
        '--end_halving_impr',
        type=float,
        default=0.01,
        help="Stop when relative loss is lower than end_halving_impr."
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=12,
        help='The num of threads to read tfrecords files.'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8,
        help="Keep probability for training dropout."
    )
    parser.add_argument(
        '--max_grad_norm',
        type=float,
        default=5.0,
        help="The max gradient normalization."
    )
    parser.add_argument(
        '--assign',
        type=str,
        default='def',
        help="Assignment method, def or opt"
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='BLSTM',
        help="BLSTM or LSTM"
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default='8000',
        help="sample rate of audio"
    )
    parser.add_argument(
        '--window_size',
        type=int, default='256',
        help="window size for STFT"
    )
    parser.add_argument(
        '--window_shift',
        type=int, default='128',
        help="window shift for STFT"
    )
    FLAGS, unparsed = parser.parse_known_args()
    pp.pprint(FLAGS.__dict__)
    sys.stdout.flush()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
