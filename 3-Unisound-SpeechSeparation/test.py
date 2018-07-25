"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""
from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
import sys
import time
import librosa
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import ctypes
import scipy
from model import SpeechSeparation
from audio import AudioReader
from ops import optimizer_factory

# hyper parameters
NUM_OF_FREQUENCY_POINTS = 257
BATCH_SIZE = 1
DATA_DIRECTORY = './pinao-corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 2000
NUM_STEPS = int(1e6)
LEARNING_RATE = 1e-4
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 10
METADATA = False
N_SEQS = 10  # Number of samples to generate every time monitoring.
N_SECS = 3
SAMPLE_RATE = 16000
NUM_GPU = 1
fs = 16000
framesz = 0.032
hop = framesz * 0.5


def stft(x, fs, framesz, hop):
    """
     x - signal
     fs - sample rate
     framesz - frame size
     hop - hop size (frame size = overlap + hop size)
    """
    audio = np.reshape(x, (-1))
    framesamp = int(framesz * fs)
    hopsamp = int(hop * fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fftpack.fft(w * audio[i:i + framesamp])
                     for i in range(0, len(x) - framesamp, hopsamp)])
    return X, X[:, :X.shape[1] / 2 + 1]


def istft(X, fs, T, hop):
    # x = scipy.zeros(T*fs)
    x = scipy.zeros(T)
    framesamp = X.shape[1]
    hopsamp = int(hop * fs)
    for n, i in enumerate(range(0, len(x) - framesamp, hopsamp)):
        x[i:i + framesamp] += scipy.real(scipy.fftpack.ifft(X[n]))
    return x


def irstft(X, fs, T, hop):
    # x = scipy.zeros(T*fs)
    x = scipy.zeros(T)
    framesamp = X.shape[1]
    hopsamp = int(hop * fs)
    for n, i in enumerate(range(0, len(x) - framesamp, hopsamp)):
        x[i:i + framesamp] += scipy.real(scipy.fftpack.irfft(X[n]))
    return x


# an audio-making function
def mk_audio(output, output_angle, num):
    fs = 16000
    framesz = 0.032
    hop = framesz * 0.5
    output = np.reshape(output, (output.shape[1], output.shape[2]))
    output_re = output * np.cos(output_angle) + 1j * output * np.sin(output_angle)

    # np.column_stack: stack 1-D arrays as columns into a 2-D array
    # np.conj: return the complex conjugate, element-wise
    output_re = np.column_stack((output_re, np.conj(output_re[:, 1:-1].T[::-1].T)))
    x_r = istft(output_re, fs, (output.shape[1] + 1) * 256, hop)
    scipy.io.wavfile.write("speeker" + str(num) + "_test.wav", fs, x_r)
    return


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--num_gpus', type=int, default=NUM_GPU,
                        help='num of gpus.. Default: ' + str(NUM_GPU) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--num_of_frequency_points', type=int, default=NUM_OF_FREQUENCY_POINTS,
                        help='num_of_frequency_points. Default: '
                             + str(NUM_OF_FREQUENCY_POINTS) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--test_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                             '(execution time, memory consumption) for use with '
                             'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                             'information for TensorBoard. '
                             'If the model already exists, it will restore '
                             'the state and will continue training. '
                             'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                             'output and generated model. These are stored '
                             'under the dated subdirectory of --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                             'This creates the new model under the dated directory '
                             'in --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='sample rate for training. Default: ' + str(SAMPLE_RATE) + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                             'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                             'Default: False')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                             'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                                               'used by sgd or rmsprop optimizer. Ignored by the '
                                               'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')

    def t_or_f(arg):
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
            raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        ivalue = int(value)
        if ivalue < 0:
            raise argparse.ArgumentTypeError("%s is not non-negative!" % value)
        return ivalue

    def check_positive(value):
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("%s is not positive!" % value)
        return ivalue

    def check_unit_interval(value):
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
            raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" % value)
        return fvalue

    '''
    # TODO: Fix the descriptions
    # Hyperparameter arguements:
    #parser.add_argument('--exp', help='Experiment name',
    #        type=str, required=False, default='_')
    '''
    parser.add_argument('--seq_len', help='How many samples to include in each\
            Truncated BPTT pass', type=check_positive, required=True)
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'],
                        required=True)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs',
                        type=check_positive, required=True)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN',
                        type=check_positive, choices=range(1, 6), required=True)
    return parser.parse_args()
    ###############SAMPLE_RNN################
    return parser.parse_args()


###############Save and Load################
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


###############Helper Functions################
def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train')
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def create_model(args):
    # Create network.
    net = SpeechSeparation(batch_size=args.batch_size,
                           rnn_type=args.rnn_type,
                           dim=args.dim,
                           n_rnn=args.n_rnn,
                           seq_len=args.seq_len,
                           num_of_frequency_points=args.num_of_frequency_points)
    return net


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    # print("================================")
    # for name in tower_grads:
    #  for name2 in name:
    #   print(name2)
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


###############Main Functions################

def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    print("logdir", logdir)
    restore_from = directories['restore_from']

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > \
                                                      EPSILON else None
        gc_enabled = args.gc_channels is not None
        reader = AudioReader(
            args.data_dir,
            args.test_dir,
            coord,
            sample_rate=args.sample_rate,
            gc_enabled=gc_enabled,
            sample_size=args.sample_size,
            silence_threshold=silence_threshold)
        audio_batch = reader.dequeue(args.batch_size)

    net = create_model(args)
    ########Multi GPU###########
    # '''
    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
    global_step = tf.get_variable('global_step',
                  [], initializer=tf.constant_initializer(0), trainable=False)

    # Create optimizer (default is Adam)
    optim = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate, momentum=args.momentum)
    tower_grads = []
    losses = []
    speech_inputs_mix = []
    speech_inputs_1 = []
    speech_inputs_2 = []

    for i in range(args.num_gpus):
        speech_inputs_2.append(tf.Variable(
            tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
            trainable=False,
            name="speech_batch_inputs",
            dtype=tf.float32))
        speech_inputs_1.append(tf.Variable(
            tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
            trainable=False,
            name="speech_batch_inputs",
            dtype=tf.float32))
        speech_inputs_mix.append(tf.Variable(
            tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
            trainable=False,
            name="speech_batch_inputs",
            dtype=tf.float32))

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('TOWER_%d' % (i)) as scope:
                    # Create model.
                    print("Creating model On Gpu:%d." % (i))
                    loss, mask_state, output1, output2 = net.loss_SampleRnn(
                        speech_inputs_1[i],
                        speech_inputs_2[i],
                        speech_inputs_mix[i],
                        l2_regularization_strength=args.l2_regularization_strength)

                    # Reuse variables for the nect tower.
                    tf.get_variable_scope().reuse_variables()

                    # UNKNOWN
                    losses.append(loss)
                    trainable = tf.trainable_variables()
                    for name in trainable:
                        print(name)

                    gradients = optim.compute_gradients(loss, trainable)
                    # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
                    # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                    print("==========================")
                    for name in gradients:
                        print(name)
                        # Keep track of the gradients across all towers.
                    tower_grads.append(gradients)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grad_vars = average_gradients(tower_grads)

    # UNKNOWN
    grads, vars = zip(*grad_vars)
    grads_clipped, _ = tf.clip_by_global_norm(grads, 5.0)
    grad_vars = zip(grads_clipped, vars)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = optim.apply_gradients(grad_vars, global_step=global_step)

    ###################

    # Set up session
    # tf_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    tf_config = tf.ConfigProto(
        # allow_soft_placement is set to True to build towers on GPU
        allow_soft_placement=True,
        log_device_placement=False,
        inter_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
    else:
        print(" No checkpoint found.")
        sys.exit(" Your model seems to be invalid. ")

    try:
        inp_dict = {}
        global audio_test
        _, audio_test = scipy.io.wavfile.read(args.test_dir)

        X_test, X_hlf_test = stft(audio_test, fs, framesz, hop)
        amplitude_test = scipy.absolute(X_hlf_test)
        angle_test = np.angle(X_hlf_test)

        ne = np.reshape(amplitude_test, (1, amplitude_test.shape[0], amplitude_test.shape[1]))
        fd = ne[:, -256:, :]
        angle = angle_test[-256:, :]
        print(fd.shape)

        inp_dict[speech_inputs_mix[0]] = fd
        outp1, outp2 = sess.run([output1, output2], feed_dict=inp_dict)

        mk_audio(outp1, angle, "_1_test_")
        mk_audio(outp2, angle, "_2_test_")

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()


if __name__ == '__main__':
    main()
