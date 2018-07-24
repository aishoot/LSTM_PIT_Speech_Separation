from __future__ import print_function
import argparse
from ops import optimizer_factory
import os
import sys
import argparse
import tensorflow as tf
import logging

# hyper parameters
NUM_OF_FREQUENCY_POINTS = 257
BATCH_SIZE = 1
DATA_DIRECTORY = 'VCTK-Corpus/'   ## 修改
#DATA_DIRECTORY = 'mix'
SAMPLE_RATE = 48000
CHECKPOINT_EVERY = 2000
NUM_STEPS = 200  # 训练步数,原:int(1e6)
LEARNING_RATE = 1e-4
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.3
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 10
N_SEQS = 10  # Number of samples to generate every time monitoring.
NUM_GPU = 1

# 新增参数
RNN_TYPE = "LSTM"
DIM = 896
N_RNN = 1
SEQ_LEN = 256
LOGDIR = "afterlog"


def get_arguments():
    def _str_to_bool(s):
        # 将字符串"true"和"false"转化为bool类型
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='PIT example network')
    parser.add_argument('--num_gpus', type=int, default=NUM_GPU,
                        help='num of gpus.. Default: ' + str(NUM_GPU) + '.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--num_of_frequency_points', type=int, default=NUM_OF_FREQUENCY_POINTS,
                        help='num_of_frequency_points. Default: ' + str(NUM_OF_FREQUENCY_POINTS) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--test_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--logdir', type = str, default = LOGDIR,
                        help='Directory in which to store the logging information for TensorBoard. '
                        'If the model already exists, it will restore the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                        'This creates the new model under the dated directory in --logdir_root. '
                        'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_rate', type=int, default=SAMPLE_RATE,
                        help='sample rate for training. Default: ' + str(SAMPLE_RATE) + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many samples. Default: '+str(SAMPLE_SIZE)+'.')
    parser.add_argument('--l2_regularization_strength', type=float, default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. Default: False')
    parser.add_argument('--silence_threshold', type=float, default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Specify the momentum to be '
                        'used by sgd or rmsprop optimizer. Ignored by the adam optimizer. Default: '+str(MOMENTUM)+'.')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '+str(MAX_TO_KEEP)+'.')

    def t_or_f(arg):
        # 确定输入的字符串arg是"true"还是"false"(不区分大小写)
        ua = str(arg).upper()
        if 'TRUE'.startswith(ua):
            return True
        elif 'FALSE'.startswith(ua):
            return False
        else:
           raise ValueError('Arg is neither `True` nor `False`')

    def check_non_negative(value):
        # 返回输入的value的整型值,如果其为负数则提示
        ivalue = int(value)
        if ivalue < 0:
             raise argparse.ArgumentTypeError("%s is not non-negative!" %value)
        return ivalue

    def check_positive(value):
        # 返回原数的整型值,其小于1会提示
        ivalue = int(value)
        if ivalue < 1:
             raise argparse.ArgumentTypeError("%s is not positive!" %value)
        return ivalue

    def check_unit_interval(value):
        # 提示输入的数不在[0, 1]之间
        fvalue = float(value)
        if fvalue < 0 or fvalue > 1:
             raise argparse.ArgumentTypeError("%s is not in [0, 1] interval!" %value)
        return fvalue

    parser.add_argument('--seq_len', help='How many samples to include in each Truncated BPTT pass',
                        type=check_positive,  default = SEQ_LEN)  # Truncated:截断的
    parser.add_argument('--rnn_type', help='GRU or LSTM', choices=['LSTM', 'GRU'], default=RNN_TYPE)
    parser.add_argument('--dim', help='Dimension of RNN and MLPs', default=DIM, type=check_positive)
    parser.add_argument('--n_rnn', help='Number of layers in the stacked RNN', default = N_RNN,
                        type=check_positive, choices=range(1,6))  # required=True该参数只能从python脚本处输入
    return parser.parse_args()


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
    print("Trying to restore saved checkpoints from {} ...".format(logdir), end="")
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def validate_directories(args):
    """验证和整理文件目录相关的参数."""
    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last checkpoint.")
    logdir = args.logdir

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive(专用的;排外的;单独的),
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {'logdir':logdir, 'restore_from':restore_from}


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


def create_inputdict(inputslist,args,speech_1,speech_2,speech_mix,test=False):
    inp_dict={} # 此时speech_1&2, speech_mix均是(1,256,257)

    s_len = inputslist[0][0].shape[1] // 3  # 源程序是"/", 结果:593//3 == 197
    seq_len = args.seq_len  # 256

    # only take seq_len of a single speaker

    if test:
        inp_dict[speech_1[0]] = inputslist[0][2][:,:seq_len,:]
        inp_dict[speech_2[0]] = inputslist[0][2][:,s_len:s_len+seq_len,:]
        inp_dict[speech_mix[0]] = inputslist[0][2][:,-s_len:-s_len+seq_len,:]
        angle_test= inputslist[0][3][:,-s_len:-s_len+seq_len,:]
        return (angle_test,inp_dict)
    else:
        if(seq_len > s_len):
            logging.error("args.seq_len %d > s_len %d", seq_len, s_len)
        for g in range(args.num_gpus):  # g:0
            inp_dict[speech_1[g]]  =inputslist[g][0][:,:seq_len,:]   # seq_len:256
            inp_dict[speech_2[g]]  =inputslist[g][0][:,s_len:s_len+seq_len,:]
            #inp_dict[speech_mix[g]]=inputslist[g][0][:,-s_len:-s_len+seq_len,:]  # 原来的代码
            inp_dict[speech_mix[g]] = inputslist[g][0][:, -seq_len:, :]
            print("inputslist[g][0].shape:", inputslist[g][0].shape)
            print("s_len+seq_len:", s_len+seq_len)

        return inp_dict
