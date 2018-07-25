"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

from utils import get_arguments, save, load
from utils import validate_directories, create_inputdict
from model import SpeechSeparation
from audio import AudioReader, mk_audio
import time
import logging
import numpy as np
import tensorflow as tf


def train(directories, args):  # args：所有的参数解析
    logdir = directories['logdir']
    print("logdir:", logdir)
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into a location that's different from logdir.
    is_overwritten_training = logdir != restore_from  # 先算后部分,返回bool值. 结果:False,意思是两个文件夹相同
    
    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    # create inputs
    gc_enabled = args.gc_channels is not None
    reader = AudioReader(args.data_dir, args.test_dir, coord,
        sample_rate=args.sample_rate, gc_enabled=gc_enabled)
    audio_batch = reader.dequeue(args.batch_size)

    # Initialize model
    net = SpeechSeparation(batch_size=args.batch_size,
        rnn_type=args.rnn_type, dim=args.dim, n_rnn=args.n_rnn,
        seq_len=args.seq_len, num_of_frequency_points=args.num_of_frequency_points)

    # need to modify net to include these
    #out =
    summary, output1, output2, losses, apply_gradient_op = net.initializer(net, args)  # output1/2:(1,256,257)
    speech_inputs_1 = net.speech_inputs_1  # (1,256,257)
    speech_inputs_2 = net.speech_inputs_2  # (1,256,257)
    speech_inputs_mix = net.speech_inputs_mix  # (1,256,257)

    # Set up session
    tf_config = tf.ConfigProto(
        # allow_soft_placement is set to True to build towers on GPU
        allow_soft_placement=True, log_device_placement=False,
        inter_op_parallelism_threads = 1)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config = tf_config)

    sess.run(tf.global_variables_initializer())
    # Create coordinator.
    
    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    run_metadata = tf.RunMetadata() # 定义TensorFlow运行元信息,记录训练运算时间和内存占用等信息

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    try:
        saved_global_step = load(saver, sess, restore_from)  # 第一次结果:None
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. We will terminate "
              "training to avoid accidentally overwriting the previous model.")
        raise

#################################Start Training####################################
    last_saved_step = saved_global_step  # -1
    try:
        for step in range(saved_global_step + 1, args.num_steps):  # (0, 1000000)
            loss_sum = 0
            start_time = time.time()

            inputslist = [sess.run(audio_batch) for i in range(args.num_gpus)]  # len:1, 里边还有四维,每一维(1,697,257),括号中1的个数指gpu个数
            inp_dict = create_inputdict(inputslist, args, speech_inputs_1,
                                        speech_inputs_2, speech_inputs_mix)
            summ, loss_value, _= sess.run([summary, losses, apply_gradient_op], feed_dict = inp_dict) #feed_dict前一个数是占位符,后一个是真实值

            for g in range(args.num_gpus):
                loss_sum += loss_value[g] / args.num_gpus
            
            writer.add_summary(summ, step)
            duration = time.time() - start_time

            if (step < 100):
                log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step')%(step, loss_sum, duration)
                logging.warning(log_str)

            elif (0==step % 100):
                log_str = ('step {%d} - loss = {%0.3f}, ({%0.3f} sec/step')%(step, loss_sum/100, duration)
                logging.warning(log_str)

            if (0==step % 2000):
                angle_test, inp_dict = create_inputdict(inputslist, args, 
                    speech_inputs_1, speech_inputs_2, speech_inputs_mix, test=True)

                outp1, outp2 = sess.run([output1,output2], feed_dict=inp_dict)
                x_r = mk_audio(outp1, angle_test, args.sample_rate, "spk1_test_"+str(step)+".wav")
                y_r = mk_audio(outp2, angle_test, args.sample_rate, "spk2_test_"+str(step)+".wav")

                amplitude_test = inputslist[0][2]
                angle_test = inputslist[0][3]              
                mk_audio(amplitude_test, angle_test, args.sample_rate, "raw_test_"+str(step)+".wav")

                # audio summary on tensorboard
                merged = sess.run(tf.summary.merge(
                    [tf.summary.audio('speaker1_' + str(step), x_r[None, :],
                        args.sample_rate, max_outputs=1),
                     tf.summary.audio('speaker2_' + str(step), y_r[None, :], 
                        args.sample_rate, max_outputs=1)]
                ))
                writer.add_summary(merged, step)

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message is on its own line.
        print()
    finally:
        #'''
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        #'''
        coord.request_stop()
        coord.join(threads)


def main():
    args = get_arguments()
    try:
        # directories返回:{'logdir':logdir, 'restore_from':restore_from}
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None  # ???默认为0
    train(directories, args)  # 此时data_dir = test_dir

    return

##############################################################
if __name__ == '__main__':
    main()
