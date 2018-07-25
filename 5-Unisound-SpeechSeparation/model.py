import tensorflow as tf
from tensorflow.python.ops import math_ops
from ops import optimizer_factory
from utils import average_gradients


class SpeechSeparation(object):
    """
    tf.variable_scope可以让变量有相同的命名(前一部分)，包括tf.get_variable得到的变量，还有tf.Variable的变量
    tf.name_scope可以让变量有相同的命名(前一部分)，只是限于tf.Variable的变量
    输出变量名格式：V1/a1, V2/a2:0
    """
    def _create_network_speechrnn(self, speech_inputs_mix):
        with tf.variable_scope('SEEPCH_RNN_LAYER'):
            speech_outputs = []
            mlp1_weights = tf.get_variable("mlp1", [self.dim, self.dim], dtype=tf.float32)
            mlp2_weights = tf.get_variable("mlp2", [self.dim, self.dim], dtype=tf.float32)
            mlp3_weights = tf.get_variable("mlp3", [self.dim, self.num_of_frequency_points*2],dtype=tf.float32)

        with tf.variable_scope("SEEPCH_RNN"):
            input_list = tf.unstack(tf.transpose(speech_inputs_mix, perm=[1, 0, 2]), axis=0)
            fb_output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(self.f_cell, self.b_cell,
                            input_list, dtype=tf.float32, scope='bi_rnn')
            for speech_cell_output in fb_output: 
                out = math_ops.matmul(speech_cell_output, mlp1_weights)
                out = tf.nn.relu(out)
                out = math_ops.matmul(out, mlp2_weights)
                out = tf.nn.relu(out)
                out = math_ops.matmul(out, mlp3_weights)
                out = tf.nn.relu(out)
                speech_outputs.append(out)

            final_speech_outputs = tf.stack(speech_outputs)
            final_speech_outputs = tf.transpose(final_speech_outputs, perm=[1, 0, 2])

        return final_speech_outputs


    def loss_SampleRnn(self, speech_inputs_1, speech_inputs_2, speech_inputs_mix,
                       l2_regularization_strength=None):
        mask_num_steps = 256
        mask_outputs = self._create_network_speechrnn(speech_inputs_mix)
        mask_1, mask_2 = tf.split(mask_outputs, 2, 2)  # 从mask_outputs第2维平均切割成两份
        output1 = speech_inputs_mix * mask_1
        output2 = speech_inputs_mix * mask_2
        tmp = output1

        with tf.name_scope('loss'):
            #mask_num_steps = self.num_of_frequency_points-1
            # only take 1 frame
            mask_num_steps = 256
            target_1     = tf.reshape(speech_inputs_1, [self.batch_size*mask_num_steps, -1])
            target_2     = tf.reshape(speech_inputs_2, [self.batch_size*mask_num_steps, -1])
            prediction_1 = tf.reshape(output1, [self.batch_size*mask_num_steps, -1])
            prediction_2 = tf.reshape(output2, [self.batch_size*mask_num_steps, -1])

            loss_1 = tf.losses.mean_squared_error(labels = target_1, predictions = prediction_1)
            loss_2 = tf.losses.mean_squared_error(labels = target_2, predictions = prediction_2)

            loss_3 = tf.losses.mean_squared_error(labels = target_2, predictions = prediction_1)
            loss_4 = tf.losses.mean_squared_error(labels = target_1, predictions = prediction_2)
             
            reduced_loss_1 = tf.reduce_mean(loss_1)
            reduced_loss_2 = tf.reduce_mean(loss_2)
            reduced_loss = reduced_loss_1 + reduced_loss_2
            #reduced_loss_a =reduced_loss_1+reduced_loss_2

            #reduced_loss_3 = tf.reduce_mean(loss_3)
            #reduced_loss_4 = tf.reduce_mean(loss_4)
            #reduced_loss_b =reduced_loss_3+reduced_loss_4

            #reduced_loss = tf.cond(tf.less(reduced_loss_b, reduced_loss_a), 
            #lambda: reduced_loss_b, lambda: reduced_loss_a)  
            #output1 = tf.cond(tf.less(reduced_loss_b, reduced_loss_a), 
            #lambda: output2, lambda: output1)  
            #output2 = tf.cond(tf.less(reduced_loss_b, reduced_loss_a), 
            #lambda: tmp, lambda: output2) 
            #reduced_loss = tf.Print(reduced_loss,[reduced_loss,reduced_loss_a,reduced_loss_b],
            #  message="The losses are:")

            if l2_regularization_strength is None:
                summary = tf.summary.scalar('loss', reduced_loss)
                return summary, reduced_loss , output1,output2
            else:
                # L2 regularization for all trainable parameters
                l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                    for v in tf.trainable_variables()
                                    if not('bias' in v.name)])

                # Add the regularization term to the loss
                total_loss = (reduced_loss + l2_regularization_strength * l2_loss)
                summary = tf.summary.scalar('loss', total_loss)

                return summary, total_loss, output1, output2


    def __init__(self, batch_size, rnn_type, dim, n_rnn, seq_len, num_of_frequency_points):
        self.batch_size = batch_size
        self.rnn_type = rnn_type
        self.dim = dim
        self.n_rnn = n_rnn
        self.seq_len = seq_len
        self.num_of_frequency_points = num_of_frequency_points
        self.merged = tf.summary.merge_all()
 
        def single_cell():
            if 'LSTM' == self.rnn_type:
                return tf.contrib.rnn.BasicLSTMCell(self.dim/2)  # LSTM高级变种,括号中参数(一个Cell中神经元的个数)
            else:
                return tf.contrib.rnn.GRUCell(self.dim/2)
        self.cell = single_cell()
        self.f_cell = single_cell()
        self.b_cell = single_cell()

        if self.n_rnn > 1:
            print("add rnn layer", self.n_rnn)
            self.cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
            self.f_cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
            self.b_cell = tf.contrib.rnn.MultiRNNCell(
                 [single_cell() for _ in range(self.n_rnn)])
        self.speech_inputs_1 = []
        self.speech_inputs_2 = []
        self.speech_inputs_mix = []


    def initializer(self, net, args):
        # Create optimizer (default is Adam)
        optim = optimizer_factory[args.optimizer](
                learning_rate=args.learning_rate, momentum=args.momentum)
        # Create a variable to count the number of steps. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable('global_step', [],
                    initializer = tf.constant_initializer(0), trainable=False)
        tower_grads = []
        losses = []

        for i in range(args.num_gpus):
            self.speech_inputs_2.append(tf.Variable(
                tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
                trainable=False, name="speech_2_batch_inputs", dtype=tf.float32))
            self.speech_inputs_1.append(tf.Variable(
                tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
                trainable=False, name="speech_1_batch_inputs", dtype=tf.float32))
            self.speech_inputs_mix.append(tf.Variable(
                tf.zeros([net.batch_size, net.seq_len, args.num_of_frequency_points]),
                trainable=False, name="speech_mix_batch_inputs", dtype=tf.float32))

        # Calculate the gradients for each model tower.
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(args.num_gpus):
                with tf.device('/gpu:%d'%i):
                    with tf.name_scope('TOWER_%d' % (i)) as scope:
                        # Create model.
                        print("Creating model On GPU:%d." % (i))
                        summary, loss, output1, output2 = net.loss_SampleRnn(
                            self.speech_inputs_1[i],
                            self.speech_inputs_2[i],
                            self.speech_inputs_mix[i],
                            l2_regularization_strength=args.l2_regularization_strength)
              
                        # Reuse variables for the nect tower.
                        tf.get_variable_scope().reuse_variables()
                        losses.append(loss)
                        trainable = tf.trainable_variables()
                        for name in trainable:
                            print(name)

                        # Calculate the gradients for the batch of data on this tower.
                        gradients = optim.compute_gradients(loss,trainable)
                        print("==========================================")
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

        return (summary, output1, output2,
            # speech_inputs_1,speech_inputs_2,speech_inputs_mix,
            losses, apply_gradient_op)
