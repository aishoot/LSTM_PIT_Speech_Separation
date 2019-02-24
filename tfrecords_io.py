#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2017, Sining Sun (NPU)

# Function in his file is used to make mini_batch for LSTM training from tfrecords files during LSTM training.
"""Utility functions for working with tf.train.SequenceExamples."""

import tensorflow as tf


def get_padded_batch(file_list, batch_size, input_size, output_size,
                     num_enqueuing_threads=4, num_epochs=1, shuffle=True):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    # file_list:(20000,), batch_size:25, input_size/output_size:258
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.一个list,包含tfrecords所有的文件地址
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        num_enqueuing_threads: The number of threads to use for enqueuing SequenceExamples.
    Returns:
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=shuffle) 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
        'labels': tf.FixedLenSequenceFeature(shape=[output_size], dtype=tf.float32),
        'genders': tf.FixedLenSequenceFeature(shape=[2], dtype=tf.float32)}

    _, sequence = tf.parse_single_sequence_example(serialized_example, sequence_features=sequence_features)
    length = tf.shape(sequence['inputs'])[0]
    capacity = 1000 + (num_enqueuing_threads + 1) * batch_size  # 1325
    queue = tf.PaddingFIFOQueue(
        capacity=capacity,
        dtypes=[tf.float32, tf.float32, tf.float32, tf.int32],
        shapes=[(None, input_size), (None, output_size), (1, 2), ()])

    enqueue_ops = [queue.enqueue([sequence['inputs'],
                                  sequence['labels'],
                                  sequence['genders'],
                                  length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    return queue.dequeue_many(batch_size)


def get_padded_batch_v2(file_list, batch_size, input_size, output_size,
                        num_enqueuing_threads=4, num_epochs=1, shuffle=True):
    """Reads batches of SequenceExamples from TFRecords and pads them.
    Can deal with variable length SequenceExamples by padding each batch to the
    length of the longest sequence with zeros.
    Args:
        file_list: A list of paths to TFRecord files containing SequenceExamples.
        batch_size: The number of SequenceExamples to include in each batch.
        input_size: The size of each input vector. The returned batch of inputs
            will have a shape [batch_size, num_steps, input_size].
        num_enqueuing_threads: The number of threads to use for enqueuing
            SequenceExamples.
    Returns:
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
        labels: A tensor of shape [batch_size, num_steps] of float32s.
        lengths: A tensor of shape [batch_size] of int32s. The lengths of each
            SequenceExample before padding.
    """
    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=num_epochs, shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)

    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
        'inputs_cmvn': tf.FixedLenSequenceFeature(shape=[input_size], dtype=tf.float32),
        'labels1': tf.FixedLenSequenceFeature(shape=[output_size], dtype=tf.float32),
        'labels2': tf.FixedLenSequenceFeature(shape=[output_size], dtype=tf.float32),
    }

    _, sequence = tf.parse_single_sequence_example(
        serialized_example, sequence_features=sequence_features)

    length = tf.shape(sequence['inputs'])[0]

    capacity = 1000 + (num_enqueuing_threads + 1) * batch_size
    queue = tf.PaddingFIFOQueue(
        capacity=capacity,
        dtypes=[tf.float32, tf.float32, tf.float32, tf.float32, tf.int32],
        shapes=[(None, input_size), (None, input_size), (None, output_size), (None, output_size), ()])

    enqueue_ops = [queue.enqueue([sequence['inputs'],
                                  sequence['inputs_cmvn'],
                                  sequence['labels1'],
                                  sequence['labels2'],
                                  length])] * num_enqueuing_threads

    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))
    return queue.dequeue_many(batch_size)
