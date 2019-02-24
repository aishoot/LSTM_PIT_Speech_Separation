import argparse
import os, sys

sys.path.append('.')
# After you read kaldi feats, you can refer to function "gen_feats" in this file to convert them to tfrecords.
import multiprocessing
from io_funcs.signal_processing import audiowrite, stft, audioread
from local.utils import mkdir_p
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser(description='Generate TFRecords files')
parser.add_argument('wav_dir', help='The parent dit of mix,s1,s2')
parser.add_argument('list_dir', help='The parent dit of mix,s1,s2')
parser.add_argument('tfrecord_dir', help='TFRecords file dir')
parser.add_argument('--gender_list', default='', type=str, help='The speekers gender list')
parser.add_argument('--sample_rate', default=8000, type=int, help='sample rate of audio')
parser.add_argument('--window_size', default=256, type=int, help='window size for STFT')
parser.add_argument('--window_shift', default=128, type=int, help='window size for STFT')


args = parser.parse_args()
wav_dir = args.wav_dir
tfrecord_dir = args.tfrecord_dir 
process_num = 8
list_dir = args.list_dir

mkdir_p(tfrecord_dir)
sample_rate = args.sample_rate
window_size = args.window_size
window_shift = args.window_shift

if args.gender_list is not '':
    apply_gender_info = True
    gender_dict = {}
    fid = open(args.gender_list, 'r')
    lines = fid.readlines()
    fid.close()
    for line in lines:
        spk = line.strip('\n').split(' ')[0]
        gender = line.strip('\n').split(' ')[1]
        if gender.lower() == 'm':
            gender_dict[spk] = 1
        else:
            gender_dict[spk] = 0


def make_sequence_example(inputs, labels, genders):
    input_features = [tf.train.Feature(float_list=tf.train.FloatList(value=input_)) for input_ in inputs]
    label_features = [tf.train.Feature(float_list=tf.train.FloatList(value=label)) for label in labels]
    gender_features = [tf.train.Feature(float_list=tf.train.FloatList(value=genders))]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features),
        'genders': tf.train.FeatureList(feature=gender_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def gen_feats(wav_name, sample_rate, window_size, window_shift):
    mix_wav_name = wav_dir + '/mix/' + wav_name
    s1_wav_name  = wav_dir + '/s1/' + wav_name
    s2_wav_name  = wav_dir + '/s2/' + wav_name

    mix_wav = audioread(mix_wav_name, offset=0.0, duration=None, sample_rate=sample_rate)
    s1_wav  = audioread(s1_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)
    s2_wav  = audioread(s2_wav_name,  offset=0.0, duration=None, sample_rate=sample_rate)

    mix_stft = stft(mix_wav, time_dim=0, size=window_size, shift=window_shift)
    s1_stft  = stft(s1_wav,  time_dim=0, size=window_size, shift=window_shift)
    s2_stft  = stft(s2_wav,  time_dim=0, size=window_size, shift=window_shift)

    s1_gender = gender_dict[wav_name.split('_')[0][0:3]]
    s2_gender = gender_dict[wav_name.split('_')[2][0:3]]

    part_name = os.path.splitext(wav_name)[0]
    tfrecords_name = tfrecord_dir + '/' + part_name + '.tfrecords'
    #print(tfrecords_name)

    with tf.python_io.TFRecordWriter(tfrecords_name) as writer:
        tf.logging.info("Writing utterance %s" %tfrecords_name)

        mix_abs = np.abs(mix_stft) 
        mix_angle = np.angle(mix_stft)

        s1_abs = np.abs(s1_stft)
        s1_angle = np.angle(s1_stft)

        s2_abs = np.abs(s2_stft)
        s2_angle = np.angle(s2_stft)

        inputs = np.concatenate((mix_abs, mix_angle), axis=1)
        labels = np.concatenate((s1_abs * np.cos(mix_angle - s1_angle), s2_abs * np.cos(mix_angle - s2_angle)), axis=1)
        gender = [s1_gender, s2_gender]

        ex = make_sequence_example(inputs, labels, gender)
        writer.write(ex.SerializeToString())


pool = multiprocessing.Pool(process_num)
workers = []
fid = open(list_dir, 'r')
lines = fid.readlines()
fid.close()

for name in lines:
    name = name.strip('\n')
    workers.append(pool.apply_async(gen_feats, (name, sample_rate, window_size, window_shift)))
    gen_feats(name, sample_rate, window_size, window_shift)

pool.close()
pool.join()
