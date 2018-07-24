"""
功能:测试混合音频.给定两段音频,每次都随机取5s混合后和原5s音频一起写入音频文件
"""
import librosa
import fnmatch
import os
import re
import numpy as np
import scipy
from numpy import random


def trim_silence(audio, threshold):
    '''Removes silence at the beginning and end of a sample.'''
    energy = librosa.feature.rmse(audio)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    #print (files)
    return files


def mix(audio_1, audio_2, sr, sav_n_secs, path_out, index):
    snr = random.randint(5, 10)
    weight = np.power(10, (snr/20))  # np.power(x1,x2)求x1关于x2对应数次方, (1.778, 3.162)

    file_mix="%07d_%02d.wav"%(index, snr)
    audio_1 = weight * audio_1
    audio_2 = audio_2
    audio_mix = (audio_1 + audio_2)/2

    outfile = os.path.join(path_out, file_mix)
    librosa.output.write_wav(outfile, np.concatenate((audio_1, audio_2, audio_mix), axis=0), sr)


def main():
    # 文件目录及参数设置
    outdir = 'mix'
    sample_rate = 16000   # VCTK: 32-bit float, 48kHz
    sav_n_secs = 5
    train_data_num = 12000

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    audio_total1, sr = librosa.load('./bird.wav', sr=sample_rate, mono=True)
    audio_total2, sr = librosa.load('./xiaodu.wav', sr=sample_rate, mono=True)

    seglen = int(sav_n_secs * sr)  # 5s采样点数

    len1 = audio_total1.shape[0] - seglen  # 除去5s后的长度
    len2 = audio_total2.shape[0] - seglen

    for i in range(train_data_num):
        if i % 100 == 0:
            print(i)
        idx1 = random.randint(0, len1)
        idx2 = random.randint(0, len2)
        mix(audio_total1[idx1:idx1+seglen], audio_total2[idx2:idx2+seglen], sample_rate, sav_n_secs, outdir, i)

if __name__ == '__main__':
    main()
