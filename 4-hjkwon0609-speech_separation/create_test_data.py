from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import signal
import stft
import h5py

INPUT_NOISE_DIR = '../../data/raw_noise/'
INPUT_CLEAN_DIR = '../../data/sliced_clean/'
OUTPUT_DIR = '../../data/test_combined/'

CLEAN_FILE = INPUT_CLEAN_DIR + 'f10_script2_clean_113.wav'
NOISE_FILE = INPUT_NOISE_DIR + 'noise1_1.wav'


def writeWav(fn, fs, data):
	data = data * 1.5 / np.max(np.abs(data))
	wavfile.write(fn, fs, data)


if __name__ == '__main__':
	spectrogram_args = {'framelength': 512}
	rate_clean, data_clean = wavfile.read(CLEAN_FILE)
	rate_noise, data_noise = wavfile.read(NOISE_FILE)

	data_len = len(data_clean)
	data_noise = data_noise[:data_len]

	print(data_clean.dtype)
	print(data_noise.dtype)

	data_combined = np.array([s1/2 + s2/2 for (s1, s2) in zip(data_clean, data_noise)], dtype=np.int16)
	# data_combined = data_noise

	print(data_combined.dtype)

	wavfile.write('%scombined.wav' % (OUTPUT_DIR), rate_clean, data_combined)

	Sx_clean = stft.spectrogram(data_clean, **spectrogram_args)
	Sx_noise = stft.spectrogram(data_noise, **spectrogram_args)

	reverted_clean = stft.ispectrogram(Sx_clean)
	reverted_noise = stft.ispectrogram(Sx_noise)

	writeWav('%soriginal_clean.wav' % (OUTPUT_DIR), rate_clean, reverted_clean)
	writeWav('%soriginal_noise.wav' % (OUTPUT_DIR), rate_noise, reverted_noise)