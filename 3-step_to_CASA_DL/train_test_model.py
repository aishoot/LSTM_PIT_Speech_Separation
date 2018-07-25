import re
import glob
import pickle
import numpy as np

import scipy.io as sio
from scipy.io import wavfile as swav

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import load_model

import speech_preprocessing


def train_model(_signal_class, _train_data_path, _target_mask, _model_file_name, _gender=None,
                _noise=None, _scale=None):
    """
    Trains a Neural Network using Keras and stores the model to a file
    :param _signal_class: instance of speech_processing.Signal class for associated functions
    :param _train_data_path: Path with training files. Each pickle file has a dictionary with keys X, S, N
                            Each key stores a list of spectrograms, where columns are the features
    :param _target_mask: The mask which we try to learn IBM/IRM
    :param _model_file_name: The path+name of the file to which the learnt model has to be saved
    :param _gender: Filter to load files with specific genders
    :param _noise: Filter to load files with specific noises
    :param _scale: Filter to load files with specific scales
    """
    print('Loading Spectrograms to memory')
    _X_train_complex, _Y_train_complex, _ = _signal_class.read_processed_data(_train_data_path, _target_mask,
                                                                              _gender, _noise, _scale)
    # Absolute and transpose
    _X_train = np.absolute(_X_train_complex).T
    _Y_train = np.absolute(_Y_train_complex).T

    _model = Sequential()
    _model.add(Dense(1024, activation='relu', input_dim=513))
    _model.add(Dropout(0.2))
    _model.add(Dense(1024, activation='relu', input_dim=513))
    _model.add(Dropout(0.2))
    _model.add(Dense(1024, activation='relu', input_dim=513))
    _model.add(Dropout(0.2))
    _model.add(Dense(513, activation='sigmoid'))

    _sgd = SGD(lr=0.01, decay=0, momentum=0.9, nesterov=False)
    _model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=_sgd)
    _model.fit(_X_train, _Y_train, epochs=500, batch_size=100, shuffle=True, verbose=2)

    print(_model.output)
    _model.save(_model_file_name)

    print('Training complete!')


def test_model(_signal_class, _test_data_path, _model_file_name, _target_mask, _output_wavefile_path,
               _snr_path, _gender, _noise, _scale):
    """
    :param _signal_class: instance of speech_processing.Signal class for associated functions 
    :param _test_data_path: Path with test files. Each pickle file has a dictionary with keys X, S, N
                            Each key stores a list of spectrograms, where columns are the features
    :param _model_file_name: The path+name of the file in which the model is saved
    :param _target_mask: The mask which we try to predict IBM/IRM
    :param _output_wavefile_path: Path to which the regenerated files are written 
    :param _snr_path: Path to which a file with SNR values is written
    :param _gender: Filter to load files with specific genders
    :param _noise: Filter to load files with specific noises
    :param _scale: Filter to load files with specific scales
    :return: 
    """
    model = load_model(_model_file_name)
    _test_file_names = speech_preprocessing.get_filtered_file_names_list(glob.glob(_test_data_path + '*'),
                                                                         _gender=_gender, _noise=_noise, _scale=_scale)
    _test_file_names_len = len(_test_file_names)

    f_num = 1
    snr_results_dict = {}
    for fname in _test_file_names:
        snr_results_list = []
        print('Processing file:', f_num, 'of', _test_file_names_len, ' name:', fname)
        fname_split = re.split('_', re.split('/', fname)[2])
        f_num += 1

        data_dict = pickle.load(open(fname, 'rb'))
        X_test_complex_list = data_dict['X']
        S_test_complex_list = data_dict['S']

        for i in range(len(X_test_complex_list)):
            X_test_complex = X_test_complex_list[i]
            S_test_complex = S_test_complex_list[i]

            Y_predicted = model.predict(np.absolute(X_test_complex).T)

            s_wav_filename = _output_wavefile_path + '_'.join(fname_split) + '_' + str(i)

            if _target_mask == 'IBM':
                S_predicted = np.multiply(X_test_complex, np.round(Y_predicted.T))
            elif _target_mask == 'IRM':
                S_predicted = np.multiply(X_test_complex, Y_predicted.T)
            else:
                print("Incorrect target name. Should be 'IRM' or 'IBM' ")
                return ValueError

            s_regenerated = signal.get_wave(np.absolute(S_predicted))
            s_regenerated = s_regenerated / s_regenerated.var()
            swav.write(data=s_regenerated, rate=16000, filename=s_wav_filename + '_pred.wav')

            s_original = signal.get_wave(np.absolute(S_test_complex))
            swav.write(data=s_original, rate=16000, filename=s_wav_filename + '.wav')

            snr_results_list.append(signal.get_snr(s_original, s_regenerated))

            results_dict_matlab = {'s_regenerated': s_regenerated, 's_original': s_original}
            sio.savemat(s_wav_filename + '.mat', results_dict_matlab)

        snr_results_dict['_'.join(fname_split)] = snr_results_list

    pickle.dump(snr_results_dict, open(_snr_path, "wb"))

    # snr_output = pickle.load(open(_snr_path, 'rb'))
    # print(snr_output.keys())

    print('Testing complete!')


if __name__ == '__main__':
    # Binary switch to select test or train mode
    # Once a mode is selected, give correct properties in the branch to execute smoothly
    run_in_train_mode = False
    np.random.seed(345)

    # instantiating signal class
    print('Instantiating Signal class')
    signal = speech_preprocessing.Signal(_frame_size=1024)

    if run_in_train_mode:
        # Initializing training parameters
        print('Initializing Training parameters')

        train_data_path = 'data/train/'
        train_model_file_name = 'trained_models/irm_female_speech_noise_500_iter_100_batch_0.1_lr.h5'
        train_target_mask = 'IRM'

        # training model filters - list of strings is the data type for each filter
        train_gender_filter = ['female']  # possible values - 'male', 'gender'
        train_noise_filter = []  # possible values - 'birds', 'computerkeyboard', 'jungle', 'motorcycles', 'ocean'
        train_scale_filter = []  # possible values - '0dB', 'neg5dB', 'pos5dB'

        # Training Data
        print('Training initiated')
        train_model(_signal_class=signal, _train_data_path=train_data_path, _target_mask=train_target_mask,
                    _model_file_name=train_model_file_name, _gender=train_gender_filter,
                    _noise=train_noise_filter, _scale=train_scale_filter)
    else:
        # Initializing Test parameters
        print('Initializing Test parameters')
        test_data_path = 'data/test/'
        test_model_file_name = 'trained_models/irm_female_speech_noise_500_iter_100_batch_0.1_lr.h5'
        test_target_mask = 'IRM'
        test_output_wave_file_path = 'wavefiles/irm_all_noise_female_gender_male_test/'
        test_snr_path = 'results/IRM_SNR_results_all_noise_female_gender_male_test.pkl'

        # testing model filters - list of strings is the data type for each filter
        test_gender_filter = ['male']  # possible values - 'male', 'gender'
        test_noise_filter = []  # possible values - 'birds', 'computerkeyboard', 'jungle', 'motorcycles', 'ocean'
        test_scale_filter = []  # possible values - '0dB', 'neg5dB', 'pos5dB'

        print('Testing initiated')
        test_model(_signal_class=signal, _test_data_path=test_data_path, _model_file_name=test_model_file_name,
                   _target_mask=test_target_mask, _output_wavefile_path=test_output_wave_file_path,
                   _snr_path=test_snr_path, _gender=test_gender_filter, _noise=test_noise_filter,
                   _scale=test_scale_filter)
