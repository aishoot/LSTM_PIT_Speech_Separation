import numpy as np
import scipy.io.wavfile as sio
import glob
import re
import pickle
"""
思路简述:根据噪声数量,从原音频文件中随机选择num_speeches(10)个分别与噪声相等相加,得到10条记录
"""


def get_filtered_file_names_list(_file_names_list, _gender=None, _noise=None, _scale=None):
    """
    Expecting each filename to be in folder1/folder2/gender/noise/scale format
    :param _file_names_list: List of filenames to be filtered
    :param _gender: Filter to load files with specific genders
    :param _noise: Filter to load files with specific noises
    :param _scale: Filter to load files with specific scales
    :return: Filenames that match the filters
    """
    _file_names_split_list = [re.split('[/_]+', fname) for fname in _file_names_list]

    if _gender:
        if type(_gender) == str:
            _gender = [_gender]
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-3] in _gender]

    if _noise:
        if type(_noise) == str:
            _noise = [_noise]
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-2] in _noise]

    if _scale:
        if type(_scale) == str:
            _scale = [_scale]
        _file_names_split_list = [f_name for f_name in _file_names_split_list if f_name[-1] in _scale]

    _file_names_list = ['_'.join(['/'.join(fname_split[:3]), fname_split[-2], fname_split[-1]])
                        for fname_split in _file_names_split_list]

    return _file_names_list


class Signal:
    def __init__(self, _frame_size):
        self.N = _frame_size
        self.F = self.generate_dft_coeffs

    @property
    def generate_dft_coeffs(self):
        """
        Funtion to Generate complex Fourier Coefficients
        :param N: frame size
        :return: Complex fourier coefficients
        """
        N = self.N
        F = np.zeros((N, N), dtype=complex)
        # F_cos = np.zeros((N, N))
        # F_sin = np.zeros((N, N))

        for f in range(N):
            for n in range(N):
                F[f, n] = np.cos(2 * np.pi * f * n / N) + (np.sin(2 * np.pi * f * n / N)) * 1j
                # F_cos[f, n] = np.cos(2 * np.pi * f * n / N)
                # F_sin[f, n] = np.sin(2 * np.pi * f * n / N)
        # return F_cos + 1j * F_sin
        return F

    def hann(self, _x):
        """
        Defining generic hann window
        :param _x: 
        :return: 
        """
        N = self.N
        return (1 - np.cos(2 * np.pi * _x / (N - 1))) / 2

    def generate_data_samples(self, _data):
        """
        NOTE: Overlapping factor is hardcoded to 0.5
        Segmentation of sample stream to blocks of size N
        and overalap between segments is N/2
        :param _data: Time series data 
        :return: Segment data into overlapping frames and return a matrix with each column corresponding to a segment 
        """
        N = self.N
        data_len = _data.shape[0]

        X = [_data[i:i + N] for i in range(0, data_len - N // 2, N // 2)]
        if len(X[-1]) != N:
            X[-1] = np.append(X[-1], np.zeros(N - X[-1].shape[0]))

        return np.array(X).transpose()

    def get_spectrogram(self, _data, _reduced=True):
        """
        Default window is hann. Multiple windows customization can be done later usign a function parameter
        
        :param _data: time series data
        :param _reduced: Boolean value to denote flag if a half spectrogram is passed
        :return: Complex spectrogram
        """
        fft_coeffs = self.F
        N = fft_coeffs.shape[0]
        X_raw = self.generate_data_samples(_data)
        hann_win = self.hann(np.array(range(N), ndmin=2))

        X = np.multiply(X_raw, hann_win.T)
        spectrogram = np.dot(fft_coeffs, X)

        if _reduced:
            # [:513] if N = 1024
            return spectrogram[:(N // 2 + 1)]
        else:
            return spectrogram

    @staticmethod
    def reconstruct_signal(_X):
        """
        Reconstruct signal from segmented matrix
        :param _X: segmentized matrix
        :return: reconstructed time series signal
        """
        width = _X.shape[1]
        N = _X.shape[0]
        n = N // 2

        head = _X[:n, 0]
        tail = _X[n:, width - 1]
        body = np.array([_X[n:, i] + _X[:n, i + 1] for i in range(width - 1)]).reshape(n * (width - 1))

        return np.append(head, np.append(body, tail))

    def get_wave(self, _spectrogram, _is_full_spectrogram=False):
        """
        Takes full or upper half of the spectrogram and returns reconstructed time series signal
        :param _spectrogram: spectrogram
        :param _is_full_spectrogram: default False, if true, the full spectrogram is used
        :return: time series signal
        """
        # in half spectrogram, 1st row is not in mirror image
        # last line row is same, so, we flip 2nd to last but one rows

        fft_coeffs = self.F
        if not _is_full_spectrogram:
            # generating full spectrogram from the upper half if 'is_full_spectrogram' is False
            full_spectrogram = np.vstack((_spectrogram[0],
                                          _spectrogram[1:-1],
                                          _spectrogram[-1],
                                          np.flipud(_spectrogram[1:-1]).conjugate()))
        else:
            full_spectrogram = _spectrogram

        # Doing Fourier inverse to get segmented matrix
        signal_segment_matrix = np.dot(fft_coeffs.conjugate().T, full_spectrogram).real
        # Recovering timeseries signal
        signal = self.reconstruct_signal(signal_segment_matrix)
        # Rescaling the signal
        # signal = (signal - signal.min()) / (signal.max() - signal.min())
        # signal = signal / signal.var()

        # returning signal
        return signal

    @staticmethod
    def get_stack_spectrogram(_spectrogram_list):
        stacked_spectrogram = _spectrogram_list[0]
        for i in range(len(_spectrogram_list) - 1):
            stacked_spectrogram = np.hstack((stacked_spectrogram, _spectrogram_list[i + 1]))

        return stacked_spectrogram

    def read_processed_data(self, _folder_path, _target='IBM', _gender=None, _noise=None, _scale=None):
        # validating target
        if _target not in ['IBM', 'IRM', 'FFT']:
            print('Enter valid target keyword')
            return

        # generic spectrogram shape for any fame size
        num_features = (self.N // 2 + 1)
        # initializing output data
        stacked_dict = {'X': np.empty((num_features, 0)),
                        'S': np.empty((num_features, 0)),
                        'N': np.empty((num_features, 0))}
        _Y = None

        # Implement gender, noise and scale filters
        file_names_list = get_filtered_file_names_list(glob.glob(_folder_path + '*'),
                                                       _gender=_gender, _noise=_noise, _scale=_scale)

        # iterate over each file, read, each file has a dict of data, append data to S, N, X
        file_names_len = len(file_names_list)
        i = 1
        for fname in file_names_list:
            print('Processing file', i, 'of', file_names_len)
            i += 1
            data_dict = pickle.load(open(fname, 'rb'))

            for key in ['X', 'S', 'N']:
                stacked_dict[key] = np.hstack((stacked_dict[key], self.get_stack_spectrogram(data_dict[key])))

        # Creating target
        if _target == 'IBM':
            _Y = (np.absolute(stacked_dict['S']) > np.absolute(stacked_dict['N'])).astype(int)
        elif _target == 'IRM':
            _S_square = np.square(np.absolute(stacked_dict['S']))
            _N_square = np.square(np.absolute(stacked_dict['N']))
            _Y = np.sqrt(np.divide(_S_square, (_S_square + _N_square)))
        elif _target == 'FFT':
            _Y = np.absolute(stacked_dict['S'])

        # return features, target, original signal spectrogram
        return stacked_dict['X'], _Y, stacked_dict['S']

    @staticmethod
    def get_snr(original_signal, noisy_signal):
        noise = noisy_signal - original_signal
        return 10 * np.log10(original_signal.var() / noise.var())


if __name__ == '__main__':
    # setting seed
    np.random.seed(122)
    print('Initial set up started.')

    # Setting Paths
    input_data_path = '/mnt/hd8t/pchao/TIMIT2/'  # 数据库根目录
    input_noise_path = input_data_path + 'NOISE/'  # 噪声数据存放地址,下面全是.wav文件

    input_male_train_path = input_data_path + 'TRAIN/*/M*/'
    input_female_train_path = input_data_path + 'TRAIN/*/F*/'

    input_male_test_path = input_data_path + 'TEST/*/M*/'
    input_female_test_path = input_data_path + 'TEST/*/F*/'

    output_folder_path = 'data/'  # 输出文件地址
    print('File paths set up complete.')

    # Parameters setting
    dft_frame_size = 1024  # frame size
    num_speeches = 10  # number of sample speeches for each noise, 每个说话人下面是10条音频数据
    # Instantiating Signal class
    signal = Signal(dft_frame_size)
    print('Instantiated Signal class object.')

    # Paths of all noise files
    noise_files = glob.glob(input_noise_path + '*wav')  # 包含前面的文件地址
    # list of all paths to be processed
    file_paths = [input_male_train_path, input_female_train_path,
                  input_male_test_path, input_female_test_path]  # 4个文件夹地址
    print('Speech processing about to start.')

    # iterating over each path
    for path in file_paths:  # path:'/mnt/hd8t/pchao/TIMIT2/TRAIN/*/M*/'
        speech_folder_path_split = re.split('\W+', path)  # ['', 'mnt', 'hd8t', 'pchao', 'TIMIT2', 'TRAIN', 'M', '']
        print('Started processing:', speech_folder_path_split[1], speech_folder_path_split[2], 'files')
        speech_files = glob.glob(path + '*WAV')  # len:3260
        random_speech_files = np.random.choice(speech_files, size=num_speeches)  # (10,)包含完整地址,从speech_files中随机抽取num_speeches个音频,可以重复

        for noise_file in noise_files:  # noise_file:'/mnt/hd8t/pchao/TIMIT2/NOISE/ssn.wav'
            noise_name = re.split('\W+', noise_file)[-2]  # extracting name of each noise file, 'ssn'
            # re.split('\W+', '20180713/haha.wav') -> ['20180713', 'haha', 'wav']
            _, noise_raw = sio.read(noise_file)  # reading noise wav file, int16
            print('Started processing', noise_name, 'noise')

            noise_raw_len = len(noise_raw)  # 1428801
            data_0dB = {'X': [], 'S': [], 'N': []}  # X:混合, S:干净语音, N:混合语音
            data_pos5dB = {'X': [], 'S': [], 'N': []}
            data_neg5dB = {'X': [], 'S': [], 'N': []}

            i = 0
            for speech_file in random_speech_files:
                if i % 10 == 0:
                    print('Status:', i, 'speech files processed...')
                i += 1
                # reading speech wav file
                _, speech = sio.read(speech_file)
                speech_len = len(speech)

                # making random cut of noise signal
                max_idx = noise_raw_len - speech_len
                idx = np.random.randint(low=0, high=max_idx)  # 随机产生一个起点
                noise = noise_raw[idx:idx + speech_len]
                assert len(noise) == speech_len

                # normalizing using standard deviation偏离
                noise_0dB = noise / noise.std()
                speech_0dB = speech / speech.std()
                # 0dB mixture
                x_0dB = speech_0dB + noise_0dB
                # 5dB mixture
                scale_pos5dB = 10 ** (-5 / 20)
                x_pos5dB = speech_0dB + scale_pos5dB * noise_0dB
                # -5dB mixture
                scale_neg5dB = 10 ** (5 / 20)
                x_neg5dB = speech_0dB + scale_neg5dB * noise_0dB

                # Generating spectrograms
                S = signal.get_spectrogram(_data=speech_0dB)
                N_0dB = signal.get_spectrogram(_data=noise_0dB)
                N_pos5dB = signal.get_spectrogram(_data=scale_pos5dB * noise_0dB)
                N_neg5dB = signal.get_spectrogram(_data=scale_neg5dB * noise_0dB)

                X_0dB = signal.get_spectrogram(_data=x_0dB)
                X_pos5dB = signal.get_spectrogram(_data=x_pos5dB)
                X_neg5dB = signal.get_spectrogram(_data=x_neg5dB)

                data_0dB['X'].append(X_0dB)
                data_0dB['S'].append(S)
                data_0dB['N'].append(N_0dB)

                data_pos5dB['X'].append(X_pos5dB)
                data_pos5dB['S'].append(S)
                data_pos5dB['N'].append(N_pos5dB)

                data_neg5dB['X'].append(X_neg5dB)
                data_neg5dB['S'].append(S)
                data_neg5dB['N'].append(N_neg5dB)
                # --- processing for a signal file ends ---

            speech_file_path_split = re.split('\W+', random_speech_files[0])  # 分割：/mnt/hd8t/pchao/TIMIT2/TRAIN/DR1/MPSW0/SX257.WAV
            speech_type = speech_file_path_split[-5].lower()  # test or train,原[1],并转换为小写
            # male or female
            speaker_gender = 'male' if speech_file_path_split[-3][0] == 'M' else 'female'
            output_file_path_name = output_folder_path + speech_type + '/' + speaker_gender + '_'
            pickle.dump(file=open(output_file_path_name + noise_name + '_0dB', 'wb'), obj=data_0dB)
            pickle.dump(file=open(output_file_path_name + noise_name + '_pos5dB', 'wb'), obj=data_pos5dB)
            pickle.dump(file=open(output_file_path_name + noise_name + '_neg5dB', 'wb'), obj=data_neg5dB)

            print('Processed', noise_name, 'noise')
