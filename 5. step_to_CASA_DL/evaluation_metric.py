import re
import glob
import pickle
import itertools
import numpy as np
from scipy.io import wavfile as swav
import scipy.io as sio
import pandas as pd

# load the .pkl files for the SNR ...

# SNR_filename = 'results/IBM_SNR_results_all_noise_all_gender.pkl'
# snr_output = pickle.load(open(SNR_filename, 'rb'))
# key_list = [k for k in snr_output.keys()]
# key_list = sorted(key_list)
# print(key_list)
#
# table_0db = pd.DataFrame()
# table_neg5db = pd.DataFrame()
# table_pos5db = pd.DataFrame()
#
# for k in key_list:
#     items = re.split('_',k)
#     print(items)
#     if items[2]== '0dB':
#         snr_values = snr_output[k]
#         avg_snr = np.mean(np.array(snr_values))
#         df = pd.DataFrame([[items[1],avg_snr]])
#         table_0db = table_0db.append(df,ignore_index=True)
#
#     elif items[2] == 'neg5dB':
#         snr_values = snr_output[k]
#         avg_snr = np.mean(np.array(snr_values))
#         df = pd.DataFrame([[items[1], avg_snr]])
#         table_neg5db = table_neg5db.append(df, ignore_index=True)
#
#     elif items[2] == 'pos5dB':
#         snr_values = snr_output[k]
#         avg_snr = np.mean(np.array(snr_values))
#         df = pd.DataFrame([[items[1], avg_snr]])
#         table_pos5db = table_pos5db.append(df, ignore_index=True)
#
#
# table_0db.to_csv('results/snr_table_all_gender_all_noise_0db.csv')
# table_neg5db.to_csv('results/snr_table_all_gender_all_noise_neg5db.csv')
# table_pos5db.to_csv('results/snr_table_all_gender_all_noise__pos5db.csv')
#


# load the .mat files for th STOI ...
file = open('irm_all_noise_male_gender_female_test_keys.txt', 'r')
stoi_keys = file.read()
file.close()

file = open('irm_all_noise_male_gender_female_test_values.txt', 'r')
stoi_values = file.read()
file.close()

stoi_keys = re.split('-', stoi_keys)
stoi_values = re.split(',', stoi_values)
stoi_values = [float(x) for x in stoi_values]

print(stoi_keys)
print(stoi_values)

stoi_dict = {}
for i in range(int(len(stoi_keys) / 10)):
    key = stoi_keys[i * 10]
    values = stoi_values[i * 10:(i + 1) * 10]
    print(values)
    k = '_'.join(re.split('_', key)[:-1])
    stoi_dict[k] = np.mean(np.array(values))

print(len(stoi_dict.keys()))
print('irm_all_noise_male_gender_female_test', stoi_dict)
