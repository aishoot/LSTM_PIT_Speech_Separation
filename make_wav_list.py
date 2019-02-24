#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright June 19, 2018, Chao Peng, EECS, Peking University.
"""
tr_wav.lst format:
...
447o030v_0.1232_050c0109_-0.1232.wav
447o030v_1.7882_444o0310_-1.7882.wav
447o030w_0.52605_446o030e_-0.52605.wav
447o030w_1.9272_420c0203_-1.9272.wav
447o030x_0.03457_441c0209_-0.03457.wav
447o030x_0.70879_420o0307_-0.70879.wav
447o030x_0.98832_441o0308_-0.98832.wav
447o030x_1.4783_422o030p_-1.4783.wav
...
"""
import os
import argparse


parser = argparse.ArgumentParser(description='Generate wav file to .lst')
parser.add_argument('--wav_dir', type=str, default=True, help='Address of a wav file folder.')
parser.add_argument('--output_lst', type=str, default=True, help='Address of the output file ".lst".')

args = parser.parse_args()
wav_dir = args.wav_dir
output_lst = args.output_lst  # "./lists/tr_wav.lst"

wav_files = os.listdir(wav_dir)
with open(output_lst, 'w') as f:
    for file in wav_files:
        f.write(file + "\n")

print("Generate wav file to .lst done!")