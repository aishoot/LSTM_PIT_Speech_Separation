#!/bin/bash
#rm -rf logdir/train/*
        #--dim=896 \

#	--data_dir=./mix/out \

python train.py \
	--data_dir=mix\
	--test_dir=mix\
        --rnn_type=GRU \
        --dim=896 \
        --n_rnn=1 \
        --seq_len=256 \
        --batch_size=1 \
	--optimizer=adam \
	--num_gpus=1 \
        --logdir=afterlog
