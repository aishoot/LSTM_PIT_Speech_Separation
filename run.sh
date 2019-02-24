#! /bin/bash
# Author:
#   Sining Sun (Northwestern Polytechnical University, China)
#   Chao Peng, EECS, Peking University, Beijing, China.
# This recipe is used to do NN-PIT (LSTM, DNN or  BLSTM)

# Program timer with bash
time_program(){
    echo `date --date='0 days ago' "+%Y-%m-%d %H:%M:%S"`
}

echo "The program starts at `time_program`."
step=0 
speaker_num="2speaker"
lists_dir=storage/lists   # lists_dir is used to store some necessary files lists
tfrecords_dir=storage/tfrecords 
mkdir -p $lists_dir
wav_dir=～Dataset/WSJ0-mix/mix/data/2speakers_0dB/wav8k/min 

# audio and stft configs
sample_rate=8000
sample_rate_name=`expr $sample_rate / 1000`KHz
window_size=256
window_shift=128

# net config
rnn_num_layers=3
rnn_size=496
model_type=LSTM
input_size=129
output_size=129
keep_prob=0.8
learning_rate=0.0005
halving_factor=0.7
gpu_id='0'
TF_CPP_MIN_LOG_LEVEL=1 

# train opts
prefix=PIT 
assignment=def
name=${prefix}_${model_type}_${rnn_num_layers}_${rnn_size}_${speaker_num}_${sample_rate_name}_"0dB"
TF_save_dir=storage/TFCheckpoint/$name
separated_dir=storage/separated/${name}_${assignment} 
resume_training=true

# note: we want to use gender information, but we didn't use in this version. 
# but when we prepared our data, we stored the gender information (maybe useful in the future).
# wsj-train-spkrinfo.txt:  https://catalog.ldc.upenn.edu/docs/LDC93S6A/wsj0-train-spkrinfo.txt

#####################################################################################################
#   NOTE for STEP 0:                                                                              ###
#       1. Extract STFT features, and convert them to the format of tfrecords                     ###
#####################################################################################################
if [ $step -le 0 ]; then
	echo -e "\nGenerate wav file to .lst file at `time_program`."
    for x in tr cv tt; do
        python local/make_wav_list.py --wav_dir ${wav_dir}/$x/mix --output_lst ${lists_dir}/${x}_wav.lst
    done

	# tfrecords are stored in data/tfrecords/{tr, cv, tt}_tfrecord/
	echo -e "\nfrom wav file generate {tr, cv, tt}_tfrecord/*.tfrecords at `time_program`."
    for x in tr cv tt; do
        python -u local/gen_tfrecords.py --gender_list local/wsj0-train-spkrinfo.txt \
		--window_size $window_size --window_shift $window_shift --sample_rate $sample_rate \
		${wav_dir}/$x/ ${lists_dir}/${x}_wav.lst ${tfrecords_dir}/${x}_tfrecord &
    done
    wait

    # Here, we have made tfrecords list file for tr, cv and tt data,
    # Make sure you have generated tfrecords files in $tfrecords_dir/{tr, cv, tt}_tfrecord/
    # The list files name must be tr_tf.lst, cv_tf.lst and tt_tf.lst.
    echo -e "\nFrom {tr, cv, tt}_tfrecord/*.tfrecords generate {tr, cv, tt}_tf.lst at `time_program`."
    for x in tr tt cv; do
        find $tfrecords_dir/${x}_tfrecord/ -iname "*.tfrecords" > $lists_dir/${x}_tf.lst
        # storage/tfrecords/tr_tfrecord/aa.tfrecords
        # storage/tfrecords/tr_tfrecord/bb.tfrecords
    done
fi


#####################################################################################################
#   NOTE for STEP 1:   Train                                                                      ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################
if [ $step -le 1 ]; then
    echo -e "\nStart Traing RNN(LSTM or BLSTM) model at `time_program`.\n"
    decode=0
    batch_size=25

    tr_cmd="python -u run_lstm.py --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size \
    --rnn_size=$rnn_size --decode=$decode --learning_rate=$learning_rate --TF_save_dir=$TF_save_dir \
    --separated_dir=$separated_dir --keep_prob=$keep_prob --input_size=$input_size --output_size=$output_size  \
    --assign=$assignment --resume_training=$resume_training --model_type=$model_type --halving_factor=$halving_factor"

    #echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi


#####################################################################################################
#   NOTE for STEP 2:     Decode                                                                   ###
#       1. Make sure that you configure the RNN/data_dir/model_dir/ all rights                    ###
#####################################################################################################
if [ $step -le 2 ]; then
    echo -e "\nStart Decoding at `time_program`.\n"
    decode=1
    batch_size=30
    tr_cmd="python -u run_lstm.py --lists_dir=$lists_dir  --rnn_num_layers=$rnn_num_layers --batch_size=$batch_size \
    --decode=$decode --learning_rate=$learning_rate --TF_save_dir=$TF_save_dir --separated_dir=$separated_dir --keep_prob=$keep_prob \
    --input_size=$input_size --output_size=$output_size  --assign=$assignment --resume_training=$resume_training --rnn_size=$rnn_size \
    --model_type=$model_type --czt_dim=128 --window_size=$window_size --window_shift=$window_shift --sample_rate=$sample_rate"

    #echo $tr_cmd
    CUDA_VISIBLE_DEVICES=$gpu_id TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL $tr_cmd
fi

echo -e "\nAll programs finish running at `time_program`."