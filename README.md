# LSTM_PIT Training for Two Speakers
The progress made in multitalker mixed speech separation and recognition, often referred to as the "cocktail-party problem", has been less impressive. Although human listeners can easily perceive separate sources in an acoustic mixture, the same task seems to be extremely difficult for computers, especially when only a single microphone recording the mixed-speech.

<img width="75%" height="75%" src="spectrogram.PNG"/>

## Speration Performance
DL Method |  SDR | SAR | SIR | STOI | PESQ 
:-: | :-: | :-: | :-: | :-: | :-: |
BLSTM | # | # | # | # | #
LSTM  | # | # | # | # | #

* DL: Deep Learning
* Speech Speration Performance Evaluation Method:
  * SDR: Source to Distortion Ratio
  * SAR: Source to Artifact Ratio
  * SIR: Source to Interference Ratio
  * [STOI](http://cas.et.tudelft.nl/pubs/Taal2010.pdf): Short Time Objective Intelligibility Measure
  * [PESQ](https://ieeexplore.ieee.org/document/941023/): Perceptual Evaluation of Speech Quality

## Dependency Library
* [librosa](https://librosa.github.io/)
* Matlab (my test version: R2016b 64-bit)
* Tensorflow (my test version: 1.4.0)
* Anaconda3 (Contains Python3.5+ and so on)

## Usage Process
#### Generate Mixed and Target Speech:
When you have WSJ0 data, you can use this code [create-speaker-mixtures.zip](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip) to create the mixed speech. We will use 2-speaker mixed audio with samplerate 8000 by default.

#### Run the command line script:
```bash
bash run.sh
```
which contains three steps:
1. Extract STFT features, and convert them to the tfrecords format of Tensorflow.
The training data is ready here. The file structure of training data is now as follows:
```
storage/
├── lists
│   ├── cv_tf.lst
│   ├── cv_wav.lst
│   ├── tr_tf.lst
│   ├── tr_wav.lst
│   ├── tt_tf.lst
│   └── tt_wav.lst
├── separated
├── TFCheckpoint
└── tfrecords
    ├── cv_tfrecord
    │   ├── 01aa010k_1.3053_01po0310_-1.3053.tfrecords
    │   ├── 01aa010p_0.93798_02bo0311_-0.93798.tfrecords
    │   ├── ...
    │   └── 409o0317_1.2437_025c0217_-1.2437.tfrecords
    ├── tr_tfrecord
    │   ├── 01aa010b_0.97482_209a010p_-0.97482.tfrecords
    │   ├── 01aa010b_1.4476_20aa010p_-1.4476.tfrecords
    │   ├── ...
    │   └── 409o0316_1.3942_20oo010p_-1.3942.tfrecords
    └── tt_tfrecord
        ├── 050a050a_0.032494_446o030v_-0.032494.tfrecords
        ├── 050a050a_1.7521_422c020j_-1.7521.tfrecords
        ├── ...
        └── 447o0312_2.0302_440c0206_-2.0302.tfrecords
```
Note: {tr,cv,tt}_wav.lst is like as follows:
```
447o030v_0.1232_050c0109_-0.1232.wav
447o030v_1.7882_444o0310_-1.7882.wav
...
447o030x_0.98832_441o0308_-0.98832.wav
447o030x_1.4783_422o030p_-1.4783.wav
```
And {tr,cv,tt}_tf.lst is like as follows:
```
storage/tfrecords/cv_tfrecord/011o031b_1.8_206a010u_-1.8.tfrecords
storage/tfrecords/cv_tfrecord/20ec0109_0.47371_020c020q_-0.47371.tfrecords
...
storage/tfrecords/cv_tfrecord/01zo030l_0.6242_40ho030s_-0.6242.tfrecords
storage/tfrecords/cv_tfrecord/20fo0109_1.1429_017o030p_-1.1429.tfrecords
```
2. Train the deep learning neural network.
3. Decode the network to generate separation audios.

## Reference Paper & Code
I'd like to thank Dong Yu et al for the paper and Sining Sun, Unisound for sharing their code.
* __Paper__: Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation.
* __Authors__: Dong Yu, Morten Kolbæk, Zheng-Hua Tan, Jesper Jensen
* __Published__: [ICASSP 2017](https://ieeexplore.ieee.org/document/7952154/) (5-9 March 2017)
* __Code__: [snsun/pit-speech-separation](https://github.com/snsun/pit-speech-separation), [Unisound/SpeechSeparation](https://github.com/Unisound/SpeechSeparation)
* __Dataset__: [WSJ0 data](https://catalog.ldc.upenn.edu/ldc93s6a)
* __SDR/SAR/SIR Toolbox__: [BSS Eval](http://bass-db.gforge.inria.fr/bss_eval/), [craffel/mir_eval/mir_eval/separation.py](https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py)

## Follow-up Work
I will study on speech separation for a long time. You can pay close attention to my recent work if interested.
