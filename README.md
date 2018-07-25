# LSTM_PIT Training for Two Speakers
The progress made in multitalker mixed speech separation and recognition, often referred to as the "cocktail-party problem", has been less impressive. Although human listeners can easily perceive separate sources in an acoustic mixture, the same task seems to be extremely difficult for computers, especially when only a single microphone recording the mixed-speech.

<img width="75%" height="75%" src="spectrogram.PNG"/>

## Speration Performance
For **LSTM**, results of the mixed audio with different gender are as follows:

Gender Combination | SDR | SAR | SIR | STOI | ESTOI | PESQ 
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
Overall| 6.453328 | 9.372059 | 11.570311 | 0.536987 | 0.429255 | 1.653391
Male & Female | 8.238905 | 9.939668 | 14.531649 | 0.521656 | 0.421868 | 1.663442
Female & Female | 3.538810 | 8.134054 | 7.230494 | 0.560099 | 0.441704 | 1.553452
Male & Male | 5.011563 | 9.026763 | 9.000010 | 0.550071 | 0.435083 | 1.675609

For **BLSTM**, results of the mixed audio with different gender are as follows:

Gender Combination | SDR | SAR | SIR | STOI | ESTOI | PESQ 
:-: | :-: | :-: | :-: | :-: | :-: | :-: |
Overall| 9.177447 | 10.629142 | 16.116564 | 0.473229 | 0.377204 | 1.651099
Male & Female | 10.647645 | 11.691969 | 18.203052 | 0.488542 | 0.393999 | 1.731112
Female & Female | 7.309365 | 9.393608 | 13.355384 | 0.459762 | 0.363213 | 1.478075
Male & Male | 7.797448 | 9.589827 | 14.198003 | 0.456667 | 0.358757 | 1.602058

From above results we can see that the separation effect of mixed gender audio is better than that of the same gender and BLSTM performs better than LSTM.

* SDR: Signal to Distortion Ratio
* SAR: Signal to Artifact Ratio
* SIR: Signal to Interference Ratio
* STOI: Short Time Objective Intelligibility Measure
* ESTOI: Extended Short Time Objective Intelligibility Measure
* PESQ: Perceptual Evaluation of Speech Quality

## Dependency Library
* [librosa](https://librosa.github.io/)
* Matlab (my test version: R2016b 64-bit)
* Tensorflow (my test version: 1.4.0)
* Anaconda3 (Contains Python3.5+)

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

## File Description
* 1-create-speaker-mixtures-V1: Version one of scripts to generate the wsj0-mix multi-speaker dataset.
* 2-create-speaker-mixtures-V2: Version two of scripts to generate the wsj0-mix multi-speaker dataset.
* 3-step_to_CASA_DL: Step to multi-speaker speech separation with Computational Auditory Scene Analysis and Deep Learning.
* 4-hjkwon0609-speech_separation: Speech separation implementation of hjkwon0609 but I can't reappearing source code because there is no experimental data.
* 5-Unisound-SpeechSeparation: Speech separation implementation of Unisound but there is some code wrong.

## Reference Paper & Code
Thank Dong Yu et al. for the paper and Sining Sun, Unisound et al. for sharing their code.
* __Paper__: Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation.
* __Authors__: Dong Yu, Morten Kolbæk, Zheng-Hua Tan, Jesper Jensen
* __Published__: [ICASSP 2017](https://ieeexplore.ieee.org/document/7952154/) (5-9 March 2017)
* __Code__: [Unisound/SpeechSeparation](https://github.com/Unisound/SpeechSeparation), [hjkwon0609/speech_separation](https://github.com/hjkwon0609/speech_separation), [Training-Targets-for-Speech-Separation-Neural-Networks](https://github.com/jaideeppatel/Training-Targets-for-Speech-Separation-Neural-Networks), [snsun/pit-speech-separation](https://github.com/snsun/pit-speech-separation), [MERL_Deep Clustering](http://www.merl.com/demos/deep-clustering)
* __Dataset__: [WSJ0 data](https://catalog.ldc.upenn.edu/ldc93s6a), [VCTK-Corpus](http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)
* __SDR/SAR/SIR__
    * Toolbox: [BSS Eval](http://bass-db.gforge.inria.fr/bss_eval/), [The PEASS Toolkit](http://bass-db.gforge.inria.fr/peass/), [craffel/mir_eval/separation.py](https://github.com/craffel/mir_eval/blob/master/mir_eval/separation.py)
    * Paper: [Performance measurement in blind audio source separation](https://ieeexplore.ieee.org/document/1643671/)

## Follow-up Work
I will study on speech separation for a long time. You can pay close attention to my recent work if interested.

*More code will be uploaded in the future! You can fork this repository.*
