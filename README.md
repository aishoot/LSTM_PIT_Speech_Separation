# LSTM_PIT_Speech_Separation
Despite the significant progress made in dictating single-speaker speech in the recent years, the progress made in multitalker mixed speech separation and recognition, often referred to as the "cocktail-party problem", has been less impressive.  Although human listeners can easily perceive separate sources in an acoustic mixture, the same task seems to be extremely difficult for automatic computing systems, especially when only a single microphone recording of the mixed-speech is available.

<img width="80%" height="80%" src="pictures/time_in_seconds.png"/>

## Model Summary
Layer |  Layer Name | Input Shape | Output Shape 
:-: | :-: | :-: | :-: 
the First Layer  | BLSTM_1 | (?, 500, 201) | (?, 500, 60) 
the Second Layer | BLSTM_2 | (?, 500, 60)  | (?, 500, 40) 
the Third Layer  | BLSTM_3 | (?, 500, 40)  | (?, 500, 80)
the Fourth Layer | maxpooling1d | (?, 500, 80) | (?, 250, 80) 
the Fifth Layer | flatten | (?, 250, 80) | (?, 20000) 
the Sixth Layer | dense | (?, 20000) | (?, 11) 
the Seventh Layer | activation | (?, 11) | (?, 11) 

"?" represents the number of samples.<br> 

## Dependency Library
* [librosa](https://librosa.github.io/)
* Tensorflow (my test version: 1.4.0)
* Anaconda3 (Contains Python3.5+)

## Usage
Run the command line script:
Script to generate the multi-speaker dataset using WSJ0. 
```

```

## Reference Paper & Code
I'd like to thank Dong Yu et al for the paper and Sining Sun, Unisound for sharing their code.
* __Paper__: Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation.
* __Authors__: Dong Yu, Morten Kolbæk, Zheng-Hua Tan, Jesper Jensen
* __Published__: [ICASSP 2017](https://ieeexplore.ieee.org/document/7952154/) (5-9 March 2017)
* __Code__: https://github.com/faroit/CountNet
* __Dataset__: [WSJ0 data](https://catalog.ldc.upenn.edu/ldc93s6a)
* __Create Mixed Speech Method__: Please click here - [create-speaker-mixtures.zip](http://www.merl.com/demos/deep-clustering/create-speaker-mixtures.zip).

## Follow-up Work
I will work on speech separation for a long time. You can fork this repository if interested and pay close attention to my recent study.
