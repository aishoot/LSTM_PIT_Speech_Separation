# LSTM_PIT_Speech_Separation
Speech Separation with LSTM/BLSTM by Permutation Invariant Training method.

# Concurrent Speakers Counter
Estimate the number of concurrent speakers from single channel mixtures to crack the "cocktail-party” problem which is based on a Bidirectional Long Short-Term Memory (BLSTM) which takes into account a past and future temporal context.<br><br>
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
```
python predict_speakers_count.py examples/5_speakers.wav
```
or run the file "predict_speakers_count.ipynb" in proper sequence.

## Reference Paper & Code
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1216072.svg)](https://doi.org/10.5281/zenodo.1216072)<br>
As we all know, it's pretty hard to solve the cocktail-party problem. This is **the ﬁrst study on data-driven speaker count estimation** and the first step to crack the problem. *Thanks for the author's paper and code which helps me a lot.*
* __Homepage__: [AudioLabs Erlangen CountNet](https://www.audiolabs-erlangen.de/resources/2017-CountNet)
* __Title__: Classification vs. Regression in Supervised Learning for Single Channel
 Speaker Count Estimation
* __Authors__: Fabian-Robert Stöter, Soumitro Chakrabarty, Bernd Edler, Emanuël A. P. Habets
* __Published__: ICASSP2018 (Apr 15, 2018 – Apr 20, 2018 in Calgary, Canada)
* __Code__: https://github.com/faroit/CountNet
* __Dataset__: [LibriCount, a dataset for speaker count estimation](https://zenodo.org/record/1216072#.WyS9AoozaUk)

## Follow-up Work
I will work on speech separation for a long time. You can fork this repository if interested and pay close attention to my recent study.
