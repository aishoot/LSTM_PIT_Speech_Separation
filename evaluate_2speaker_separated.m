%====================================================================================
%               Performance Measurement in Multi-speaker Separation
%                   Author: Chao Peng, EECS, Peking University
%            Github: https://github.com/pchao6/LSTM_PIT_Speech_Separation
%                            Revision 1.0, June 2018
%====================================================================================
tic
addpath('/usr/local/MATLAB/R2016b/toolbox/voicebox');
addpath('PESQ');  %PESQ Toolbox According to ITU-T P.862;

sample_rate = 8000;
tt_wav_dir = 'SpeechSeparation/mix/data/2speakers_0dB/wav8k/min/tt';
model_name = 'PIT_BLSTM_3_496_2speaker_8KHz_0dB';
rec_wav_dir = ['SpeechSeparation/separated/' model_name '/'];

spk1_dir = [tt_wav_dir, '/s1/'];
spk2_dir = [tt_wav_dir, '/s2/'];

lists = dir(spk2_dir);
len = length(lists) - 2;
SDR =  zeros(len, 2);
SIR = SDR;
SAR = SDR;
STOI = SDR;
ESTOI = SDR;
PESQ = SDR;
error_num_STOI = 0;
error_num_ESTOI = 0;
error_num_PESQ = 0;

fprintf('Model Name: %s.\n', model_name)
for i = 3:len+2 
    name = lists(i).name; 
    part_name = name(1:end-4);
    fprintf('Computing Audio:%s, Number:%d ...\n', [part_name '.wav'], i-2)

    rec_wav1 = audioread([rec_wav_dir part_name '_1.wav']);
    rec_wav2 = audioread([rec_wav_dir part_name '_2.wav']);
    rec_wav = [rec_wav1, rec_wav2];
    
    ori_wav1 = audioread([spk1_dir part_name '.wav']);
    ori_wav2 = audioread([spk2_dir part_name '.wav']);
    ori_wav = [ori_wav1, ori_wav2];  %35269*2 double
    
    min_len = min(size(ori_wav, 1), size(rec_wav, 1));  %35269
    rec_wav = rec_wav(1:min_len, :);
    ori_wav = ori_wav(1:min_len, :);
    [SDR(i-2, :),SIR(i-2, :),SAR(i-2, :),perm]=bss_eval_sources(rec_wav', ori_wav');

    x1 = stoi(ori_wav(:,1), rec_wav(:,1), sample_rate);
    x2 = stoi(ori_wav(:,2), rec_wav(:,2), sample_rate);
    if ~isnan(x1) & ~isnan(x2)
        STOI(i-2, 1) = x1;
        STOI(i-2, 2) = x2;
    else
    	STOI(i-2, 1) = 0;
        STOI(i-2, 2) = 0;
    	error_num_STOI = error_num_STOI + 1;
    end
    
    e1 = estoi(ori_wav(:,1), rec_wav(:,1), sample_rate);
    e2 = estoi(ori_wav(:,2), rec_wav(:,2), sample_rate);
    if ~isnan(x1) & ~isnan(x2)
        ESTOI(i-2, 1) = e1;
        ESTOI(i-2, 2) = e2;
    else
    	STOI(i-2, 1) = 0;
        STOI(i-2, 2) = 0;
    	error_num_ESTOI = error_num_ESTOI + 1;
    end

    try
        PESQ(i-2, 1) = pesq([spk1_dir part_name '.wav'], [rec_wav_dir part_name '_1.wav']);
        PESQ(i-2, 2) = pesq([spk2_dir part_name '.wav'], [rec_wav_dir part_name '_2.wav']);
    catch ErrorInfo
        PESQ(i-2, 1) = 0;
        PESQ(i-2, 2) = 0;
        disp(ErrorInfo)
        error_num_PESQ = error_num_PESQ + 1;
    end
end

fprintf('The mean SDR is %f.\n', mean(mean(SDR)))
fprintf('The mean SAR is %f.\n', mean(mean(SAR)))
fprintf('The mean SIR is %f.\n', mean(mean(SIR)))
fprintf('The mean STOI is %f.\n', mean(sum(STOI)/(len - error_num_STOI)))
fprintf('The mean ESTOI is %f.\n', mean(sum(ESTOI)/(len - error_num_ESTOI)))
fprintf('The mean PESQ is %f.\n', mean(sum(PESQ)/(len - error_num_PESQ)))
save(['matfiles/evaluate_' model_name], 'SDR', 'SAR', 'SIR', 'STOI', 'ESTOI', 'PESQ', 'lists');

time_length = toc;
hour = floor(time_length/3600);
remaining = mod(time_length, 3600);
minute = floor(remaining/60);
second = mod(remaining, 60);
fprintf('\nElapsed time is %d hour(s), %d minute(s), %d second(s).\n', hour, minute, floor(second))