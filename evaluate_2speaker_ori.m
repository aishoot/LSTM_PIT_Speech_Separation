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
model_name = '2speakers_0dB_original';
mix_wav_dir = [tt_wav_dir '/mix/'];
spk1_dir = [tt_wav_dir, '/s1/'];
spk2_dir = [tt_wav_dir, '/s2/'];

lists = dir(spk2_dir);  %3002*1 struct
len = length(lists) - 2;  %3000
SDR =  zeros(len, 2);
SIR = SDR;
SAR = SDR;
STOI = SDR;
ESTOI = SDR;
PESQ = SDR;
error_num_STOI = 0;
error_num_ESTOI = 0;
error_num_PESQ = 0;


for i = 3:len+2 
    name = lists(i).name;
    part_name = name(1:end-4);
    fprintf('Computing Audio:%s, Number:%d ...\n', [part_name '.wav'], i-2)

    mix_wav1 = audioread([mix_wav_dir part_name '.wav']);  %35328*1 double
    mix_wav = [mix_wav1, mix_wav1];  %35328*2 double
    
    ori_wav1 = audioread([spk1_dir part_name '.wav']);  %35269*1 double
    ori_wav2 = audioread([spk2_dir part_name '.wav']);  %35269*1 double
    ori_wav = [ori_wav1, ori_wav2];  %35269*2 double
    
    min_len = min(size(ori_wav, 1), size(mix_wav, 1));  %35269
    mix_wav = mix_wav(1:min_len, :);  %35269*2 double
    ori_wav = ori_wav(1:min_len, :);  %35269*2 double
    [SDR(i-2, :),SIR(i-2, :),SAR(i-2, :),perm]=bss_eval_sources(mix_wav', ori_wav');

    x1 = stoi(ori_wav(:,1), mix_wav(:,1), sample_rate);
    x2 = stoi(ori_wav(:,2), mix_wav(:,2), sample_rate);
    if ~isnan(x1) & ~isnan(x2)
    %if x1 ~= NaN & x2 ~= NaN
        STOI(i-2, 1) = x1;
        STOI(i-2, 2) = x2;
    else
    	STOI(i-2, 1) = 0;
        STOI(i-2, 2) = 0;
    	error_num_STOI = error_num_STOI + 1;
        fprintf('STOI NaN happens in computing the audio:%s, i=%d.\n', [part_name '.wav'], i-2)
    end
    
    e1 = estoi(ori_wav(:,1), mix_wav(:,1), sample_rate);
    e2 = estoi(ori_wav(:,2), mix_wav(:,2), sample_rate);
    if ~isnan(x1) & ~isnan(x2)
    %if e1 ~= NaN & e2 ~= NaN 
        ESTOI(i-2, 1) = e1;
        ESTOI(i-2, 2) = e2;
    else
    	STOI(i-2, 1) = 0;
        STOI(i-2, 2) = 0;
    	error_num_ESTOI = error_num_ESTOI + 1;
        fprintf('ESTOI NaN happens in computing the audio:%s, i=%d.\n', [part_name '.wav'], i-2)
    end

    try
        PESQ(i-2, 1) = pesq([spk1_dir part_name '.wav'], [mix_wav_dir part_name '.wav']);
        PESQ(i-2, 2) = pesq([spk2_dir part_name '.wav'], [mix_wav_dir part_name '.wav']);
    catch ErrorInfo
        PESQ(i-2, 1) = 0;
        PESQ(i-2, 2) = 0;
        disp(ErrorInfo)
        error_num_PESQ = error_num_PESQ + 1;
        fprintf('PESQ Error happens in computing the audio:%s, i=%d.\n', [part_name '.wav'], i-2)
    end
end


fprintf('Model Name: %s.\n', model_name)
fprintf('The mean SDR is %f.\n', mean(mean(SDR)))
fprintf('The mean SAR is %f.\n', mean(mean(SAR)))
fprintf('The mean SIR is %f.\n', mean(mean(SIR)))
fprintf('Mean STOI is %f.\n', mean(sum(STOI)/(len - error_num_STOI)))
fprintf('Mean ESTOI is %f.\n', mean(sum(ESTOI)/(len - error_num_ESTOI)))
fprintf('Mean PESQ is %f.\n', mean(sum(PESQ)/(len - error_num_PESQ)))
save(['matfiles/evaluate_' model_name], 'SDR', 'SAR', 'SIR', 'STOI', 'ESTOI', 'PESQ', 'lists');

time_length = toc;
hour = floor(time_length/3600);
remaining = mod(time_length, 3600);
minute = floor(remaining/60);
second = mod(remaining, 60);
fprintf('\nElapsed time is %d hour(s), %d minute(s), %d second(s).\n', hour, minute, floor(second))
