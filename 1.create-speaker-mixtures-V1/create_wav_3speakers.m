% create_wav_3_speakers.m
%
% Create 3-speaker mixtures
% 
% This script assumes that WSJ0's wv1 sphere files have already
% been converted to wav files, using the original folder structure
% under wsj0/, e.g., 
% 11-1.1/wsj0/si_tr_s/01t/01to030v.wv1 is converted to wav and 
% stored in YOUR_PATH/wsj0/si_tr_s/01t/01to030v.wav, and
% 11-6.1/wsj0/si_dt_05/050/050a0501.wv1 is converted to wav and
% stored in YOUR_PATH/wsj0/si_dt_05/050/050a0501.wav.
% Relevant data from all disks are assumed merged under YOUR_PATH/wsj0/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (C) 2016 Mitsubishi Electric Research Labs 
%                          (Jonathan Le Roux, John R. Hershey, Zhuo Chen)
%   Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath('./voicebox')
data_type = {'tr','cv','tt'};
wsj0root = './'; % YOUR_PATH/,  the folder containing wsj0/
output_dir16k='./data/3speakers/wav16k';
output_dir8k='./data/3speakers/wav8k';

min_max = {'min'}; %{'min','max'};

for i_mm = 1:length(min_max)
    for i_type = 1:length(data_type)
        if ~exist([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        if ~exist([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}],'dir')
            mkdir([output_dir8k '/' min_max{i_mm} '/' data_type{i_type}]);
        end
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/s3/']); %#ok<NASGU>
        status = mkdir([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/mix/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s3/']); %#ok<NASGU>
        status = mkdir([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/']);
                
        TaskFile = ['mix_3_spk_' data_type{i_type} '.txt'];
        fid=fopen(TaskFile,'r');
        C=textscan(fid,'%s %f %s %f %s %f');
        
        Source1File = ['mix_3_spk_' min_max{i_mm} '_' data_type{i_type} '_1'];
        Source2File = ['mix_3_spk_' min_max{i_mm} '_' data_type{i_type} '_2'];
        Source3File = ['mix_3_spk_' min_max{i_mm} '_' data_type{i_type} '_3'];
        MixFile     = ['mix_3_spk_' min_max{i_mm} '_' data_type{i_type} '_mix'];
        fid_s1 = fopen(Source1File,'w');
        fid_s2 = fopen(Source2File,'w');
        fid_s3 = fopen(Source3File,'w');
        fid_m  = fopen(MixFile,'w');
        
        num_files = length(C{1});
        fs8k=8000;
        
        scaling_16k = zeros(num_files,3);
        scaling_8k = zeros(num_files,3);
        scaling16bit_16k = zeros(num_files,1);
        scaling16bit_8k = zeros(num_files,1);
        fprintf(1,'%s\n',[min_max{i_mm} '_' data_type{i_type}]);
        for i = 1:num_files
            [inwav1_dir,invwav1_name,inwav1_ext] = fileparts(C{1}{i});
            [inwav2_dir,invwav2_name,inwav2_ext] = fileparts(C{3}{i});
            [inwav3_dir,invwav3_name,inwav3_ext] = fileparts(C{5}{i});
            fprintf(fid_s1,'%s\n',C{1}{i});%[inwav1_dir,'/',invwav1_name,inwav1_ext]);
            fprintf(fid_s2,'%s\n',C{3}{i});%[inwav2_dir,'/',invwav2_name,inwav2_ext]);
            fprintf(fid_s3,'%s\n',C{5}{i});%[inwav3_dir,'/',invwav3_name,inwav3_ext]);
            inwav1_snr = C{2}(i);
            inwav2_snr = C{4}(i);
            inwav3_snr = C{6}(i);
            mix_name = [invwav1_name,'_',num2str(inwav1_snr),...
                        '_',invwav2_name,'_',num2str(inwav2_snr),...
                        '_',invwav3_name,'_',num2str(inwav3_snr)];
            fprintf(fid_m,'%s\n',mix_name);
            
            % get input wavs
            [s1, fs] = wavread([wsj0root C{1}{i}]);
            s2       = wavread([wsj0root C{3}{i}]);
            s3       = wavread([wsj0root C{5}{i}]);
            
            % resample, normalize 8 kHz file, save scaling factor
            s1_8k=resample(s1,fs8k,fs);
            [s1_8k,lev1]=activlev(s1_8k,fs8k,'n'); % y_norm = y /sqrt(lev);
            s2_8k=resample(s2,fs8k,fs);
            [s2_8k,lev2]=activlev(s2_8k,fs8k,'n');
            s3_8k=resample(s3,fs8k,fs);
            [s3_8k,lev3]=activlev(s3_8k,fs8k,'n');
            
            weight_1=10^(inwav1_snr/20);
            weight_2=10^(inwav2_snr/20);
            weight_3=10^(inwav3_snr/20);
            
            s1_8k = weight_1 * s1_8k;
            s2_8k = weight_2 * s2_8k;
            s3_8k = weight_3 * s3_8k;
            
            switch min_max{i_mm}
                case 'max'
                    mix_8k_length = max([length(s1_8k),length(s2_8k),length(s3_8k)]);
                    s1_8k = cat(1,s1_8k,zeros(mix_8k_length - length(s1_8k),1));
                    s2_8k = cat(1,s2_8k,zeros(mix_8k_length - length(s2_8k),1));
                    s3_8k = cat(1,s3_8k,zeros(mix_8k_length - length(s3_8k),1));
                case 'min'
                    mix_8k_length = min([length(s1_8k),length(s2_8k),length(s3_8k)]);
                    s1_8k = s1_8k(1:mix_8k_length);
                    s2_8k = s2_8k(1:mix_8k_length);
                    s3_8k = s3_8k(1:mix_8k_length);
            end
            mix_8k = s1_8k + s2_8k + s3_8k;
                    
            max_amp_8k = max(cat(1,abs(mix_8k(:)),abs(s1_8k(:)),abs(s2_8k(:)),abs(s3_8k(:))));
            mix_scaling_8k = 1/max_amp_8k*0.9;
            s1_8k = mix_scaling_8k * s1_8k;
            s2_8k = mix_scaling_8k * s2_8k;
            s3_8k = mix_scaling_8k * s3_8k;
            mix_8k = mix_scaling_8k * mix_8k;
            
            % apply same gain to 16 kHz file
            s1_16k = weight_1 * s1 / sqrt(lev1);
            s2_16k = weight_2 * s2 / sqrt(lev2);
            s3_16k = weight_3 * s3 / sqrt(lev3);
            
            switch min_max{i_mm}
                case 'max'
                    mix_16k_length = max([length(s1_16k),length(s2_16k),length(s3_16k)]);
                    s1_16k = cat(1,s1_16k,zeros(mix_16k_length - length(s1_16k),1));
                    s2_16k = cat(1,s2_16k,zeros(mix_16k_length - length(s2_16k),1));
                    s3_16k = cat(1,s3_16k,zeros(mix_16k_length - length(s3_16k),1));
                case 'min'
                    mix_16k_length = min([length(s1_16k),length(s2_16k),length(s3_16k)]);
                    s1_16k = s1_16k(1:mix_16k_length);
                    s2_16k = s2_16k(1:mix_16k_length);
                    s3_16k = s3_16k(1:mix_16k_length);
            end
            mix_16k = s1_16k + s2_16k + s3_16k;
            
            max_amp_16k = max(cat(1,abs(mix_16k(:)),abs(s1_16k(:)),abs(s2_16k(:)),abs(s3_16k(:))));
            mix_scaling_16k = 1/max_amp_16k*0.9;
            s1_16k = mix_scaling_16k * s1_16k;
            s2_16k = mix_scaling_16k * s2_16k;
            s3_16k = mix_scaling_16k * s3_16k;
            mix_16k = mix_scaling_16k * mix_16k;
            
            % save 8 kHz and 16 kHz mixtures, as well as
            % necessary scaling factors
            
            scaling_16k(i,1) = weight_1 * mix_scaling_16k/ sqrt(lev1);
            scaling_16k(i,2) = weight_2 * mix_scaling_16k/ sqrt(lev2);
            scaling_16k(i,3) = weight_3 * mix_scaling_16k/ sqrt(lev3);
            scaling_8k(i,1) = weight_1 * mix_scaling_8k/ sqrt(lev1);
            scaling_8k(i,2) = weight_2 * mix_scaling_8k/ sqrt(lev2);
            scaling_8k(i,3) = weight_3 * mix_scaling_8k/ sqrt(lev3);
            
            scaling16bit_16k(i) = mix_scaling_16k;
            scaling16bit_8k(i)  = mix_scaling_8k;
            
            wavwrite(s1_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav']);
            wavwrite(s1_16k,fs,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s1/' mix_name '.wav']);
            wavwrite(s2_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav']);
            wavwrite(s2_16k,fs,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s2/' mix_name '.wav']);
            wavwrite(s3_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/s3/' mix_name '.wav']);
            wavwrite(s3_16k,fs,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/s3/' mix_name '.wav']);
            wavwrite(mix_8k,fs8k,[output_dir8k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav']);
            wavwrite(mix_16k,fs,[output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/mix/' mix_name '.wav']);
            
            if mod(i,10)==0
                fprintf(1,'.');
                if mod(i,200)==0
                    fprintf(1,'\n');
                end
            end
            
        end
        save([output_dir8k  '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_8k','scaling16bit_8k');
        save([output_dir16k '/' min_max{i_mm} '/' data_type{i_type} '/scaling.mat'],'scaling_16k','scaling16bit_16k');
        
        fclose(fid);
        fclose(fid_s1);
        fclose(fid_s2);
        fclose(fid_s3);
        fclose(fid_m);
    end
end
