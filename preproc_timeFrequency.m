%% This is a very basic time frequency analysis to see the average induced 
% responses independently of the conditions, so as to approximately 
% identify the general frequency band of interest in each channel.

for sbj_number = 1:length(SubjectsList);
    
    % Select subject
    sbj_initials = SubjectsList{sbj_number};
    data_path = [path 'data/' sbj_initials '/'] ;
    
    
    %% identify files
    files = dir([data_path 'fif/*.fif']);
    files = files(SelFileOrder{sbj_number}); % put files in order
    
    %% preprocessing of each block
    for f = 1:length(files)
        %% Trial definition
        cfg = [];
        cfg.dataset             = fullfile([data_path 'fif'],files(f).name);
        cfg.headerformat        = 'neuromag_mne';
        cfg.dataformat          = 'neuromag_mne';
        %cfg.trialfun           = 'ft_trialfun_general';
        cfg.trialdef.eventtype  = 'STI101';
        cfg.trialdef.eventvalue = 1:25;
        cfg.trialdef.prestim    = 1.000; % number, latency in seconds (optional);
        cfg.trialdef.poststim   = 2.000; % number, latency in seconds (optional);
        cfg                     = ft_definetrial(cfg);
        
        %% Read in the data
        cfg.channel     = {'MEG','MEGREF'};
        cfg.continuous  = 'yes';
        cfg.feedback    = 'gui';
        %     cfg.channel     = 1:306;
        data            = ft_preprocessing(cfg);
        
        %% wavelet decomposition
        cfg = [];
        cfg.method     = 'wavelet';
        cfg.width      = 7;
        cfg.output     = 'pow';
        cfg.foi        = logspace(log(2),log(10),70);
        cfg.toi        = -0.5:0.020:.800;
        data     = ft_freqanalysis(cfg, data);
      
        
        %% store all data
        datas{f} = data;
        clear data;
    end


%% we we cannot concatenate across runs because we did not keep single trials

%% save data
save([data_path 'preprocessed/' sbj_initials '_preprocessed_meanTF.mat'], 'datas'); % save details


end


%% Plot
% retrieve results across subjects
m = zeros(306,69,66,length(SubjectsList));
for s = 1:length(SubjectsList)
    
    % Select subject
    sbj_initials = SubjectsList{s};
    data_path = [path 'data/' sbj_initials '/'] ;
    load([data_path 'preprocessed/' sbj_initials '_preprocessed_meanTF.mat'], 'datas');
    
    for f = 1:length(datas)
        % dB baseline
        datas{f}.powspctrm = 10*log10(datas{f}.powspctrm);
        cfg              = [];
        cfg.baseline     = [-0.5 -0.1];
        cfg.baselinetype = 'absolute';
        [data] = ft_freqbaseline(cfg, datas{f});
        % integrate across files (i.e. average)
        m(:,:,:,s) = m(:,:,:,s) + data.powspctrm;
    end
    % mean
    m(:,:,:,s) = m(:,:,:,s)/length(datas);
end

% Plot average TF across all channels across all subjects
figure(1);
subplot(1,2,1); % average TF response across all channels
imagesc(data.time, [], squeeze(mean(mean(m,1),4)), [-1.5 1])
set(gca,'ydir', 'normal', ...
    'ytick', 1:4:length(data.freq), ...
    'yticklabel', round(data.freq(1:4:end)))
title('mean TF across channels')

subplot(1,2,2); % as TF may increase and decrease in different channels, 
% you may want to just vizualize the tf bands where there is an induced 
% response
imagesc(data.time, [], squeeze(mean(log(mean(m.^2,1)),4)))
set(gca,'ydir', 'normal', ...
    'ytick', 1:4:length(data.freq), ...
    'yticklabel', round(data.freq(1:4:end)))
title('mean TF^2 across channels') 
% plot mean TF for each channel across subjects


figure(2); % topographies across time freqs
data.powspctrm  = mean(m,4);
cfg = [];
cfg.showlabels   = 'yes';
cfg.layout       = 'neuromag306mag';
ft_multiplotTFR(cfg, data)


figure(3); % single subject TF on a occipital channel
for s = 1:length(SubjectsList)
    subplot(3,4,s);
    imagesc(data.time, [], squeeze(mean(m(237,:,:,s),1)), [-1.5 1])
set(gca,'ydir', 'normal', ...
    'ytick', 1:4:length(data.freq), ...
    'yticklabel', round(data.freq(1:4:end)))

end
