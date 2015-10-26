for f = 1:length(files)
    %% Trial definition
    cfg = [];
    cfg.dataset             = fullfile(data_path,files(f).name);
    cfg.headerformat        = 'neuromag_mne';
    cfg.dataformat          = 'neuromag_mne';
    %cfg.trialfun           = 'ft_trialfun_general';
    cfg.trialdef.eventtype  = 'STI101';
    cfg.trialdef.eventvalue = 1:25;
    cfg.trialdef.prestim    = 0.300; % number, latency in seconds (optional);
    cfg.trialdef.poststim   = 2.200; % number, latency in seconds (optional);
    cfg = ft_definetrial(cfg);
    
    %% Read in the data
    %cfg.channel     = {'MEG','MEGREF'};
    cfg.continuous  = 'yes';
    cfg.feedback    = 'gui';
    data            = ft_preprocessing(cfg);

     selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);
   
    good_channels   = selchan(data.label,'MEG');% to be defined
    ft_data = data;
    
    
    %% Compute Gamma power
    if ~isfield(cfg.gamma, 'freqs'), cfg.gamma.freqs = 70:5:150; end
    
    cfg.gamma.freqs
    nfreqs = length(cfg.gamma.freqs)-1;
    X = [];
    for f = 1:nfreqs
        for c = length(good_channels):-1:1 % inverted for memory usage
            fbar(length(good_channels)-c+1,length(good_channels))
            %-- filtering
            cfg_tmp             = [];
            cfg_tmp.continuous  = 'yes';
            cfg_tmp.channel     = good_channels(c);
            cfg_tmp.bpfilter    = 'yes';
            cfg_tmp.bpfreq      = [cfg.gamma.freqs(f) cfg.gamma.freqs(f+1)];
            cfg_tmp.bpdir       = 'twopass';
            cfg_tmp.bptype      = 'but';
            cfg_tmp.bpfiltord   = 3;
            cfg_tmp.hilbert     = 'abs';
            evalc('x            = ft_preprocessing(cfg_tmp,ft_data);');
            
            %-- log power estimate
            for trial = length(x.trial):-1:1
                x.trial{trial}  = log(abs(x.trial{trial}).^2);
                whitening(trial) = mean(x.trial{trial},2);
            end
            whitening = nanmean(whitening);
            
            %-- all sequence whitening (baseline correction)
            for trial = length(x.trial):-1:1
                x.trial{trial}  = x.trial{trial} - repmat(whitening,[size(x.trial{1},1),1]);
            end
            
            %-- downsample
            if ~isfield(cfg.resample,'resamplefs'), cfg.resample.resamplefs = 128; end
            if ~isfield(cfg.resample,'detrend'),    cfg.resample.detrend    = 'yes'; end
            evalc('x           = ft_resampledata(cfg.resample,x);');
            %-- save all
            if f == 1
                X(c,:,:)          = reshape(cell2mat(x.trial),[size(x.trial{1}) length(x.trial)]);
            else
                X(c,:,:)          = X(c,:,:)+reshape(cell2mat(x.trial),[size(x.trial{1}) length(x.trial)]);
            end
        end
    end
    
    %% mean across freqs
    X = X./nfreqs;
    
    %% concatenates
    datas{f} = X;
    
end
%% save data
save datas
    
    
    
