%% Select subject
for sbj_number = 1:20
    
    sbj_initials = SubjectsList{sbj_number};
    data_path = [path 'data/' sbj_initials '/'] ;
    
    
    % identify files
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
        cfg.trialdef.prestim    = 1.200; % number, latency in seconds (optional);
        cfg.trialdef.poststim   = 2.000; % number, latency in seconds (optional);
        cfg                     = ft_definetrial(cfg);
        
        %% Read in the data
        cfg.channel     = {'MEG','MEGREF'};
        cfg.continuous  = 'yes';
        cfg.feedback    = 'gui';
        %     cfg.channel     = 1:10;
        data_            = ft_preprocessing(cfg);
        
        %% TF decomposition
        FOIs        = [6.5 10 13.5 17.5 25 65];
        for foi = FOIs
            method = 'mtm';
            cfg = [];
            cfg.keeptrials = 'yes';
            cfg.output     = 'pow';
            cfg.channel    = 'MEG';
            % frequencies of interest should be defined based on literature
            cfg.foi        = foi;
            cfg.toi        = -0.5:0.020:.800;
            switch method
                case 'mtm' % In theory slightly better for broadband signals, but slower to compute
                    cfg.method     = 'mtmconvol';
                    cfg.t_ftimwin  = 2 ./ cfg.foi; % duration of taper
                    cfg.tapsmofrq  = 0.5 * cfg.foi; % frequency of taper
                    % check that taper parameters are valid
                    K = 2.*cfg.t_ftimwin.*cfg.tapsmofrq-1;
                    if min(K)<0
                        warning('Tapers invalid!');
                    end
                case 'wavelet'
                    cfg.method = 'wavelet';
                    cfg.width = 7;
            end
            data     = ft_freqanalysis(cfg, data_);
            
            %% store temporary data
            save([data_path 'preprocessed/_' sbj_initials '_preprocessed_TFoi' num2str(f) '_' method '_' num2str(round(foi)) 'Hz.mat'], 'data'); % save details
            clear data
        end
    end
    
    
    %% Concatenate across runs
    clear datas
    for foi = FOIs
        for f = 1:length(files)
            f
            clear data
            load([data_path 'preprocessed/_' sbj_initials '_preprocessed_TFoi' num2str(f) '_' method '_' num2str(round(foi)) 'Hz.mat'], 'data'); % save details
            datas{f} = data;
            clear data
        end
        cfg             = [];
        cfg.parameter   = 'powspctrm';
        data            = ft_appendfreq(cfg,datas{:});
        data.grad       = datas{1}.grad; % fix fieldtrip bug
        clear datas;
        
        %% Baseline
        cfg              = [];
        cfg.baseline     = [-0.500 -0.250];
        cfg.baselinetype = 'db';
        data             = ft_freqbaseline(cfg, data);
        
        %% Plot to check
        if 0
            cfg.showlabels   = 'yes';
            cfg.layout       = 'neuromag306mag';
            ft_multiplotTFR(cfg, data)
        end
        %% Save data
        save([data_path 'preprocessed/' sbj_initials '_preprocessed_Tfoi_' method '_' num2str(round(foi)) 'Hz.mat'], 'data'); % save details
        
    end

    %% Delete temporary files
    if 1
        % list temporary files
        tmp_files = dir([data_path 'preprocessed/_' sbj_initials '_preprocessed_TFoi*.mat']);
        % delete them
        for f = 1:length(tmp_files)
            delete([data_path 'preprocessed/' tmp_files(f).name ]);
        end
    end
end
