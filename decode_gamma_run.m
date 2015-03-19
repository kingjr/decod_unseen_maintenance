%% select channels of interest
selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);

%% common parameters across classifiers
cfg.features    = selchan(data.label,'MEG')-1; % take all channels with an M
% cfg.fs        = .5; % feature selection: 25%
cfg.n_folds     = 8; % number of fold
cfg.path        = [data_path 'mvpas/timeFreq/']; % where the results is save

%% specific parameters
tic
switch cfg.clf_type
    case 'SVC'
        cfg.nameX   = [subject '_preprocessed_' num2str(round(foi)) 'Hz'];
        cfg.namey   = [cfg.contrast '_' cfg.clf_type cfg.gentime]; % classifier (and file) name
        cfg.run_svm = true;
        % do not recompute everything but only retrieve the codes
        if 0 %~exist([cfg.path subject '_preprocessed_' cfg.namey '_results.mat'],'file'),
                cfg.load_results = false;
            %             MEEG = binload([path 'data/' subject '/preprocessed/' subject '_preprocessed.dat'], data.Xdim);
            results     = jr_classify(file_header,class,cfg);
        else
            results     = jr_classify(squeeze(data.powspctrm),class,cfg);
        end
    case 'SVR'
        cfg.load_results = false;
        switch cfg.contrast
            case {'targetAngle', 'probeAngle'}
                cfg.nameX   = [subject '_preprocessed_' num2str(round(foi)) 'Hz'];
                cfg.compute_probas  = false;
                cfg.compute_predict = true;
                
                % predict x axis of orientation
                cfg.namey   = [cfg.contrast '_' cfg.clf_type cfg.gentime '_x']; % classifier (and file) name
                results_x   = jr_classify(squeeze(data.powspctrm),class_x,cfg);
                
                % predict y axis of orientation
                cfg.namey   = [cfg.contrast '_' cfg.clf_type cfg.gentime '_y']; % classifier (and file) name
                results_y   = jr_classify(squeeze(data.powspctrm),class_y,cfg);
            otherwise
                cfg.compute_probas  = false;
                cfg.compute_predict = true;
                cfg.namey           = [cfg.contrast '_' cfg.clf_type cfg.gentime]; % classifier (and file) name
                results             = jr_classify(squeeze(data.powspctrm),class_x,cfg);
        end
        
end
toc