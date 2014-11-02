%% select channels of interest
selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);

%% common parameters across classifiers
cfg.Xdim    = data.Xdim; % dimensions of binary files
cfg.features= selchan(data.label,'MEG')-1; % take all channels with an M
% cfg.fs      = .5; % feature selection: 25%
cfg.n_folds = 8; % number of fold
cfg.path    = [data_path 'mvpas/']; % where the results is save

%% specific parameters
tic
switch cfg.clf_type
    case 'SVC'
        cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime]; % classifier (and file) name
        cfg.run_svm = true;
        % do not recompute everything but only retrieve the codes
        if 0 ...~exist([cfg.path subject '_preprocessed_' cfg.namey '_results.mat'],'file'),
                cfg.load_results =false;
            %             MEEG = binload([path 'data/' subject '/preprocessed/' subject '_preprocessed.dat'], data.Xdim);
            results     = jr_classify(file_header,class,cfg);
        else
            results     = jr_classify(file_binary,class,cfg);
        end
    case 'SVR'
        switch contrast
            case {'targetAngle', 'probeAngle'}
                cfg.compute_probas = false;
                cfg.compute_predict = true;
                
                % predict x axis of orientation
                cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime '_x']; % classifier (and file) name
                results_x               = jr_classify(file_binary,x,cfg);
                
                % predict y axis of orientation
                cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime '_y']; % classifier (and file) name
                results_y               = jr_classify(file_binary,y,cfg);
            otherwise
                cfg.compute_probas = false;
                cfg.compute_predict = true;
                cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime]; % classifier (and file) name
                results               = jr_classify(file_binary,class,cfg);
        end
        
end
toc