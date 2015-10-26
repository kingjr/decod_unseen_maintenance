
%% common paramters across classifiers
cfg.Xdim    = data.Xdim; % dimensions of binary files
cfg.features= selchan(data.label,'MEG')-1; % take all channels with an M
% cfg.fs      = .5; % feature selection: 25%
cfg.n_folds = 8; % number of fold
cfg.path    = [data_path 'mvpas/']; % where the results is save

if isfield(cfg, 'dims') && length(cfg.dims)==1
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = ['_t' num2str(round(1000*time(cfg.dims)))];
elseif ... % it is a full time generalization matrix
        isfield(cfg, 'dims_tg') &&...
        ((size(cfg.dims_tg,1)>1 && size(cfg.dims_tg,2)>1)  || cfg.dims_tg<0)
    cfg.wsize   = 1;
    cfg.gentime = '_tAll';
else
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = '';
end


%% specific parameters
tic
switch cfg.clf_type
    case 'SVC'
        cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime]; % classifier (and file) name
        results     = jr_classify(file_binary,class,cfg);
    case 'SVR'
        cfg.compute_probas = false;
        cfg.compute_predict = true;
        
        % predict x axis
        cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime '_x']; % classifier (and file) name
        results_x               = jr_classify(file_binary,x,cfg);
        
        % predict y axis
        cfg.namey   = [contrast '_' cfg.clf_type cfg.gentime '_y']; % classifier (and file) name
        results_y               = jr_classify(file_binary,y,cfg);
end
toc