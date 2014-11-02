%% load subjects details
data_path = [path 'data/' subject '/'] ;
file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
load(file_behavior, 'trials');
file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
load(file_header, 'data');
file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
time    = data.time{1}; % in secs

%% Specify time region of interest
toi     = 1:length(time); % careful: this will be long. otherwise, only take relevant data points
% to be faster -----------------------
toi     = find(time>-.200,1):2:find(time>1.500,1);


%% SVC classic
contrasts = {'targetAngle', 'probeAngle','lambda', 'responseButton','tilt', 'visibility',...
    'visibilityPresent', 'presentAbsent', 'accuracy'}; % may increase over time
for c = 1:length(contrasts)
    contrast    = contrasts{c};
    cfg         = [];
    cfg.clf_type= 'SVC';
    cfg.dims    = toi';
    cfg.gentime = '';
    decode_defineContrast;
    decode_run;
%     plot_decode;
end

%% SVR: classic

contrasts   = {'targetAngle','probeAngle', '4visibilitiesPresent'}; 
for c = 1:length(contrasts)
    %% SVR: trained at each time point
    contrast = contrasts{c};
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.dims    = toi';
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = '';
    decode_defineContrast;
    decode_run;
%     plot_decode;
    
    %% SVR: slice of time t = .176
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.dims    = find(time>=.176,1)+1;
    cfg.dims_tg = toi;
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = ['_t' num2str(round(1000*time(cfg.dims)))];
    decode_defineContrast;
    decode_run;
%     plot_decode;
    
    %% SVR: slice of time t = .300
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.dims    = find(time>=.300,1)+1;
    cfg.dims_tg = toi;
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = ['_t' num2str(round(1000*time(cfg.dims)))];
    decode_defineContrast;
    decode_run;
%     plot_decode;

    %% SVR: slice of time t = .970 for probe only
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.dims    = find(time>=.970,1)+1;
    cfg.dims_tg = toi;
    cfg.wsize   = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
    cfg.gentime = ['_t' num2str(round(1000*time(cfg.dims)))];
    decode_defineContrast;
    decode_run;
%     plot_decode;

    %% SVR: slice etc, depending on contrast
    
    %% SVR: time generalization
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.dims    = toi';
    cfg.dims_tg = repmat(toi,length(toi),1);
    cfg.wsize   = 1;
    cfg.gentime = '_tAll';
    
    decode_defineContrast;
    decode_run;
%     plot_decode;
end
