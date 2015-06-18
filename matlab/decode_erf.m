%% load subjects details
data_path = [path 'data/' subject '/'] ;
file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
load(file_behavior, 'trials');
file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
load(file_header, 'data');
file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
time    = data.time{1}; % in secs

%% Specify time region of interest
% to be faster -----------------------
toi     = find(time>-.200,1):2:find(time>1.500,1);


%% SVC classic
contrasts = {'targetAngle', 'probeAngle','lambda', 'responseButton','tilt', 'visibility',...
    'visibilityPresent', 'presentAbsent', 'accuracy'}; % may increase over time
for c = 2%1:length(contrasts)
    disp(['SVC: ' num2str(c)])
    
    cfg             = [];
    cfg.contrast    = contrasts{c};
    cfg.clf_type    = 'SVC';
    cfg.dims        = toi';
    cfg.gentime     = '';
    [class ~]       = decode_defineContrast(cfg,trials);
    decode_run;
%     plot_decode;
    
    % clear ram memory after each classifier
    system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')
    if 0
    %% SVC: time generalization
    cfg             = [];
    cfg.contrast    = contrasts{c};
    cfg.clf_type    = 'SVC';
    cfg.dims        = toi';
    cfg.dims_tg     = repmat(toi,length(toi),1);
    cfg.gentime     = '_tAll';
    cfg.wsize       = 4; % careful...
    [class ~]       = decode_defineContrast(cfg,trials);
    decode_run;
    
    % clear ram memory after each classifier
    system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')
    
%     plot_decode;
    end
end

%% SVR: classic
% here go all contrasts whose variables are in principle continuous like
% angle and visibility ratings.
if 0
contrasts   = {'targetAngle','probeAngle', '4visibilitiesPresent' 'target_contrast'}; 
for c = 1:length(contrasts)
    disp(['SVR: ' num2str(c)])
%     %% SVR: trained at each time point
    
%     cfg                   = [];
%     cfg.contrast          = contrasts{c};
%     cfg.clf_type          = 'SVR';
%     cfg.dims              = toi';
%     cfg.wsize             = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
%     cfg.gentime           = '';
%     [class_x class_y]     = decode_defineContrast(cfg,trials);
%     decode_run;
% %     plot_decode;
%     
%     %% SVR: slice of time t = .176
%     cfg                   = [];
%     cfg.contrast          = contrasts{c};
%     cfg.clf_type          = 'SVR';
%     cfg.dims              = find(time>=.176,1)+1;
%     cfg.dims_tg           = toi;
%     cfg.wsize             = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
%     cfg.gentime           = ['_t' num2str(round(1000*time(cfg.dims)))];
%     [class_x class_y]     = decode_defineContrast(cfg,trials);
%     decode_run;
% %     plot_decode;
%     
%     %% SVR: slice of time t = .300
%     cfg                   = [];
%     cfg.contrast          = contrasts{c};
%     cfg.clf_type          = 'SVR';
%     cfg.dims              = find(time>=.300,1)+1;
%     cfg.dims_tg           = toi;
%     cfg.wsize             = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
%     cfg.gentime           = ['_t' num2str(round(1000*time(cfg.dims)))];
%     [class_x class_y]     = decode_defineContrast(cfg,trials);
%     decode_run;
% %     plot_decode;
% 
%     %% SVR: slice of time t = .970 for probe only
%     cfg                   = [];
%     cfg.contrast          = contrasts{c};
%     cfg.clf_type          = 'SVR';
%     cfg.dims              = find(time>=.970,1)+1;
%     cfg.dims_tg           = toi;
%     cfg.wsize             = 4; % don't go beyond 32 ms if possible (wsize=8 if 256Hz)
%     cfg.gentime           = ['_t' num2str(round(1000*time(cfg.dims)))];
%     [class_x class_y]     = decode_defineContrast(cfg,trials);
%     decode_run;
% %     plot_decode;
% 
%     %% SVR: slice etc, depending on contrast
    
    %% SVR: time generalization
    cfg             = [];
    cfg.contrast    = contrasts{c};
    cfg.clf_type    = 'SVR';
    cfg.dims        = toi';
    cfg.dims_tg     = repmat(toi,length(toi),1);
    cfg.wsize       = 4; % careful...
    cfg.gentime     = '_tAll';
    
    [class_x class_y] = decode_defineContrast(cfg,trials);
    decode_run;
    
    % clear ram memory after each classifier
    system('sync && echo 3 | sudo tee /proc/sys/vm/drop_caches')
    
%     plot_decode;
end
end