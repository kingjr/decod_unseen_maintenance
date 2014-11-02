clear all
close all
clc

%% libraries & toolboxes
switch 'niccolo'
    case 'niccolo'
    case 'jr'
        cd('/media/DATA/Pro/Projects/Paris/Orientation/Niccolo/script/201301');
        path = '/media/My Passport/Paris 16-01-2014/';
        addpath('/media/DATA/Pro/Toolbox/JR_toolbox/');
        addpath('/media/DATA/Pro/Toolbox/export_fig/');
        addpath('/media/DATA/Pro/Toolbox/fieldtrip/fieldtrip-20130225/'); ft_defaults;
        addpath('/media/DATA/Pro/Toolbox/circular_stats/');
    case 'imen'
        cd('/home/imen/Desktop/Backup_ICM/Projects_ICM/2014_meg_invisible_orientations/scripts/data_processing/');
        path = '/media/My Passport/Paris 16-01-2014/';
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/JR_toolbox/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/export_fig/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/fieldtrip-20130319/'); ft_defaults;
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/circular_stats/');
        addpath('/home/imen/Desktop/Backup_ICM/Projects_ICM/toolboxes/statistics/');
end


%% All subjects

SubjectsList ={'ak130184' 'el130086' 'ga130053' 'gm130176' 'hn120493'...
    'ia130315' 'jd110235' 'jm120476' 'ma130185' 'mc130295'...
    'mj130216' 'mr080072' 'oa130317' 'rg110386' 'sb120316'...
    'tc120199' 'ts130283' 'yp130276' 'av130322' 'ps120458'};

for s = 1:length(SubjectsList)
    try
        tic
        sbj_initials = SubjectsList{s};
        data_path = [path 'data/' sbj_initials '/'] ;
        
        %% Decoding:  of all orientations independently of contrasts and rating
        % generic function
        % select channels of interest
        selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);
        
        % load details
        file_behavior   = [data_path 'behavior/' sbj_initials '_fixed.mat'];
        file_header     = [data_path 'preprocessed/' sbj_initials '_preprocessed.mat'];
        file_binary     = [data_path 'preprocessed/' sbj_initials '_preprocessed.dat'];
        load(file_behavior, 'trials');
        load(file_header, 'data');
        time    = data.time{1}; % in s
        
        
        %% SVC classic
        toi     = 1:length(time);
        contrasts = {'lambda', 'responseButton','tilt', 'visibility',...
            'visibilityPresent', 'presentAbsent', 'accuracy'};
        
        
        % to be faster -----------------------
        toi     = find(time>-.100,1):4:find(time>1.500);
        contrasts = {'visibilityPresent','responseButton'};
        %-------------------------------------
        for c = 1:length(contrasts)
            contrast    = contrasts{c};
            cfg         = [];
            cfg.clf_type= 'SVC';
            cfg.dims    = toi';
            postproc_defineContrast;
            try postproc_decode; end
            %plot_decode;
        end
        
        %% SVR: classic
        contrasts   = {'targetAngle', 'probeAngle'};
        toi     = 1:length(time);
        
        % to be faster -----------------------
        toi     = find(time>-.100,1):4:find(time>1.500);
        contrasts   = {'targetAngle', 'probeAngle'};
        %-------------------------------------
        
        for c = 1:length(contrasts)
            %% SVR: trained at each time poitn
            contrast = contrasts{c};
            cfg         = [];
            cfg.clf_type= 'SVR';
            cfg.dims    = toi';
            postproc_defineContrast;
            postproc_decode;
            %plot_decode;
            
            %% SVR: slice of time
            %cfg.dims = find(time>=.176,1)+1;
            %cfg.dims_tg = toi;
            %postproc_defineContrast;
            %postproc_decode;
            %plot_decode;
            
            %% SVR: time generalization
            cfg         = [];
            cfg.clf_type= 'SVR';
            cfg.dims    = toi';
            cfg.dims_tg = repmat(toi,length(toi),1);
            postproc_defineContrast;
            postproc_decode;
            %plot_decode;
        end
        toc
    catch
        warning(['couldn''t do subject ' num2str(s) ]);
    end
end