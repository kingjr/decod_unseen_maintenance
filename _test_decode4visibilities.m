across_subjects =[];
for s = 1:20
    % select subject and details
    subject = SubjectsList{s};
    
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
    toi     = find(time>-.050,1):5:find(time>.700,1);
    
    %% SVC classic
    cfg             = [];
    cfg.contrast    = '4visibilitiesPresent';
    cfg.clf_type    = 'SVC';
    cfg.dims        = toi';
    cfg.gentime     = '';
    [class_x ~]     = decode_defineContrast(cfg,trials);
    decode_run;
    
    %% SVR
    cfg             = [];
    cfg.contrast    = '4visibilitiesPresent';
    cfg.clf_type    = 'SVR';
    cfg.dims        = toi';
    cfg.gentime     = '';
    [class_x ~]     = decode_defineContrast(cfg,trials);
    decode_run;
end

%% load results
for s = 1:20
    subject = SubjectsList{s};
    results=load([path 'data/' subject '/mvpas/' subject '_preprocessed_4visibilitiesPresent_SVC_results.mat']);
    for vis = 1:4
        across_subjects(s,:,vis) = mean(results.probas(1, results.y==vis, :, 1, vis),2);
        across_subjects(s,:,vis) = across_subjects(s,:,vis) - mean(across_subjects(s,:,vis),2);
    end
end

figure();
colors = colorGradient([1, 0, 0], [0, 1, 0], 4);
for vis=1:4
    plot_eb(time(toi), across_subjects(:,:,vis), colors(vis, :));
    hold on;
end

%% same with svr

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
across_subjects =[];
for s = 1:20
    %% select subject and details
    subject = SubjectsList{s};
    
    
    
    %% load subjects details
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
    load(file_header, 'data');
    file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
    time    = data.time{1}; % in secs
    
    %% Specify time region of interest
    toi     = find(time>-.200,1):2:find(time>1.500,1);
    
    
    %% SVR: trained at each time point
    cfg             = [];
    cfg.contrast    = '4visibilitiesPresent';
    cfg.clf_type    = 'SVR';
    cfg.dims        = toi';
    cfg.gentime     = '';
    [class_x ~]     = decode_defineContrast(cfg,trials);
    decode_run;
    %     plot_decode;
    
end


%% load results
across_subjects = [];
across_subjects_c = [];

for s = 1:20
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    results=load([path 'data/' subject '/mvpas/' subject '_preprocessed_4visibilitiesPresent_SVR_results.mat']);
    for vis = 1:4
        % mean prediction for each visibility
        across_subjects(s,:,vis) = mean(results.predict(1, results.y==vis, :, 1),2);
        % find the contrast for each trial
        yc = [trials.contrast]'-1;
        yc(not(results.y_all>0)) = []; % remove the absent trial from these contrasts
        % gather mean results
        for c = 1:3
            across_subjects_c(s,:,vis, c) = nanmean(results.predict(1, results.y==vis & yc==c, :, 1),2);
        end
    end
end

figure();
colors = colorGradient([1, 0, 0], [0, 1, 0], 4);
for vis=1:4
    [hl(vis) hf(vis)] = plot_eb(time(toi), across_subjects(:,:,vis), colors(vis, :));
    hold on;
end
legend(cell2mat(hl),'invisible','brief glimpse','almost clear','clear','location','southwest')
set(gca,'fontsize',15);
xlabel('time');ylabel('SVR visibility prediction');
title('present trials only')
plot2svg([im_path '/4visibilitiesPresent/visibilityPrediction'])

figure();set(gcf,'position',get(0,'screensize'))
for c = 1:3
    subplot(3,1,c);
    for vis=1:4
        [hl(vis) hf(vis)] = plot_eb(time(toi), across_subjects_c(:,:,vis,c), colors(vis, :));
        hold on;
        legend(cell2mat(hl),'invisible','brief glimpse','almost clear','clear', ...
            'location','NorthEastOutside')
        set(gca,'fontsize',15);
        xlabel('time');ylabel('SVR visibility prediction');
        title(['contrast ' num2str(contrasts(c+1))])
    end
end
plot2svg([im_path '/4visibilitiesPresent/visibilityPrediction_by_contrast'])


%% anova
ntime = length(toi);
nfactor = 3; % subjects x visibility x contrast
P = zeros(nfactor,ntime);
for t = 1:ntime
    [Y, GROUP] = prepare_anovan(squeeze(across_subjects_c(:,t,:,:)));
    P(:,t) = anovan(Y, GROUP, 'random', 1, 'model', 'linear', 'display', 'off');
end
plot(time(toi),-log10(P(2:end,:)),'linewidth',3)
set(gca,'fontsize',15)
xlabel('time');ylabel('log_1_0 p-value');
legend('visibility','contrast','location','northwest')
plot2svg([im_path '/4visibilitiesPresent/anova_contrast_vs_visibility'])



%% plot 4 visibility present tAll
%% load results
across_subjects = [];
across_subjects_c = [];

for s = 1:20
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    results=load([path 'data/' subject '/mvpas/' subject '_preprocessed_4visibilitiesPresent_SVR_tAll_results.mat']);
    
    % compute error
    tmp = [squeeze([results.predict]) - repmat(double([results.y]'),1,length(toi),length(toi))].^2;
    across_subjects(s,:,:) = squeeze(mean(sqrt(tmp)));
    
end

imagesc(time(toi),time(toi),squeeze(mean(across_subjects)))
set(gca,'ydir','normal','fontsize',15)
colorbar
title('mean square root error')
xlabel('test time');ylabel('train time');

plot2svg([im_path '/SVR_4visibilitiesPresent_tAll'])


