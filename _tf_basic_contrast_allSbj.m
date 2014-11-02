clear datas
for s = 1:length(SubjectsList)
    s
    %% Load data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    file_ft         = [data_path 'preprocessed/' subject '_preprocessed_Tfoi_mtm.mat'];
    load(file_ft, 'data');
    load(file_behavior, 'trials');
    time = data.time;
    freq = data.freq;
    visibility = [trials.response_visibilityCode];
    present = [trials.present];
    orientation = [trials.orientation];orientations(~present) = NaN;
    
    % Average of present and absent topographis across trial for each
    % frequency and channel; store across subjects
    if s == 1, datas = rmfield(data,'powspctrm');end
    datas.powspctrm(s,:,:,:,1) = mean(data.powspctrm(present, :, :, :), 1);
    datas.powspctrm(s,:,:,:,2) = mean(data.powspctrm(~present, :, :, :), 1);
end

%% Power across time (mean across all channels)
figure(1);clf;set(gcf,'color','w');
for f =1:length(freq)
    subplot(length(freq),1,f);
    plot_eb(time,squeeze(mean(datas.powspctrm(:,:,f,:,1), 2)), [0 1 0]);
    hold on
    plot_eb(time,squeeze(mean(datas.powspctrm(:,:,f,:,2), 2)), [1 0 0]);
    axis tight;title(freq(f));box off
end

%% Topography present vs absent
for f = 1:length(freq)
    figure(f);clf;set(gcf,'color','w');
    % select one frequency power
    data_ = rmfield(datas, 'powspctrm');
    data_.freq = datas.freq(f);
    data_.powspctrm = datas.powspctrm(:,:,f,:, :);
    
    % define contrast
    data_.powspctrm = ...
        mean(data_.powspctrm(:, :, :, :, 1)) - ...
        mean(data_.powspctrm(:, :, :, :, 2));
    
    cfg = [];
    cfg.layout = 'neuromag306mag';
    % force similar scale across topographies by taking minmax of mean
    % power across trials
    cfg.zlim = max(abs(prctile(reshape(mean(data_.powspctrm,1),[1 306*length(time)]), [.5 99.5])));
    cfg.zlim = [-cfg.zlim cfg.zlim];
    cfg.comment = 'no';
    cfg.style = 'straight';
    cfg.marker = 'off';
    toi = -0.150:.050:.800;
    for t = 1:(length(toi)-1)
        cfg.xlim = [toi(t) toi(t+1)];
        subplot(4,5,t);
        ft_topoplotTFR(cfg,data_);
        title(round(toi([t t+1])*1000));
%         pause;
    end
    set(gcf, 'name', num2str(freq(f)));
    colormap([...
        colorGradient([0 0 1], [1 1 1], 32 ); ...
        colorGradient([1 1 1], [1 0 0], 32 )]); 
end

%% Decoding (diagonal)
clear all_trial_prop all_vstat
for s = 1:length(SubjectsList)
    s
    %% Load data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    file_ft         = [data_path 'preprocessed/' subject '_preprocessed_Tfoi_mtm.mat'];
    load(file_ft, 'data');
    load(file_behavior, 'trials');
    time = data.time;
    freq = data.freq;
    visibility = [trials.response_visibilityCode];
    present = [trials.present];
    orientation = [trials.orientation];orientations(~present) = NaN;
    
    % for f = 1:length(freq);
    f = 3;
    X = squeeze(data.powspctrm(:,:,f,:)); % decode only on one frequency
   
    % define contrast
    cfg         = [];
    cfg.clf_type= 'SVR';
    contrast = 'targetAngle';
    decode_defineContrast;
    
    % parameters
    toi = find(time>-.050,1):size(X,3); % decimate time samples
    coi = 3:3:size(X,2); % only magnetometers
    
    cfg.compute_probas = false;
    cfg.compute_predict = true;
    
    % predict x axis of orientation
    results_x               = jr_classify(X(:,coi,toi),x,cfg);
    % predict y axis of orientation
    results_y               = jr_classify(X(:,coi,toi),y,cfg);
    
    %% get angle distance
    targetAngles = [trials.orientation]';
    targetAngles([trials.present]==0) = [];
    [trial_prop predict] = decode_reg2angle(results_x,results_y,targetAngles,20);
    
    %% stats
    [p vstat] = circ_vtest(2*predict-pi,0);
    
    % end
    
    all_trial_prop(s, :, :) = trial_prop;
    all_vstat(s, :) = vstat;
end


% figure
colors = colorGradient([0 0 0], [0 1 0], length(freq));

figure(1);set(gcf,'color', 'w');
subplot(length(freq),1,f);
imagesc(time(toi),[],squeeze(mean(all_trial_prop)))

figure(2);plot_eb(time(toi),all_vstat, 'color', colors(f, :));
hold on;


%% Generalization Across Time
clear all_trial_prop all_rstat
for s = 1:length(SubjectsList)
    s
    %% Load data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    file_ft         = [data_path 'preprocessed/' subject '_preprocessed_Tfoi_mtm.mat'];
    load(file_ft, 'data');
    load(file_behavior, 'trials');
    time = data.time;
    freq = data.freq;
    visibility = [trials.response_visibilityCode];
    present = [trials.present];
    orientation = [trials.orientation];orientations(~present) = NaN;
    
    % for f = 1:length(freq);
    f = 4;
    X = squeeze(data.powspctrm(:,:,f,:)); % decode only on one frequency
    % define contrast
    cfg         = [];
    cfg.clf_type= 'SVR';
    contrast = 'targetAngle';
    decode_defineContrast;
    
    % parameters
    toi = find(time>-.050,1):2:size(X,3); % decimate time samples
    coi = 3:3:size(X,2); % only magnetometers
    
    cfg.compute_probas = false;
    cfg.compute_predict = true;
    cfg.dims = toi';
    cfg.dims_tg = repmat(toi,length(toi),1);
    
    % predict x axis of orientation
    results_x               = jr_classify(X(:,coi,:),x,cfg);
    % predict y axis of orientation
    results_y               = jr_classify(X(:,coi,:),y,cfg);
    
    %% get angle distance
    targetAngles = [trials.orientation]';
    targetAngles([trials.present]==0) = [];
    [trial_prop predict] = decode_reg2angle(results_x,results_y,targetAngles,20);
    
    %% stats
    [p rstat] = circ_rtest(2*predict-pi);
    
    % end
    all_rstat(s, :, :) = rstat;
end

%% figure
% train from ERF and generelize to alpha.
figure();
imagesc(time(toi),time(toi),squeeze(mean(all_rstat)));
set(gca,'ydir', 'normal');




%% Generalize across conditions
clear all_vstat_tf all_vstat_erf
for s = 1:length(SubjectsList)
    s
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    
    %% LOAD DATA
    % Load behavior
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');

    present = [trials.present];
    orientations = [trials(present).orientation];
    
    % Parameter for decimating for memory purposes
    coi = 3:2:size(X,2); % only magnetometers
    
    % Load TF data
    file_ft         = [data_path 'preprocessed/' subject '_preprocessed_Tfoi_mtm.mat'];
    load(file_ft, 'data');
    time_tf = data.time;
    freq = data.freq;
    X_tf = squeeze(data.powspctrm(present,coi,3,:)); % decimate tf data
    clear data;
    
    % Load ERF
    file_ft         = [data_path 'preprocessed/' subject '_preprocessed'];
    load([file_ft '.mat'], 'data'); % fieldtrip structure
    time_erf = data.time{1};
    X_erf = binload([file_ft '.dat'], data.Xdim); % binary data (normally in data.trial)
    toi_erf = find(time_erf>-.050,1)+cumsum(2*ones(1,size(X_tf,3))); %make sure that the ERF and TF have same number of time sample (which is stupid, but will be corrected in the next script version)
    X_erf = X_erf(present,coi,toi_erf); % decimate erf data
    clear data;
    
    % concatenate TF and ERF
    X_both = cat(1,X_erf, X_tf);
    
    % define contrast
    cfg         = [];
    cfg.clf_type= 'SVR';
    contrast = 'targetAngle';
    decode_defineContrast; % this create two regressors corresponding to cos(orientation) sin(orientation) so as to be usable for SVR
    % As we have twice the data, we double the regressors
    reg_x = cat(1,x(present)', -x(present)'); % here -corresponds to the generalization set
    reg_y = cat(1,y(present)', -y(present)'); % 
    
    % parameters
    cfg.compute_probas = false;
    cfg.compute_predict = true;
    toi = 1:length(time_tf);
    cfg.dims = toi';
    cfg.dims_tg = repmat(toi,length(toi),1);
  
    % predict x axis of orientation
    results_x               = jr_classify(X_both,reg_x,cfg);
    % predict y axis of orientation
    results_y               = jr_classify(X_both,reg_y,cfg);
    
    %% get angle distance
    predict_erf = decode_reg2angle_predict(...
        results_x.predict,...
        results_y.predict,...
        orientations);
    predict_tf = decode_reg2angle_predict(...
        mean(results_x.predictg,5),... % here average across prediction emitted from each fold. Ask JR if you actually include this analysis in the paper, as we should only take the crossvalidated results for the tf.
        mean(results_y.predictg,5),...
        orientations);
    
    %% stats
    [p vstat_erf] = circ_rtest(2*predict_erf-pi);
    [p vstat_tf] = circ_rtest(2*predict_tf-pi);
    
    all_vstat_erf(s, :, :) = vstat_erf;
    all_vstat_tf(s, :, :) = vstat_tf;
end


