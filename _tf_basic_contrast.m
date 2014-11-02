%% Load data
subject = SubjectsList{10};
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

%% Power across time (mean across all channels)
figure(1);clf;set(gcf,'color','w');
for f =1:length(freq)
    subplot(length(freq),1,f);
    plot_eb(time,squeeze(mean(data.powspctrm(present,:,f,:), 2)), [0 1 0]);
    hold on
    plot_eb(time,squeeze(mean(data.powspctrm(not(present),:,f,:), 2)), [1 0 0]);
    axis tight;title(freq(f,:));box off
end

%% Topography present vs absent
for f = 1:length(freq)
    figure(f);clf;set(gcf,'color','w');
    % select one frequency power
    data_ = rmfield(data, 'powspctrm');
    data_.freq = data.freq(f);
    data_.powspctrm = data.powspctrm(:,:,f,:);
    
    % define contrast
    data_.powspctrm = ...
        mean(data_.powspctrm(present, :, :, :)) - ...
        mean(data_.powspctrm(not(present), :, :, :));
    
    cfg = [];
    cfg.layout = 'neuromag306mag';
    % force similar scale across topographies by taking minmax of mean
    % power across trials
    cfg.zlim = minmax(reshape(mean(data_.powspctrm,1),[1 306*length(time)]));
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
end


%% Test of decoding
% present versus absent
figure();clf;set(gcf,'color', 'w');
colors = colorGradient([0 0 0], [0 1 0], length(freq));
for f = 1:length(freq);
    X = squeeze(data.powspctrm(:,:,f,:)); % decode only on one frequency
    y = present+1;
    
    sel = 1:length(y); % only take half of the trials
    toi = 1:2:size(X,3); % decimate time samples
    coi = 3:3:size(X,2); % only magnetometers
    cfg = [];
    % cfg.compute_predict = true;
    % cfg.compute_probas = false;
    results=jr_classify(X(sel,coi,toi),y(sel),cfg);
    % acc= mean(squeeze(results.predict) == repmat(results.y, [1, length(toi)]));
    % plot(time(toi),acc)
    auc = colAUC(squeeze(results.probas(1,:,:,1,1)), results.y);
    plot(time(toi),auc, 'linewidth', 2, 'color', colors(f, :));
    hold on;
end

% orientation SVR
colors = colorGradient([0 0 0], [0 1 0], length(freq));
for f = 1:length(freq);
    X = squeeze(data.powspctrm(:,:,f,:)); % decode only on one frequency
    y = present+1;
    % define contrast
    cfg         = [];
    cfg.clf_type= 'SVR';
    contrast = 'targetAngle';
    decode_defineContrast;
    
    % parameters
    toi = 1:size(X,3); % decimate time samples
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
    figure(1);set(gcf,'color', 'w');
    subplot(length(freq),1,f);
    imagesc(time(toi),[],trial_prop)
    
    %% stats
    [p vstat] = circ_vtest(2*predict-pi,0);
    figure(2);plot(time(toi),vstat, 'linewidth', 2, 'color', colors(f, :));
    hold on;
end

%% orientation GAT
f = 6;
X = squeeze(data.powspctrm(:,:,f,:)); % decode only on one frequency
y = present+1;
% define contrast
cfg         = [];
cfg.clf_type= 'SVR';
contrast = 'targetAngle';
decode_defineContrast;

% parameters
toi = 1:2:size(X,3); % decimate time samples
coi = 3:3:size(X,2); % only magnetometers
cfg.compute_probas = false;
cfg.compute_predict = true;
cfg.dims    = toi';
cfg.dims_tg = repmat(toi,length(toi),1);

% predict both axes of orientation
results_x = jr_classify(X(:,coi,:),x,cfg);
results_y = jr_classify(X(:,coi,:),y,cfg);

% combine results
targetAngles = [trials.orientation]';
targetAngles([trials.present]==0) = [];
[trial_prop predict] = decode_reg2angle(results_x,results_y,targetAngles,20);

% stats
[p rstat] = circ_rtest(2*predict-pi);

% plot
figure(3);set(gcf,'color', 'w');
imagesc(time(toi),time(toi),rstat);
axis image
set(gca,'ydir', 'normal');