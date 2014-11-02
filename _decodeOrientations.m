%% Initialize

clear mvpas
path_images = [path 'images_acrossSubjs/'];
% use first subject data to retrieve time labels
if ~exist('cfg','var'),cfg=[];end
if ~exist('subject','var'),subject = SubjectsList{1}; end
if ~isfield(cfg,'dims')
    % load subjects details
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
    load(file_header, 'data');
    file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
    time    = data.time{1}; % in s
    
    % Specify time region of interest
    toi     = 1:length(time); % careful: this will be long.
    % to be faster -----------------------
    toi     = find(time>-.200,1):2:find(time>1.500,1);
    cfg.dims    = toi';
end


all_p_r_vis =[];
all_trial_prop_vis = [];
all_trial_prop_vis_a = [];
contrast='targetAngle';

for s = 1:length(SubjectsList)
    s
    %% INITIALIZE
    % load individual data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    
    % retrieve the results
    results_x = load([data_path 'mvpas/' subject '_preprocessed_targetAngle_SVR_tAll_x_results.mat']); % classifier (and file) name
    results_y = load([data_path 'mvpas/' subject '_preprocessed_targetAngle_SVR_tAll_y_results.mat']); % classifier (and file) name
    
    
    % Combine cos and sin predictions into angle
    px = squeeze(results_x.predict-2); % remove 2 to get back to normal (i.e. this is because of the abd scripting of the decoding pipeline that does not take y<=0, so i added to so that the sin and cos are between 2 and 3
    py = squeeze(results_y.predict-2);
    %---- gives the predicted angle theta and the predicted radius
    [angle_predicted radius_predicted] = cart2pol(px,py); 
    %---- computed from clf training set
    angle_truth = cart2pol(results_x.y-2, results_y.y-2); 
    %---- or (but its similar) here is an example of how it can be retrieved directly from trials.orientations
    angle_truth = mod(deg2rad([trials([trials.present]==1).orientation]*60-30-180), 2*pi)-pi; 
    
    %% EFFECT ACROSS ALL TRIALS
    % 1. ANGLE DIFFERENCE (between truth and prediction)
    angle_error = angle_predicted - repmat(angle_truth', [1 sz(angle_predicted, [2, 3])]);
    angle_error = abs(mod(angle_error+pi, 2*pi)-pi);
    angle_error = mod(angle_error+pi, 2*pi)-pi;
    
    
    % 2. CORRELATION ON ALL TRIALS
    % correlate predicted angle with true angle: not so great
    % combine the results to get a predicted angle
    % correlation between predicted angle and real angle
    [rho p] = circ_corrcc(repmat(theta', [1 sz(predict_theta, [2, 3])]),predict_theta);
    rhos(s,:,:) = rho;
    
    % 3. RELIAGNMENT AND DISTRIBUTION OF ALL TRIALS
    [trial_prop predict] = decode_reg2angle(...
        results_x,....
        results_y,...
        [trials([trials.present]==1).orientation]);
   
    % store peak of distribution
    all_predict(s,:,:) = squeeze(trial_prop(round(end/2),:,:));
    
   % 4. CIRC ANALYSES
   % r test: uniformity of angle distribution (valid everywhere)
   [p z] = circ_rtest(2*predict-pi);
   all_p_r(s,:,:) = squeeze(-log10(p));
   % v test: uniformity of angle distribution around a specified angle
   % (only valid on diagonal)
   [p z] = circ_vtest(2*predict-pi, 0);
   all_p_v(s,:,:) = squeeze(-log10(p));
   
   
    
    %% EFFECT PER CATEGORY
    % define division of interest
    sel = [trials.present]==1; % select only present trials
    angles = [trials(sel).orientation];
    visibilities = [trials(sel).response_visibilityCode];
    
    
    % VISIBILITY
    for vis = 1:4
        sel = visibilities==vis;
        try
            x.predict = results_x.predict(:,sel,:,:);
            y.predict = results_y.predict(:,sel,:,:);
            [trial_prop_vis predict_vis] = decode_reg2angle(x,y,angles(sel),10);
            all_trial_prop_vis(s,:,:,:,vis) = trial_prop_vis;
            [p z] = circ_rtest(2*predict_vis-pi);
            all_p_r_vis(s,:,:,vis) = squeeze(z);
        catch
            disp(['problem with s' num2str(s) ' vis' num2str(vis)])
            all_trial_prop_vis(s,:,:,:,vis) = NaN;
            all_p_r_vis(s,:,:,vis) = NaN;
        end
    end
    
    % VISIBILITY X ANGLE
    % mean tuning curve per angle so as to give equivalent weights to
    % each orientationm although some visibilities have heterogeneous
    % amounts of each orientation.
    for vis = 1:4
        for a = 1:6
            sel = visibilities==vis & angles==a;
            try
                x.predict = results_x.predict(:,sel,:,:);
                y.predict = results_y.predict(:,sel,:,:);
                [trial_prop_vis predict_vis] = decode_reg2angle(x,y,angles(sel),5);
                all_trial_prop_vis_a(s,:,:,:,vis, a) = trial_prop_vis;
            catch
                disp(['problem with s' num2str(s) ' vis' num2str(vis)])
                all_trial_prop_vis_a(s,:,:,:,vis, a) = NaN;
            end
        end
    end
    
    % mean angle error per condition
    for vis = 1:4
        for a = 1:6
            % select trials
            sel = visibilities==vis & angles==a;
            mean_angle_error(s,:,:,vis,a) = mean(angle_error(sel,:,:),1);
        end
    end
    
    % PREPARE ANOVA
    for vis = 1:4
        for a = 1:6
            sel = visibilities == vis & angles==a;
            
            for t = 1:length(toi)
                Y(s,vis,a,t) = mean(angle_error(sel,t,t));
            end
        end
    end
end

% run anova
p = [];
for t = 1:length(toi)
    [y groups] = prepare_anovan(Y(:,:,:,t));
    p(:,t) = anovan(y, groups, 'random', 1, 'model', 'linear', 'display', 'off');
end
plot(time(toi),-log10(p))


for vis =1:4,
    %     subplot(2,4,vis);
    %     imagesc(squeeze(nanmean(all_p_r_vis(:,:,:,vis)))); colorbar;
    %
    subplot(2,4,4+vis);
    imagesc(squeeze(nanmedian(all_trial_prop_vis(:,round(end/2),:,:,vis))), [.13 .24]);
    colorbar;
end

%% Plot prediction error distributions
close all
for vis=1:4
    % GAT for each level of visibility: peak of distribution
    subplot(2,4,vis);
    clim = [.13 .24];
    imagesc(time(toi),time(toi), ...
        squeeze(nanmedian(all_trial_prop_vis(:,round(end/2),:,:,vis))));
    
    % Tuning curver: distribution of prediction error on GAT diagonal
    subplot(2,4,4+vis);
    clear m
    for t = 1:length(toi)
        % mean across orientations
        m(:,t) = squeeze(nanmean(all_trial_prop_vis(:,:,t,t,vis)));
    end
    imagesc(time(toi),[], m)
end

%% Plot prediction error distributions after taking orientation bias into account.


%% Plot mean angle error computed after taking orientation bias into account.
close all
colors = colorGradient([1 0 0], [0 1 0], 4);
for vis=1:4
    % GAT for each level of visibility: angle error
    subplot(2,4,vis);
    imagesc(time(toi),time(toi), ...
        squeeze(-nanmean(nanmean(mean_angle_error(:,:,:,vis,:),5),1)));
    
    % Angle per visibility across time (diagonal)
    subplot(2,1,2);
    clear m
    for t = 1:length(toi)
        % mean across orientations
        m(:,t) = -squeeze(nanmean(mean_angle_error(:,t,t,vis,:),5));
    end
    hold on;
    plot_eb(time(toi),m, colors(vis,:))
end
