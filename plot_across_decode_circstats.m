%% load time and label data from a standard subject
subject = SubjectsList{1};
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
labels = data.label;
time = data.time{1};

%% define images path 
path_images = [path 'images_acrossSubjs'];
im_path = '/home/niccolo/vboxshared/DOCUP/orientationDecoding';

%% all subjects target angle decoding (diagonal only, thus redundant: it can be inferred from tAll)
%     clear all_trialProp* all_p all_v_vis
%     for s = length(SubjectsList):-1:1
%         s
%         subject = SubjectsList{s};
%         %% load data
%         load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
%         switch models{mdl}
%             case 'angle'
%                 results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_x_results.mat'],'predict', 'y', 'dims');
%                 results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_y_results.mat'],'predict', 'y');
%             case 'probe'
%         end
%         toi = time(results_x.dims);
%         
%         %% get angle distance
%         targetAngles = [trials.orientation]';
%         targetAngles([trials.present]==0) = [];
%         [trial_prop predict] = recombine_SVR_prediction(results_x,results_y,targetAngles,20);
%         all_trialProp(:,:,s) = trial_prop;
%         
%         %% stats
%         % test of non uniformity
%         %[p z] = circ_rtest(2*predict-pi);
%         % test of non uniformity and mean 0
%         %XXXXX circ_corrcc(deg2rad
%         
%         [p vstat] = circ_vtest(2*predict-pi,0);
%         all_p(:,:,s) = p;
%         
%         %% sort by visibility
%         visibility = [trials.response_visibilityCode]';
%         visibility([trials.present]==0) = [];
%         probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
%         %probeAngles([trials.present]==0) = []; % only if defineContrast
%         %hasn't removed them yet
%         
%         for vis = 4:-1:1
%             sel = targetAngles;
%             sel = probeAngles;
%             sel(visibility~=vis) = NaN;
%             [trial_prop_vis predict_vis] = recombine_SVR_prediction(results_x,results_y,sel,6);
%             all_trialProp_vis(:,:,vis,s) = trial_prop_vis;
%             % stats
%             [p vstat] = circ_vtest(2*predict(visibility==vis,:)-pi,0);
%             n = sum(visibility==vis);
%             all_v_vis(s,:,vis) = vstat/n;
%             n_vis(s,vis) = n;
%             all_p_vis(s,:,vis) = p;
%         end
%     end
%     imagesc(toi,[],mean(all_trialProp,3));
%     clf;hold on;
%     colors = colorGradient([1 0 0], [0 1 0],4);
%     for v = 1:4
%         %subplot(4,1,v);
%         %imagesc(toi,[],mean(all_trialProp_vis(:,:,v,:),4));
%         [hl(v)]=plot_eb(toi,all_v_vis(:,:,v),colors(v,:));
%     end
%     set(gca,'FontSize',24)
%     title('test of non-uniformity with specified direction(v stat)');
%     legend(cell2mat(hl),'vis0','vis1','vis2','vis3')
%     set(gcf,'Position',get(0,'ScreenSize'))
%     ylabel('v-stat ');xlabel('time');
%     % saveas(gcf,[path_images '/SVR_TAngle_Vtest_vis.jpeg']);
%     
%     % visualize p values
%     figure,
%     temp= -log10(all_p_vis);
%     imagesc(toi,1:4,squeeze(median(temp))'),colorbar
%     set(gca,'FontSize',24,'YTick',[1:4],'YTickLabel',{'vis0' 'vis1' 'vis2' 'vis3'})
%     set(gcf,'Position',get(0,'ScreenSize'))
%     xlabel('time');title('median -log10 (p-val)')
%     % saveas(gcf,[path_images '/SVR_TAngle_pval_vis.jpeg']);


%% all subjects target and probe angle decoding generalization across time
% XXX Externalize function so that it become a generic prediction,
% independently of whether models are fitted on probe or on target.

models = {'target' 'probe'};

for mdl = 1:length(models)
    models{mdl}
    
    clear all_trialProp* all_p_* all_z_* rhos 
    for s = length(SubjectsList):-1:1
        subject = SubjectsList{s};
        s
        %% load data
        load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
        results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_' models{mdl} 'Angle_SVR_tAll_x_results.mat'],'predict', 'y','dims');
        results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_' models{mdl} 'Angle_SVR_tAll_y_results.mat'],'predict', 'y');
        toi = time(results_x.dims);

        % Combine cos and sin predictions into angle
        px = squeeze(results_x.predict-2); % remove 2 to get back to normal (i.e. this is because of the abd scripting of the decoding pipeline that does not take y<=0, so I added 2 so that the sin and cos are between 2 and 3
        py = squeeze(results_y.predict-2);
        %---- gives the predicted angle theta and the predicted radius
        [angle_predicted radius_predicted] = cart2pol(px,py);
        %---- computed real angle from clf training set
        angle_truth = cart2pol(results_x.y-2, results_y.y-2);

        %% get angle and genAngle
        switch models{mdl}
            case 'target'
                % target
                angles = [trials.orientation]';
                angles([trials.present]==0) = [];
                
                % generalize to realigned probe
                genAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
                genAngles([trials.present]==0) = [];
            case 'probe'
                %probe
                angles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
                angles([trials.present]==0) = [];
                
                % generalize to target
                genAngles = [trials.orientation]';
                genAngles([trials.present]==0) = [];
        end
        
        %% below different measures are reviewed. Source:
        % _decodeOrientations.m
        
        % 1. ANGLE DIFFERENCE (between truth and prediction)
        angle_error = angle_predicted - repmat(angle_truth', [1 sz(angle_predicted, [2, 3])]);
        angle_error = abs(mod(angle_error+pi, 2*pi)-pi);
        angle_error = mod(angle_error+pi, 2*pi)-pi;
        
        %----- 2. CORRELATION ON ALL TRIALS
        % correlate predicted angle with true angle: not so great
        % combine the results to get a predicted angle
        % correlation between predicted angle and real angle
        [rho p] = circ_corrcc(repmat(angle_truth', [1 sz(angle_predicted, [2, 3])]),angle_predicted);
        rhos(s,:,:) = rho;
        
        %----- 3. RELIAGNMENT AND DISTRIBUTION OF ALL TRIALS
        % model trained and tested on same data
        [trial_prop predict] = recombine_SVR_prediction(results_x,results_y,angles,6);
        all_trialProp(:,:,:,s) = trial_prop;
        
        % generalize model trained on one type (eg target) to the other
        % type (eg probe)
        [trial_prop_gen predict_gen] = recombine_SVR_prediction(results_x,results_y,genAngles,6);
        all_trialProp_gen(:,:,:,s) = trial_prop_gen;
        
        %----- 4. CIRC ANALYSES (stats)
        % r test: uniformity of angle distribution (valid everywhere,eg generalization)
        [p z] = circ_rtest(2*predict-pi);
        all_p_r(:,:,s) = squeeze(-log10(p));
        all_z_r(:,:,s) = z;
        
        % v test: uniformity of angle distribution around a specified angle
        % (only valid on diagonal)
        [p z] = circ_vtest(2*predict-pi,0);
        all_p_v(:,:,s) = squeeze(-log10(p));
        all_z_v(:,:,s) = z;
        
        % on realigned other angle
        [p z] = circ_rtest(2*predict_gen - pi);
        all_p_r_gen(:,:,s) = squeeze(-log10(p));
        all_z_r_gen(:,:,s) = z;
        
        % on realigned other angle
        [p z] = circ_vtest(2*predict_gen - pi,0);
        all_p_v_gen(:,:,s) = squeeze(-log10(p));
        all_z_v_gen(:,:,s) = z;
%     end        
        %% EFFECT PER CATEGORY
        % define division of interest
        sel = [trials.present]==1; % select only present trials
        angles = [trials(sel).orientation];
        visibilities = [trials(sel).response_visibilityCode];
        
        % VISIBILITY
        for vis = 4:-1:1
            sel = visibilities==vis;
            try
                x.predict = results_x.predict(:,sel,:,:);
                y.predict = results_y.predict(:,sel,:,:);
                [trial_prop_vis predict_vis] = recombine_SVR_prediction(x,y,angles(sel),10);
                all_trial_prop_vis(s,:,:,:,vis) = trial_prop_vis;
                [p z] = circ_rtest(2*predict_vis-pi);
                all_p_r_vis(s,:,:,vis) = squeeze(z);
            catch
                disp(['problem with s' num2str(s) ' vis' num2str(vis)])
                all_trial_prop_vis(s,:,:,:,vis) = NaN;
                all_p_r_vis(s,:,:,vis) = NaN;
            end
        end
        
    end
    
    %% GAT plots
    all_tprop = squeeze(all_trialProp(round(end/2),:,:,:)); % take only peak of the tuning curve
    all_tprop_gen = squeeze(all_trialProp_gen(round(end/2),:,:,:));
    angle_error = shiftdim(angle_error,1);
    rhos = shiftdim(rhos,1); 
    
    plots = {{'angle_error'} ...
        {'rhos'} ...
        {'all_tprop' 'all_tprop_gen'} ...
        {'all_p_r' 'all_p_r_gen'} ...
        {'all_z_r' 'all_z_r_gen'} ...
        {'all_p_v' 'all_p_v_gen'} ...
        {'all_z_v' 'all_z_v_gen'} ...
        };
    titles={['angle error'] ...
        ['circular correlation'] ...
        ['proportion correct' ] ...
        ['Rayleigh test for non-uniformity (-log-p)'] ...
        ['Rayleigh test for non-uniformity (z)' ] ...
        ['V test for non-uniformity(-log-p)' ] ...
        ['V test for non-uniformity(v)'] ...
        };
    file_names = {'MAD' ...
        'circCorrelation' ...
        'trialsProps' ...
        'Rtest_pvalue' ...
        'Rtest_zvalue' ...
        'Vtest_pvalue' ...
        'Vtest_zvalue'};
    
    for pl = 1:length(plots) % plot
        figure
        set(gcf,'Position',get(0,'ScreenSize'))
        subplot(1,length(plots{pl}),1);
        eval(['imagesc(toi,toi,squeeze(mean(' plots{pl}{1} '(:,:,:),3)));']);
        colorbar;
        set(gca,'FontSize',24,'ydir', 'normal')
        title([titles{pl} ': ' models{mdl}]);axis image;
        ylabel('train time');xlabel('test time');
        
        if length(plots{pl})==2
            subplot(1,2,2);
            eval(['imagesc(toi,toi,squeeze(mean(' plots{pl}{2} '(:,:,:),3)));']);
            colorbar; set(gca,'FontSize',24,'ydir', 'normal')
            title([titles{pl} ': aligned to ' models{3-mdl}]);axis image;
            ylabel('train time');xlabel('test time');
        end
        plot2svg([im_path '/' models{mdl} '_tAll_' file_names{pl}]) 
    end
    
    %% GAT plots divided per visibility and contrast
    all_tprop_vis = squeeze(all_trial_prop_vis(:,round(end/2),:,:,:)); % take only peak of the tuning curve
    
    plots       = {'all_tprop_vis' ...
        'all_p_r_vis'};
    titles      = {'proportion correct' ...
        'Rayleigh test for non-uniformity (-log-p)'};
    file_names  = {'trialsProps_vis' ...
        'Rtest_pvalue_vis' };
    for pl = 1:length(plots)
        figure
        set(gcf,'Position',get(0,'ScreenSize'))
        for vis = 1:4
            subplot(2,2,vis)
            eval(['imagesc(toi,toi,squeeze(nanmean(' plots{pl} '(:,:,:,vis))));']);
            colorbar; set(gca,'FontSize',24,'ydir', 'normal')
            title(titles{pl});axis image;
            ylabel('train time');xlabel('test time');
        end
    end
    plot2svg([im_path '/' models{mdl} '_tAll_' file_names{pl}])
    
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%% TO BE CLEANED AND POLISHED %%%%%%%%%%%%%%%%%
% The following part is a draft for the part before that is a comprehensive
% loop across models [probe and target] and visualization plots [different 
% stats, correlations and prediction errors]

%% all subjects probe decoding
clear all_trialProp* all_p all_v_vis
for s = length(SubjectsList):-1:11 % the first 10 subjects need to be done but it's useless given that is can be inferred from tAll
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_x_results.mat'],'predict', 'y', 'dims');
    results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_y_results.mat'],'predict', 'y');
    toi = time(results_x.dims);
    
    %% get angle distance
    probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
    [trial_prop predict] = recombine_SVR_prediction(results_x,results_y,probeAngles,20);
    all_trialProp(:,:,s) = trial_prop;
    
    %% stats
    % test of non uniformity
    %[p z] = circ_rtest(2*predict-pi);
    % test of non uniformity and mean 0
    [p vstat] = circ_vtest(2*predict-pi,0);
    all_p(:,:,s) = p;
    
    %% sort by visibility
    visibility = [trials.response_visibilityCode]';
    visibility([trials.present]==0) = [];
    for vis = 4:-1:1
        sel = probeAngles;
        sel(visibility~=vis) = NaN;
        [trial_prop_vis predict_vis] = recombine_SVR_prediction(results_x,results_y,sel,6);
        all_trialProp_vis(:,:,vis,s) = trial_prop_vis;
        % stats
        [p vstat] = circ_vtest(2*predict(visibility==vis,:)-pi,0);
        all_v_vis(s,:,vis) = vstat;
        all_p_vis(s,:,vis) = p;
    end
end
imagesc(toi,[],mean(all_trialProp,3));
clf;hold on;
colors = colorGradient([1 0 0], [0 1 0],4);
for v = 1:4
    %subplot(4,1,v);
    %imagesc(toi,[],mean(all_trialProp_vis(:,:,v,:),4));
    [hl(v)]=plot_eb(toi,all_v_vis(:,:,v),colors(v,:));
end
set(gca,'FontSize',24)
title('test of non-uniformity with specified direction(v stat)');
legend(cell2mat(hl),'vis0','vis1','vis2','vis3')
set(gcf,'Position',get(0,'ScreenSize'))
ylabel('v-stat ');xlabel('time');
% saveas(gcf,[path_images '/SVR_pAngle_Vtest_vis.jpeg']);

% visualize p values
figure,
temp= -log10(all_p_vis);
imagesc(toi,1:4,squeeze(median(temp))'),colorbar
set(gca,'FontSize',24,'YTick',[1:4],'YTickLabel',{'vis0' 'vis1' 'vis2' 'vis3'})
set(gcf,'Position',get(0,'ScreenSize'))
xlabel('time');title('median -log10 (p-val)')
% saveas(gcf,[path_images '/SVR_pAngle_pval_vis.jpeg']);

%% all subjects probe decoding generalization across time
clear all_trialProp* all_p
for s = length(SubjectsList):-1:1
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_tAll_x_results.mat'],'predict', 'y');
    results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_tAll_y_results.mat'],'predict', 'y');
    
    %% get angle distance
    probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
    probeAngles([trials.present]==0) = [];
    targetAngles = [trials.orientation]';
    targetAngles([trials.present]==0) = [];
    [trial_prop predict] = recombine_SVR_prediction(results_x,results_y,probeAngles,6);
    % after Probe to Target realignment to target: HERE add: store values, plot etc
    [trial_prop_P2T predict_P2T] = recombine_SVR_prediction(results_x,results_y,targetAngles,6);
    
    all_trialProp(:,:,:,s) = trial_prop;
    
    %% stats
    % test of non uniformity
    [p z] = circ_rtest(2*predict-pi);
    % test of non uniformity and mean 0
    %[p z] = circ_vtest(2*predict-pi,0);
    all_p(:,:,s) = p;
    all_z(:,:,s) = z;
end
figure
imagesc(toi,toi,squeeze(mean(all_trialProp(round(end/2),:,:,:),4)));colorbar;
set(gca,'FontSize',24,'ydir', 'normal')
title('proportion correct');
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_pAngle_propCorrect_tAll.jpeg']);

figure
imagesc(toi,toi,mean(all_z,3));colorbar;
set(gca,'FontSize',24,'ydir', 'normal')
title('Rayleigh''s test of non-uniformity (z-stat)');
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_pAngle_Rayleigh_tAll.jpeg']);

figure
imagesc(toi,toi,mean(-log10(all_p),3),[0 4]);colorbar();
set(gca,'FontSize',24,'ydir', 'normal')
title('test of non-uniformity (-log10(p) value)');
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_pAngle_pval_tAll.jpeg']);