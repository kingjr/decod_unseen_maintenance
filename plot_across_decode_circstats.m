%% load time and label data from a standard subject
subject = SubjectsList{1};
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
labels = data.label;
time = data.time{1};

%% define images path 
path_images = [path 'images_acrossSubjs'];

%% all subjects target angle decoding
clear all_trialProp* all_p all_v_vis
for s = length(SubjectsList):-1:1
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_x_results.mat'],'predict', 'y', 'dims');
    results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_y_results.mat'],'predict', 'y');
    toi = time(results_x.dims);
    
    %% get angle distance
    targetAngles = [trials.orientation]';
    targetAngles([trials.present]==0) = [];
    [trial_prop predict] = decode_reg2angle(results_x,results_y,targetAngles,20);
    all_trialProp(:,:,s) = trial_prop;
    
    %% stats
    % test of non uniformity
    %[p z] = circ_rtest(2*predict-pi);
    % test of non uniformity and mean 0
    %XXXXX circ_corrcc(deg2rad
    
    [p vstat] = circ_vtest(2*predict-pi,0);
    all_p(:,:,s) = p;
    
    %% sort by visibility
    visibility = [trials.response_visibilityCode]';
    visibility([trials.present]==0) = [];
 probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
   probeAngles([trials.present]==0) = [];
 
    for vis = 4:-1:1
        sel = targetAngles;
        sel = probeAngles;
        sel(visibility~=vis) = NaN;
        [trial_prop_vis predict_vis] = decode_reg2angle(results_x,results_y,sel,6);
        all_trialProp_vis(:,:,vis,s) = trial_prop_vis;
        % stats
        [p vstat] = circ_vtest(2*predict(visibility==vis,:)-pi,0);
        n = sum(visibility==vis);
        all_v_vis(s,:,vis) = vstat/n;
        n_vis(s,vis) = n;
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
% saveas(gcf,[path_images '/SVR_TAngle_Vtest_vis.jpeg']);

% visualize p values
figure,
temp= -log10(all_p_vis);
imagesc(toi,1:4,squeeze(median(temp))'),colorbar
set(gca,'FontSize',24,'YTick',[1:4],'YTickLabel',{'vis0' 'vis1' 'vis2' 'vis3'})
set(gcf,'Position',get(0,'ScreenSize'))
xlabel('time');title('median -log10 (p-val)')
% saveas(gcf,[path_images '/SVR_TAngle_pval_vis.jpeg']);

%% all subjects target angle decoding generalization across time
% XXX Externalize function so that it become a generic prediction,
% independently of whether models are fitted on probe or on target.
clear all_trialProp* all_p
for s = length(SubjectsList):-1:1
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_tAll_x_results.mat'],'predict', 'y');
    results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_targetAngle_SVR_tAll_y_results.mat'],'predict', 'y');
    
    %% get angle distance
    targetAngles = [trials.orientation]';
    targetAngles([trials.present]==0) = []; % XXX Mmmh this is weird, probably the Target angle contrast is defined without present trials, whereas the Probe angle contrast is defined with present trials?
    % decode_reg2angle(cos prediction of angle, sin prediction of angle, target angle in index [1, 2, 3...6], number of bin for distribution)
    [trial_prop predict] = decode_reg2angle(results_x,results_y,targetAngles,6);
    all_trialProp(:,:,:,s) = trial_prop;
    
    % do the same for prediction realigned to the probe:
    probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
    probeAngles([trials.present]==0) = []; % XXX Mmmh this is weird, probably the Target angle contrast is defined without present trials, whereas the Probe angle contrast is defined with present trials?
    [trial_prop_T2P predict_T2P] = decode_reg2angle(results_x,results_y,probeAngles,6);
    all_trialProp_T2P(:,:,:,s) = trial_prop_T2P;
    
    
    %% stats
    % test of non uniformity
    [p z] = circ_rtest(2*predict-pi);
    % test of non uniformity and mean 0
    %[p z] = circ_vtest(2*predict-pi,0);
    all_p(:,:,s) = p;
    all_z(:,:,s) = z;
    
    % Example of stat for target to probe realignement
    [all_p_T2P(:,:,s) z] = circ_rtest(2*predict-pi);    
end
figure
subplot(1,2,1);
imagesc(toi,toi,squeeze(mean(all_trialProp(round(end/2),:,:,:),4)));colorbar;
set(gca,'FontSize',24,'ydir', 'normal')
title('proportion correct');axis image;
ylabel('train time');xlabel('test time');

subplot(1,2,2);
imagesc(toi,toi,squeeze(mean(all_trialProp_T2P(round(end/2),:,:,:),4)));colorbar;
set(gca,'FontSize',24,'ydir', 'normal')
title('proportion correct');axis image;
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_TAngle_propCorrect_tAll.jpeg']);

figure
imagesc(toi,toi,mean(all_z,3));colorbar;
set(gca,'FontSize',24,'ydir', 'normal')
title('Rayleigh''s test of non-uniformity (z-stat)');
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_TAngle_Rayleigh_tAll.jpeg']);

figure
imagesc(toi,toi,mean(-log10(all_p),3),[0 4]);colorbar();
set(gca,'FontSize',24,'ydir', 'normal')
title('test of non-uniformity (-log10(p) value)');
ylabel('train time');xlabel('test time');
set(gcf,'Position',get(0,'ScreenSize'))
% saveas(gcf,[path_images '/SVR_TAngle_pval_tAll.jpeg']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% all subjects probe decoding
clear all_trialProp* all_p all_v_vis
for s = length(SubjectsList):-1:1
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results_x = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_x_results.mat'],'predict', 'y', 'dims');
    results_y = load([path 'data/' subject '/mvpas/' subject '_preprocessed_probeAngle_SVR_y_results.mat'],'predict', 'y');
    toi = time(results_x.dims);
    
    %% get angle distance
    probeAngles = mod([trials.orientation]'+[trials.tilt]'-1,6)+1;
    [trial_prop predict] = decode_reg2angle(results_x,results_y,probeAngles,20);
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
        [trial_prop_vis predict_vis] = decode_reg2angle(results_x,results_y,sel,6);
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
    targetAngles = [trials.orientation]';
    [trial_prop predict] = decode_reg2angle(results_x,results_y,probeAngles,6);
    % after Probe to Target realignment to target: HERE add: store values, plot etc
    [trial_prop_P2T predict_P2T] = decode_reg2angle(results_x,results_y,targetAngles,6);
    
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