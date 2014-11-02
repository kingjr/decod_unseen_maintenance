

%% FIRST DRAFT: Plotting: example to plot topographies for magnetometers only
data.trial{1} = -log10(p);
cfg         = [];
cfg.colorbar= 'no';
cfg.comment = 'no';
cfg.marker  = 'off';
cfg.style   = 'straight';
cfg.layout = 'neuromag306mag.lay';
cfg.clim    = [0 10];
toi = 0:.050:1.4;
for t = 1:length(toi)
    subaxis(5,6,t,'SpacingHoriz',0,'SpacingVert',0);
    cfg.xlim = toi(t)+[-.010 .010];
    ft_topoplotER(cfg,data);
    axis tight;axis([xlim ylim zlim 0 .3]);
    title([num2str(1000*toi(t)), 'ms']);
end

%% FIRST DRAFT: Plotting: This part of a function is used to re-align the angle prediction from a 
% SVR decoder trained on the probe's orientation to test its ability to
% generalize to the target

% trained on probe angle => generalize to re-aligned target angle?
% relign according to the probe true angle
targetAngles = [trials.orientation];
targetAngles([trials.present]==0) = [];

% align
align = 'target';
switch align
    case 'target'
        % predictions aligned on target's angle
        [trial_prop predict] = decode_reg2angle(...
            results_x,....
            results_y,...
            targetAngles,7);
    case 'probe'
        % predictions aligned on probe's angle
        [trial_prop predict] = decode_reg2angle(...
            results_x,....
            results_y,...
            mod([trials.orientation]+[trials.tilt]-1,6)+1);
end
% plot
switch cfg.gentime
    case '_tAll'
        % if not gen time
        imagesc(time(cfg.dims),[],trial_prop,[0 .5]);
    otherwise
        % if gen time
        %--- plot gentime on closest angle distance
        imagesc(squeeze(trial_prop(round(end/2),:,:)));
        set(gca,'ydir', 'normal');
        
        %---- plot p value
        predict(isnan(predict(:,1,1)),:,:) = [];
        [p z] = circ_rtest(2*predict-pi);
        imagesc(time(toi),time(toi),squeeze(-log10(p)));
        % imagesc(time(cfg.dims),time(cfg.dims),squeeze(p));
        set(gca,'ydir', 'normal');
end
















%% FIST DRAFT: single subject plots and topo

sbj_initials = 'ma130185';
data_path = [path 'data/' sbj_initials '/'] ;

%---- load details
file_behavior   = [data_path 'behavior/' sbj_initials '_fixed.mat'];
file_header     = [data_path 'preprocessed/' sbj_initials '_preprocessed.mat'];
file_binary     = [data_path 'preprocessed/' sbj_initials '_preprocessed.dat'];
load(file_behavior, 'trials');
load(file_header, 'data');
time    = data.time{1}; % in s

%---- load all MEG data
MEEG = binload(file_binary,data.Xdim);


%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);

figure(1);%clf;set(gcf,'color','w');
for a = 1:6
    % define Event Related Field:
    ERF_All = squeeze(trimmean(MEEG([trials.present]==1,:,:),90));
    ERF_Angle = squeeze(trimmean(MEEG([trials.orientation]==a & [trials.present]==1,:,:),90));
    
    ERF = ERF_Angle-ERF_All;
    
    switch 'topo'
        case 'quick_look'
            subplot(6,1,a);
            imagesc_meg(time,ERF,layout.pos,[-1 1]*2e-12);
        case 'topo'
            ts = 0:.050:.500; % plotting time
            
            % magnetometers
            figure(1);
            magnetometers = data;
            magnetometers.trial{1} = ERF;
            cfg         = [];
            cfg.colorbar= 'no';
            cfg.layout  = 'neuromag306cmb.lay';
            cfg.comment = 'no';
            cfg.marker  = 'off';
            cfg.style   = 'straight';
            cfg.layout = 'neuromag306mag.lay';
            cfg.clim    = [-5 5]*1e-14;
            for t = 1:length(ts)
                subaxis(6,length(ts),(a-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.050 .050];
                ft_topoplotER(cfg,magnetometers);
                axis tight;
                if a == 1
                   title([num2str(1000*ts(t)), 'ms']);
                end
            end
            % combined planars
            figure(2);
            cfg = [];
            cfg.zlim = [0 3*10^-12];
            [combined_planar cfg] = plot_topo_meg(ERF,time,cfg);
            for t = 1:length(ts)
                subaxis(6,length(ts),(a-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.050 .050];
                ft_topoplotER(cfg,combined_planar);
                axis tight;title([num2str(1000*ts(t)), 'ms']);
            end
    end
end

%---- statistics
X = MEEG([trials.present]==1,:,:);
y = [trials([trials.present]==1).orientation]';

chans = selchan(data.label,'MEG');
toi = 1:length(time);

clear p t stats;
stats_type = 'anova';
stats_type = 'circ_corrcl';
switch stats_type
    case 'anova'
for chan = length(chans):-1:1
    fprintf('\n');
    for t = length(toi):-1:1
        fprintf('*');
        [p(chan,t),~,stats(chan,t)] = anovan(X(:,chan,t),{y},'display','off');
    end
end
    case 'circ_corrcl'
        [rho p] = circ_corrcl(...
            repmat(2*deg2rad(y*30-15),[1,size(X,2),size(X,3)]),...
            X(:,chans,:));
end

%---- quick plot of statistics: using order channels from back to front
[~,order] = sort(layout.pos(chans,2), 'descend');
imagesc(time,[],-log10(p(order,:)),[0 10]);

%---- plot topo of p values
figure(3);clf;set(gcf,'color','w');
magnetometers = data;
magnetometers.trial{1}(chans,:) = -log10(p);
cfg         = [];
cfg.colorbar= 'no';
cfg.layout  = 'neuromag306cmb.lay';
cfg.comment = 'no';
cfg.marker  = 'off';
cfg.style   = 'straight';
cfg.layout = 'neuromag306mag.lay';
cfg.clim    = [0 10];
toi = 0:.050:1.4;
for t = 1:length(toi)
    subaxis(5,6,t,'SpacingHoriz',0,'SpacingVert',0);
    cfg.xlim = toi(t)+[-.010 .010];
    ft_topoplotER(cfg,magnetometers);
    axis tight;axis([xlim ylim zlim 0 10]);
    title([num2str(1000*toi(t)), 'ms']);
end
% movie;
toi = 0:.010:1.4;
for t = 1:length(toi)
    cfg.xlim = toi(t)+[-.010 .010];
    ft_topoplotER(cfg,magnetometers);
    axis tight;axis([xlim ylim zlim 0 10]);
    title([num2str(1000*toi(t)), 'ms']);
    pause;
end

%% PLOT TOPO UNIVARIATE ANGLE EFFECT (mag only)
rho = [];
for s = length(SubjectsList):-1:1
    subject = SubjectsList{s};
    load([path 'data/' subject '/univariate/' subject '_univariate_angleCirc.mat'],'angleCirc', 'time');
    rho(:,:,s) = angleCirc.rho;
end
topo = mean(rho,3);

%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);
figure(5);set(gcf,'color','w');
imagesc_meg(time,topo,layout.pos,[.05 .14],1);
axis([-.200 1.800 ylim]);colorbar;

ts = -.200:.050:1.500; % plotting time
% magnetometers
figure(4);clf;set(gcf,'color','w');
cfg         = [];
cfg.colorbar= 'no';
cfg.comment = 'no';
cfg.marker  = 'off';
cfg.style   = 'straight';
cfg.layout = 'neuromag306mag.lay';
cfg.zlim    = [.05 .10];
for t = 1:length(ts)
    t
    subaxis(5,length(ts)/5,t,'SpacingHoriz',0);
    cfg.xlim = ts(t)+[-.012 .012];
    plot_topo_meg(topo,time,cfg);
    axis tight;
    title([num2str(1000*ts(t)), 'ms']);
end



%% retrieve classifier features: NOT READY YET
load([path 'data/ak130184/mvpas/ak130184_preprocessed_presentAbsent_SVC_results.mat'], 'coef', 'wsize', 'dims')
load([path 'data/ma130185/mvpas/ma130185_preprocessed_responseButton_SVC_results.mat'],'coef','wsize','dims')
size(coef) ;% split x  fold x time train x features (chans x time)

% average classifiers across folds
coef = squeeze(mean(coef,2));
% average classifiers across time if multiple time poitns were used by the
% decoder
coef = reshape(coef,[length(dims) 306 double(wsize) ]);
%coef = squeeze(mean(coef,3));
coef = squeeze(coef(:,:,1));


% plot
imagesc(time,[],coef',[-1 1]*1e-12);



