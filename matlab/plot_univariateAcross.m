%% TO BE COMPLETED
%-- see fieldtrip tutorial to highlight significant channels (after having
%-- reloaded the respective stats)
%-- also plot the imagesc_meg(x,time,pos) 

%% Topography Angles
%--- Plot specific topography of each angle (across subjects)

% load data from each subject
var_name = 'angles';
clear all_angles
for s = length(SubjectsList):-1:1
    clear angles
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    all_angles(s,:,:,:) = angles(1:306,:,:); % subjects x channels x time x angle_category
end

% plot topos
figure(1);clf;set(gcf,'color','w','name','mag');hold on 
figure(2);clf;set(gcf,'color','w','name','grad');hold on
figure(3);%would be used for the across subjects stats (which we haven't determined yet)
for a = 1:6 % for each angle
    clear ERF
    for s = 20:-1:1
        ERF(:,:,s) = ...
            all_angles(s,1:306,:,a)-...
            mean(all_angles(s,1:306,:,:,:),4); % we remove the mean present_erf from each orientation to make it specific
    end
    % get planar combination for each subject separately
    [ft_data cfg] = plot_topo_meg(ERF,time);
    
    clf;set(gcf,'color','w');
    toi = -.200:.050:.800;
    for t = 1:length(toi)
        cfg.xlim    = toi(t)+[-1 1]*.025;
        %% magnetometers
        figure(1);hold on
        cfg.layout  = 'neuromag306mag.lay';
        cfg.zlim   = [-1 1]*5e-14;
        subaxis(6,length(toi),length(toi)*(a-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if a==1,title([num2str(1000*toi(t)), 'ms']);end
        
        %% combined planars
        figure(2);hold on
        cfg.layout  = 'neuromag306cmb.lay';
        cfg.zlim   = [0 1.5]*1e-12;
        subaxis(6,length(toi),length(toi)*(a-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if a==1,title([num2str(1000*toi(t)), 'ms']);end
    end
end

%% Topography Present versus Absent
% load data from each subject
var_name = 'presentAbsent';
clear presentAbsent
for s = length(SubjectsList):-1:1
    clear presentAbsent
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    all_presentAbsent(s,:,:,:) = presentAbsent(1:306,:,:); % subjects x channels x time x presentAbsent
end

% plot topos
figure(4);clf;set(gcf,'color','w','name', 'mag');
figure(5);clf;set(gcf,'color','w','name', 'grad');
ERF_types = {'present', 'absent', 'present-absent'};
for type = 1:length(ERF_types);
    clear ERF
    for s = 20:-1:1
        switch ERF_types{type}
            case 'present', ERF(:,:,s) = all_presentAbsent(s,1:306,:,1);
            case 'absent', ERF(:,:,s) = all_presentAbsent(s,1:306,:,2);
            case 'present-absent', ERF(:,:,s) = all_presentAbsent(s,1:306,:,1)-all_presentAbsent(s,1:306,:,2);
        end
    end
    % get planar combination for each subject separately
    [ft_data cfg] = plot_topo_meg(ERF,time);
    
    toi = 0:.100:.800;
    for t = 1:length(toi)
        cfg.xlim    = toi(t)+[-1 1]*.025;
        %% magnetometers
        figure(4);
        cfg.layout  = 'neuromag306mag.lay';
        cfg.zlim   = [-1 1]*1e-13;
        subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if type==1,title([num2str(1000*toi(t)), 'ms']);end
        
        %% combined planars
        figure(5);
        subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        cfg.zlim   = [0 5]*1e-12;
        cfg.layout  = 'neuromag306cmb.lay';
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if type==1,title([num2str(1000*toi(t)), 'ms']);end
    end
end


%% Topography Seen versus Unseen
% load data from each subject
var_name = 'seenUnseen';
clear seenUnseen
for s = length(SubjectsList):-1:1
    clear seenUnseen
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    all_seenUnseen(s,:,:,:) = seenUnseen(1:306,:,:); % subjects x channels x time x seenUnseen
end

% plot topos
figure(6);clf;set(gcf,'color','w','name', 'mag');
figure(7);clf;set(gcf,'color','w','name', 'grad');
ERF_types = {'present', 'absent', 'present-absent'};
for type = 1:length(ERF_types);
    clear ERF
    for s = 20:-1:1
        switch ERF_types{type}
            case 'present', ERF(:,:,s) = all_seenUnseen(s,1:306,:,1);
            case 'absent', ERF(:,:,s) = all_seenUnseen(s,1:306,:,2);
            case 'present-absent', ERF(:,:,s) = all_seenUnseen(s,1:306,:,1)-all_seenUnseen(s,1:306,:,2);
        end
    end
    % get planar combination for each subject separately
    [ft_data cfg] = plot_topo_meg(ERF,time);
    
    toi = 0:.100:.800;
    for t = 1:length(toi)
        cfg.xlim    = toi(t)+[-1 1]*.025;
        %% magnetometers
        figure(6);
        cfg.layout  = 'neuromag306mag.lay';
        cfg.zlim   = [-1 1]*1e-13;
        subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if type==1,title([num2str(1000*toi(t)), 'ms']);end
        
        %% combined planars
        figure(7);
        subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
        cfg.zlim   = [0 5]*1e-12;
        cfg.layout  = 'neuromag306cmb.lay';
        ft_topoplotER(cfg,ft_data);
        axis tight;
        if type==1,title([num2str(1000*toi(t)), 'ms']);end
    end
end