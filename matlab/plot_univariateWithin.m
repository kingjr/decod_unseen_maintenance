%% Topography Angles
close all
var_name = 'angles';
clear all_angles
for s = length(SubjectsList):-1:1
    % create folder destination for images
    if ~exist([path 'data/' SubjectsList{s} '/images/univariate'],'dir'),
        mkdir([path 'data/' SubjectsList{s} '/images/univariate'])
    end
    path_images = [path 'data/' SubjectsList{s} '/images/univariate'];
    
    % load data from each subject
    clear angles
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    
    toi = -.200:.050:.800;
    for a = 1:6 % for each angle
        clear ERF
        ERF(:,:) = ...
            angles(1:306,:,a)-...
            mean(angles(1:306,:,:),3); % we remove the mean present_erf from each orientation to make it specific
        
        
        % get planar combination for each subject separately
        [ft_data cfg] = plot_topo_meg(ERF,time);
        
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
    figure(1)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': magnetometers']);
    saveas(gcf,[path_images '/angles_magnetometers.jpeg']);
    figure(2)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': combined']);
    saveas(gcf,[path_images '/angles_combined.jpeg']);
    close all
end

%% Topography Present versus Absent
% load data from each subject
var_name = 'presentAbsent';
clear presentAbsent

ERF_types = {'present', 'absent', 'present-absent'};

for s = length(SubjectsList):-1:1
    path_images = [path 'data/' SubjectsList{s} '/images/univariate'];
    
    % load data from each subject
    clear presentAbsent
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    for type = 1:length(ERF_types);
        clear ERF
        switch ERF_types{type}
            case 'present', ERF(:,:) = presentAbsent(1:306,:,1);
            case 'absent', ERF(:,:) = presentAbsent(1:306,:,2);
            case 'present-absent', ERF(:,:,s) = presentAbsent(1:306,:,1)-presentAbsent(1:306,:,2);
        end
        % get planar combination for each subject separately
        [ft_data cfg] = plot_topo_meg(ERF,time);
        
        toi = 0:.100:.800;
        for t = 1:length(toi)
            cfg.xlim    = toi(t)+[-1 1]*.025;
            %% magnetometers
            figure(1);
            cfg.layout  = 'neuromag306mag.lay';
            cfg.zlim   = [-1 1]*1e-13;
            subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
            ft_topoplotER(cfg,ft_data);
            axis tight;
            if type==1,title([num2str(1000*toi(t)), 'ms']);end
            
            %% combined planars
            figure(2);
            subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
            cfg.zlim   = [0 5]*1e-12;
            cfg.layout  = 'neuromag306cmb.lay';
            ft_topoplotER(cfg,ft_data);
            axis tight;
            if type==1,title([num2str(1000*toi(t)), 'ms']);end
        end
    end
    
    figure(1)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': magnetometers']);
    saveas(gcf,[path_images '/presentAbsent_magnetometers.jpeg']);
    figure(2)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': combined']);
    saveas(gcf,[path_images '/presentAbsent_combined.jpeg']);
    close all
end


%% Topography Seen versus Unseen

var_name = 'seenUnseen';
clear seenUnseen

ERF_types = {'present', 'absent', 'present-absent'};
for s = length(SubjectsList):-1:1
    path_images = [path 'data/' SubjectsList{s} '/images/univariate'];
    
    % load data from each subject
    clear seenUnseen
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name, 'time', 'labels');
    
    for type = 1:length(ERF_types);
        clear ERF
        for s = 20:-1:1
            switch ERF_types{type}
                case 'present', ERF(:,:) = seenUnseen(1:306,:,1);
                case 'absent', ERF(:,:) = seenUnseen(1:306,:,2);
                case 'present-absent', ERF(:,:) = seenUnseen(1:306,:,1)-seenUnseen(1:306,:,2);
            end
        end
        % get planar combination for each subject separately
        [ft_data cfg] = plot_topo_meg(ERF,time);
        
        toi = 0:.100:.800;
        for t = 1:length(toi)
            cfg.xlim    = toi(t)+[-1 1]*.025;
            %% magnetometers
            figure(1);
            cfg.layout  = 'neuromag306mag.lay';
            cfg.zlim   = [-1 1]*1e-13;
            subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
            ft_topoplotER(cfg,ft_data);
            axis tight;
            if type==1,title([num2str(1000*toi(t)), 'ms']);end
            
            %% combined planars
            figure(2);
            subaxis(3,length(toi),length(toi)*(type-1)+t,'SpacingHoriz',0,'SpacingVert',0);
            cfg.zlim   = [0 5]*1e-12;
            cfg.layout  = 'neuromag306cmb.lay';
            ft_topoplotER(cfg,ft_data);
            axis tight;
            if type==1,title([num2str(1000*toi(t)), 'ms']);end
        end
    end
    
    figure(1)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': magnetometers']);
    saveas(gcf,[path_images '/seenUnseen_magnetometers.jpeg']);
    figure(2)
    set(gca,'FontSize',24);
    set(gcf,'Position',get(0,'ScreenSize'))
    title([SubjectsList{s} ': combined']);
    saveas(gcf,[path_images '/seenUnseen_combined.jpeg']);
    close all
end