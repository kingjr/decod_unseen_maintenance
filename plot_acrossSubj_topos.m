%% load concatenated mean erf data
load([path 'data/across_subjects/withinSubject_erf.mat'], 'erf');

%% load time 
if ~exist('time','var'),
    % load data from first subject to get time labels
    subject = SubjectsList{1};

    clear data MEEG trials
    load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'], 'trials');
    MEEG = binload([path 'data/' subject '/preprocessed/' subject '_preprocessed.dat'], data.Xdim);
    time = data.time{1};
end


%% compute average across subjects
clear topos
for s = length(SubjectsList):-1:1
    %% target orientations across time
    topos_angles(s,:,:,:) = erf(s).angles(1:306,:,:); %retrieve sbj information in one matrix   
    %% each visibility across time
    topos_visibility(s,:,:,:) = erf(s).visibility(1:306,:,:); %retrieve sbj information in one matrix 
    %% seen vs unseen across time
    topos_seenUnseen(s,:,:,:) = erf(s).seenUnseen(1:306,:,:); %retrieve sbj information in one matrix
    
    %% seenXpresent (seenUnseen(1 seen 2 unseen) x presence (1 present 2 absent))
    topos_seenXpresent(s,:,:,:,:) = erf(s).seenXpresent(1:306,:,:,:); %retrieve sbj information in one matrix
end
%% average across subjects
meantopos_angles = squeeze(nanmean(...
    topos_angles-repmat(mean(topos_angles,4),[1 1 1 6])...
,1));

meantopos_visibility = squeeze(nanmean(topos_visibility(:,:,:,:),1));
meantopos_seenUnseen = squeeze(nanmean(topos_seenUnseen(:,:,:,:),1));
meantopos_seenXpresent = squeeze(nanmean(topos_seenXpresent(:,:,:,:,:),1));
% compute difference
meantopos_seenXpresent(:,:,3,:) = squeeze(nanmean(...
    topos_seenXpresent(:,:,:,1,:)-topos_seenXpresent(:,:,:,2,:)...
    ,1));

%% ---------------PLOT AVERAGE ACROSS SUBJECTS-----------------------------
%% target orientation
%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);
for a =6:-1:1
    switch 'topo'
        case 'quick_look'
            subplot(6,1,a);
            imagesc_meg(time,meantopos_angles(:,:,a),layout.pos,[-1 1]*2e-12);
        case 'topo'
            ts = 0:.050:.500; % plotting time
            
            % magnetometers
            figure(1);
            cfg         = [];
            cfg.colorbar= 'no';
            cfg.layout  = 'neuromag306cmb.lay';
            cfg.comment = 'no';
            cfg.marker  = 'off';
            cfg.style   = 'straight';
            cfg.layout = 'neuromag306mag.lay';
            cfg.zlim = [-2 2]*1e-14;
            for t = 1:length(ts)
                subaxis(6,length(ts),(a-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                plot_topo_meg(meantopos_angles(:,:,a),time,cfg);
                axis tight;
                if a == 1
                    title([num2str(1000*ts(t)), 'ms']);
                end
            end
            % combined planars
            figure(2);
            cfg = [];
            cfg.zlim = [0 1*10^-12];
            [combined_planar cfg] = plot_topo_meg(meantopos_angles(:,:,a),time,cfg);
            for t = 1:length(ts)
                subaxis(6,length(ts),(a-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                ft_topoplotER(cfg,combined_planar);
                axis tight;title([num2str(1000*ts(t)), 'ms']);
            end
    end
end

%% visibility rating
%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);
for v =4:-1:1
    switch 'topo'
        case 'quick_look'
            subplot(4,1,v);
            imagesc_meg(time,meantopos_visibility(:,:,v),layout.pos,[-1 1]*2e-12);
        case 'topo'
            ts = 0:.050:.800; % plotting time
            
            % magnetometers
            figure(3);
            cfg         = [];
            cfg.colorbar= 'no';
            cfg.layout  = 'neuromag306cmb.lay';
            cfg.comment = 'no';
            cfg.marker  = 'off';
            cfg.style   = 'straight';
            cfg.layout = 'neuromag306mag.lay';
            cfg.zlim    = [-1 1]*1e-13;
            for t = 1:length(ts)
                subaxis(4,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                plot_topo_meg(meantopos_visibility(:,:,v),time,cfg);
                axis tight;
                if v == 1
                    title([num2str(1000*ts(t)), 'ms']);
                end
            end
            % combined planars
            figure(4);
            cfg = [];
            cfg.zlim = [0 3*10^-12];
            [combined_planar cfg] = plot_topo_meg(meantopos_visibility(:,:,v),time,cfg);
            for t = 1:length(ts)
                subaxis(4,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                ft_topoplotER(cfg,combined_planar);
                axis tight;title([num2str(1000*ts(t)), 'ms']);
            end
    end
end


%% Seen versus Snseen
%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);
for v =2:-1:1
    switch 'topo'
        case 'quick_look'
            subplot(2,1,v);
            imagesc_meg(time,meantopos_visibility(:,:,v),layout.pos,[-1 1]*2e-12);
        case 'topo'
            ts = 0:.050:.500; % plotting time
            
            % magnetometers
            figure(5);
            cfg         = [];
            cfg.colorbar= 'no';
            cfg.layout  = 'neuromag306cmb.lay';
            cfg.comment = 'no';
            cfg.marker  = 'off';
            cfg.style   = 'straight';
            cfg.layout = 'neuromag306mag.lay';
            cfg.zlim    = [-5 5]*1e-14;
            for t = 1:length(ts)
                subaxis(2,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                plot_topo_meg(meantopos_visibility(:,:,v),time,cfg);
                axis tight;
                if v == 1
                    title([num2str(1000*ts(t)), 'ms']);
                end
            end
            % combined planars
            figure(6);
            cfg = [];
            cfg.zlim = [0 3*10^-12];
            [combined_planar cfg] = plot_topo_meg(meantopos_visibility(:,:,v),time,cfg);
            for t = 1:length(ts)
                subaxis(2,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                cfg.xlim = ts(t)+[-.012 .012];
                ft_topoplotER(cfg,combined_planar);
                axis tight;title([num2str(1000*ts(t)), 'ms']);
            end
    end
end

%% Seen X Present
%---- prepare layout
cfg = [];
cfg.layout = 'neuromag306all.lay';
layout = ft_prepare_layout(cfg);
% plot only seen present versus unseen present
for v =3:-1:1       %(1 seen 2 unseen, 3 difference) 
    for p = 1  %presence (1 present 2 absent)
        switch 'topo'
            case 'quick_look'
                subplot(2,2,v);
                imagesc_meg(time,meantopos_seenXpresent(:,:,v,p),layout.pos,[-1 1]*2e-12);
            case 'topo'
                ts = -.050:.050:.700; % plotting time
                
                % magnetometers
                figure(7);
                cfg         = [];
                cfg.colorbar= 'no';
                cfg.layout  = 'neuromag306cmb.lay';
                cfg.comment = 'no';
                cfg.marker  = 'off';
                cfg.style   = 'straight';
                cfg.layout = 'neuromag306mag.lay';
                cfg.zlim    = [-1 1]*1e-13;
                for t = 1:length(ts)
                    subaxis(3,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                    cfg.xlim = ts(t)+[-.012 .012];
                    evalc('plot_topo_meg(meantopos_seenXpresent(:,:,v,p),time,cfg);')
                    axis tight;
                    if v == 1
                        title([num2str(1000*ts(t)), 'ms']);
                    end
                end
                % combined planars
                figure(8);
                cfg = [];
                cfg.zlim = [0 3*10^-12];
                [combined_planar cfg] = plot_topo_meg(meantopos_seenXpresent(:,:,v,p),time,cfg);
                for t = 1:length(ts)
                    subaxis(3,length(ts),(v-1)*length(ts)+t,'SpacingHoriz',0,'SpacingVert',0);
                    cfg.xlim = ts(t)+[-.012 .012];
                    evalc('ft_topoplotER(cfg,combined_planar);')
                    axis tight;title([num2str(1000*ts(t)), 'ms']);
                end
        end
    end
end