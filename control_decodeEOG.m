%% TEST FOR EOG ARTEFACTS: FIRST DRAFT
for s = 1:length(SubjectsList)
    sbj_initials = SubjectsList{s};
    data_path = [path 'data/' sbj_initials '/'] ;
    
    %% load details
    file_behavior   = [data_path 'behavior/' sbj_initials '_fixed.mat'];
    file_header     = [data_path 'preprocessed/' sbj_initials '_preprocessed.mat'];
    file_binary     = [data_path 'preprocessed/' sbj_initials '_preprocessed.dat'];
    load(file_behavior, 'trials');
    load(file_header, 'data');
    time    = data.time{1}; % in s
    
    % load all MEG data
    MEEG = binload(file_binary,data.Xdim);
    
    % only keep EOG
    selchan = @(c,s) find(cell2mat(cellfun(@(x) ~isempty(strfind(x,s)),c,'uniformoutput', false))==1);
    MEEG = MEEG(:,selchan(data.label,'EOG'),:);
    % remove average activity across orientations
    MEEG = MEEG - repmat(median(MEEG),[size(MEEG,1), 1, 1]);
    
    % define time points of interest
    toi = find(time>-.200,1):5:find(time>1.500,1);
    
    
    %% plot univariate
    figure(5);clf;set(gcf,'color','w');
    clear x y;
    for a = 6:-1:1
        subplot(6,1,a);hold on;
        sel = [trials.orientation]==a;
        plot_eb(time,squeeze(MEEG(sel,1,:)),[0 0 1]);
        plot_eb(time,squeeze(MEEG(sel,2,:)),[1 0 0]);
        x(a,:) = squeeze(median(squeeze(MEEG(sel,1,:))));
        y(a,:) = squeeze(median(squeeze(MEEG(sel,2,:))));
    end
    
%     clf;
%     subplot(1,2,1);imagesc(time,[],x);
%     subplot(1,2,2);imagesc(time,[],y);
    
    %% decode from EOG
    % define classifier
    cfg         = [];
    cfg.clf_type= 'SVR';
    cfg.n_folds = 8;
    cfg.wsize   = 4;
    cfg.dims    = toi';
    cfg.compute_probas = false;
    cfg.compute_predict = true;
    
    % define contrast
    angles  = deg2rad([trials.orientation]*30-15);
    x                       = 2+cos(2*angles);
    x([trials.present]==0)  = 0;
    y                       = 2+sin(2*angles);
    y([trials.present]==0)  = 0;
    
    %% decode
    results_x   = jr_classify(MEEG,x,cfg);
    results_y   = jr_classify(MEEG,y,cfg);
    
    % realign
    [trial_prop(:,:,s) predict] = decode_reg2angle(...
        results_x,....
        results_y,...
        [trials([trials.present]==1).orientation]);
    
    % plot each subject
    figure(1)
    subplot(4,5,s), hold on
    imagesc(time(cfg.dims),[],trial_prop(:,:,s));%,[0 .5]);
    
    %% generalization across time analysis
    figure(2)
    cfg.wsize   = 1;
    cfg.dims_tg = 0;
    results_x   = jr_classify(MEEG,x,cfg);
    results_y   = jr_classify(MEEG,y,cfg);
    
    % realign
    [trial_prop_tg(:,:,:,s) predict] = decode_reg2angle(...
        results_x,....
        results_y,...
        [trials([trials.present]==1).orientation]);
    
    subplot(4,5,s), hold on
    imagesc(time(cfg.dims),time(cfg.dims),squeeze(trial_prop_tg(round(end/2),:,:,s)));
    set(gca,'ydir', 'normal');axis image;
    
    % plot p value
    figure(3)
    subplot(4,5,s), hold on
    [p(:,:,:,s) z] = circ_rtest(2*predict-pi);
    imagesc(time(cfg.dims),time(cfg.dims),squeeze(-log10(p(:,:,:,s))));
    set(gca,'ydir', 'normal');axis image;
end

%% plot across subjects
% plot EOG
figure()
imagesc(time(cfg.dims),[-90:30:90],nanmedian(trial_prop,3))
set(gca,'FontSize',24,'YTick',[-90:30:90]),xlabel('time'),ylabel('distance from target'),
title('decoding from EOG')
saveas(gcf,[path_images 'EOG.jpg']);

% plot time gen
figure()
imagesc(time(cfg.dims),time(cfg.dims),squeeze(nanmedian(trial_prop_tg(round(end/2),:,:,:),4)));
set(gca,'ydir', 'normal');axis image;
set(gca,'FontSize',24),xlabel('testing time'),ylabel('training time'),
title('decoding from EOG tg')
saveas(gcf,[path_images 'EOG_tg.jpg']);

% plot p vals
figure()
imagesc(time(cfg.dims),time(cfg.dims),squeeze(-log10(nanmedian(p,4))))
set(gca,'ydir', 'normal');axis image;
set(gca,'FontSize',24),xlabel('testing time'),ylabel('training time'),
title('decoding from EOG tg (p vals in -log10 scale)')
saveas(gcf,[path_images 'EOG_tg_p.jpg']);