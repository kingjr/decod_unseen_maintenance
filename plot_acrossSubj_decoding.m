%% compute average across subjects
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
%% SVC classic
contrasts = {'targetAngle' 'probeAngle' 'lambda', 'responseButton','tilt', 'visibility',...
    'visibilityPresent', 'presentAbsent', 'accuracy'}; % may increase over time
colors = colorGradient([1 0 0], [0 1 0],4);
for c =1:length(contrasts)
    clear all_probas all_probasb all_probas_vis
    for s = length(SubjectsList):-1:1
        %% load individual data
        subject = SubjectsList{s};
        data_path = [path 'data/' subject '/'] ;
        file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
        load(file_behavior, 'trials');
        file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
        load(file_header, 'data');
        file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
        time    = data.time{1}; % in s

        results = load([data_path 'mvpas/' subject '_preprocessed_' contrasts{c} '_SVC_results.mat']); % classifier (and file) name
        
        % define division of interest
        visible = [trials([trials.present]==1).response_visibilityCode]';
        
        %% concatenate in one matrix
        % plot
        % realign angles to the 4th line
        switch contrasts{c}
            case 'targetAngle'
                % ---------------------   all trials  --------------------
                probas = [];
                rads = unique(results.y); %angles in radiants
                for a = 1:6
                    sel = results.y==rads(a);
                    probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
                end
                all_probas(s,:,:) = squeeze(trimmean(probas(:,:,[4:6 1:4]),80))';

                %---------for visibility   -------------------------------
                invis = visible == 1;
                vis   = visible > 1;
                all_probas_vis(s,:,:,2) = squeeze(trimmean(probas(vis,:,[4:6 1:4]),80,'round',1))';
                all_probas_vis(s,:,:,1) = squeeze(trimmean(probas(invis,:,[4:6 1:4]),80,'round',1))';
                
%                 % same depending on contrast
%                 con = [trials.contrast];
%                 con([trials.present]==0) = [];
%                 for c = 1:4
%                     all_probas_cont(s,:,:,c) = squeeze(trimmean(results.probas(1,con==c,:,:,aorder),80))';
%                 end
                
            case 'probeAngle'
                %-------------   all trials   -----------------------------
                probas = [];
                % realign angles to the 4th line
                angles = [15:30:165];
                for a = 1:6
                    sel = results.y==angles(a);
                    probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
                end
                all_probas(s,:,:) = squeeze(trimmean(probas(:,:,[4:6 1:4]),80))';
                
            otherwise
                % all trials
                [~,~,class] = unique(results.y);
                all_probasb(s,:,1) = nanmean(squeeze(results.probas(1,class==1,:,1,1))); % binary: class 1
                all_probasb(s,:,2) = nanmean(squeeze(results.probas(1,class==2,:,1,1))); % binary: class 2
                
                % sort by visibility
                [~,~,class] = unique(results.y);
                for v =1:4
                    %plot only class 1
                    all_probasb_vis(s,:,v,1) = nanmean(squeeze(results.probas(1,class==1 & [trials.response_visibilityCode]'==v,:,1,1))); % binary: class 1
                    %plot only class 2
                    all_probasb_vis(s,:,v,2) = nanmean(squeeze(results.probas(1,class==2 & [trials.response_visibilityCode]'==v,:,1,1))); % binary: class 2
                end
        end
    end
    
    %% plot across subjects
    figure;
    if strcmp(contrasts{c},'targetAngle') | strcmp(contrasts{c},'probeAngle')
        imagesc(time(cfg.dims),[],squeeze(nanmean(all_probas)));
        xlabel('time (ms)');ylabel('angle distance');
        set(gca,'FontSize',24,'YTickLabel',90:-30:-90);colorbar;
        title(contrasts{c})
        saveas(gcf,[path_images contrasts{c} '.jpg']);
        
        % ------------ divide by visibility  -----------------------------
        figure()
        for sp = 1:2
            subplot(1,2,sp)
            imagesc(time(cfg.dims),[],squeeze(nanmean(all_probas_vis(:,:,:,sp),1)));
            xlabel('time (ms)');ylabel('angle distance');
            set(gca,'FontSize',24,'YTickLabel',90:-30:-90);
            title(contrasts{c})
            saveas(gcf,[path_images contrasts{c} '_visInv.jpg']);
        end
        
    else
        % all trials 
        clf;hold on;
        plot_eb(time(results.dims),squeeze(all_probasb(:,:,1)),[0 0 1]); % class 1
        plot_eb(time(results.dims),squeeze(all_probasb(:,:,2)),[1 0 0]); % class 2
        set(gca,'FontSize',24),xlabel('time'),ylabel('prob')
        title(contrasts{c})
        saveas(gcf,[path_images contrasts{c} '.jpg']);
        
        % sort by visibility
        clf;hold on;
        for v = 1:4
            % plot class 1 only according to visibility
            [hl(v)]=plot_eb(time(results.dims),squeeze(all_probasb_vis(:,:,v,1)),'color',colors(v,:),'LineStyle','-');
            % plot class 2 only according to visibility
            plot_eb(time(results.dims),squeeze(all_probasb_vis(:,:,v,2)),'color',colors(v,:),'LineStyle','--');
        end
        set(gca,'FontSize',24),xlabel('time'),ylabel('prob(absent)')
        title(contrasts{c})
        legend(cell2mat(hl),'vis0','vis1','vis2','vis3')
        set(gcf,'Position',get(0,'ScreenSize')); % maximize figure
        saveas(gcf,[path_images contrasts{c} '_vis.jpg']);
    end
    
end




%% SVR: classic
contrasts   = {'targetAngle', 'probeAngle'};
timeg       = {'' '_t184' '_t309' '_t981' '_tAll'};

for c =1:length(contrasts)
    for tg = 1:5
        clear all_predict all_pval all_prop_invis all_prop_vis
        for s = length(SubjectsList):-1:1
            clear train_angles trial_prop trial_prop_invis trial_prop_vis
            %% load individual data
            subject = SubjectsList{s};
            data_path = [path 'data/' subject '/'] ;
            file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
            load(file_behavior, 'trials');
            file_header     = [data_path 'preprocessed/' subject '_preprocessed.mat'];
            load(file_header, 'data');
            file_binary     = [data_path 'preprocessed/' subject '_preprocessed.dat'];
            time    = data.time{1}; % in s
            
            
            results_x = load([data_path 'mvpas/' subject '_preprocessed_' contrasts{c} '_SVR' timeg{tg} '_x_results.mat']); % classifier (and file) name
            results_y = load([data_path 'mvpas/' subject '_preprocessed_' contrasts{c} '_SVR' timeg{tg} '_y_results.mat']); % classifier (and file) name
            
            % define division of interest
            visible = [trials([trials.present]==1).response_visibilityCode];
                    
            %% relign predictions
            switch contrasts{c}
                case 'targetAngle';
                    [trial_prop predict] = decode_reg2angle(...
                        results_x,....
                        results_y,...
                        [trials([trials.present]==1).orientation]);
                case 'probeAngle'
                    [trial_prop predict] = decode_reg2angle(...
                        results_x,....
                        results_y,...
                        mod([trials.orientation]+[trials.tilt]-1,6)+1);
            end
            
            %% concatenate subjects
          
            switch timeg{tg}
                case '_tAll'
                    %------all trials -------------------------------------
                    % plot gentime on closest angle distance
                    all_predict(s,:,:) = squeeze(trial_prop(round(end/2),:,:));
                    
                    % plot p value
                    [p z] = circ_rtest(2*predict-pi);
                    all_pval(s,:,:) = squeeze(-log10(p));
                    
                    %------ separate visible versus invisible   -----------
                    % invisible
                    train_angles = [trials([trials.present]==1).orientation];
                    train_angles(visible>1) = NaN;
                    [trial_prop_invis predict] = decode_reg2angle(results_x,results_y,train_angles);
                    all_prop_invis(s,:,:) = squeeze(trial_prop_invis(round(end/2),:,:));
                    
                    % visible
                    train_angles = [trials([trials.present]==1).orientation];
                    train_angles(visible<=1) = NaN;
                    [trial_prop_vis predict] = decode_reg2angle(results_x,results_y,train_angles);
                    all_prop_vis(s,:,:) = squeeze(trial_prop_vis(round(end/2),:,:));
                    
                otherwise
                    %------all trials -------------------------------------
                    all_predict(s,:,:) = trial_prop;%,[0 .5]);
                    
                    %------ separate visible versus invisible   -----------
                    % invisible
                    train_angles = [trials([trials.present]==1).orientation];
                    train_angles(visible>1) = NaN;
                    [trial_prop_invis predict] = decode_reg2angle(results_x,results_y,train_angles);
                    all_prop_invis(s,:,:) = trial_prop_invis;
                    
                    % visible
                    train_angles = [trials([trials.present]==1).orientation];
                    train_angles(visible<=1) = NaN;
                    [trial_prop_vis predict] = decode_reg2angle(results_x,results_y,train_angles);
                    all_prop_vis(s,:,:) = trial_prop_vis;
                    
            end
            
        end
        
        %% plot
        switch timeg{tg}
            case '_tAll'
                %-----------all trials   ----------------------------------
                figure()
                imagesc(time(cfg.dims),time(cfg.dims),squeeze(nanmean(all_predict,1)));
                set(gca,'ydir', 'normal');xlabel('test time');ylabel('train time')
                title([contrasts{c} 'time gen. : tAll'])
                saveas(gcf,[path_images 'SVR_' contrasts{c} '_tAll.jpg'])
        
                figure()
                imagesc(time(cfg.dims),time(cfg.dims),squeeze(nanmean(all_pval,1)));
                set(gca,'ydir', 'normal');xlabel('test time');ylabel('train time')
                title([contrasts{c} '(pvals); time gen. : tAll'])
                saveas(gcf,[path_images 'SVR_' contrasts{c} '_tAll_pvals.jpg'])
                
                %-------- separate visible versus invisible   -------------
                % invisible
                figure()
                subplot(1,2,1);
                imagesc(time(cfg.dims),time(cfg.dims),squeeze(nanmean(all_prop_invis,1)));
                set(gca,'ydir', 'normal');axis image;
                title([contrasts{c} '; time gen.: ' timeg{tg} '; present invisible'])
                % visible
                subplot(1,2,2);
                imagesc(time(cfg.dims),time(cfg.dims),squeeze(nanmean(all_prop_vis,1)));
                set(gca,'ydir', 'normal');axis image;
                title([contrasts{c} '; time gen.: ' timeg{tg} '; present visible'])
                saveas(gcf,[path_images 'SVR_' contrasts{c} '_tAll_visInvis.jpg']);
                
            otherwise
                %--------all trials   -------------------------------------
                figure()
                imagesc(time(cfg.dims),[],squeeze(nanmean(all_predict)));
                xlabel('time (ms)');ylabel('angle distance');
                set(gca,'FontSize',24,'YTickLabel',90:-30:-90);colorbar;
                title([contrasts{c} '; time gen. : ' timeg{tg}])
                saveas(gcf,[path_images 'SVR_' contrasts{c} timeg{tg} '.jpg'])
                
                %------ separate visible versus invisible   ---------------
                % invisible
                figure()
                subplot(1,2,1);
                imagesc(time(cfg.dims),[],squeeze(nanmean(all_prop_invis,1)));
                set(gca,'FontSize',24,'YTickLabel',90:-30:-90);
                title([contrasts{c} '; time gen.: ' timeg{tg} '; present invisible'])
                % visible
                subplot(1,2,2);
                imagesc(time(cfg.dims),[],squeeze(nanmean(all_prop_vis,1)));
                set(gca,'FontSize',24,'YTickLabel',90:-30:-90);
                title([contrasts{c} '; time gen.: ' timeg{tg} '; present visible'])
                saveas(gcf,[path_images 'SVR_' contrasts{c} timeg{tg} '_visInvis.jpg']);
        end
    end
end


%% Train on 4 visibilities discrimination (with and SVR) (present trials only), and analyze the results as a function of 
% visibility and contrasts to dissociate the two effects

% Gather all results from a single classifier trained on all dataset, and
% sort its prediction according to trials' visibility and contrast

% initialize variables
mean_vis = [];
mean_vis_c = [];
mean_vis_c_acc = [];

% loop across subjects
for s = 1:20
    % load data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    results=load([path 'data/' subject '/mvpas/' subject '_preprocessed_4visibilitiesPresent_SVR_results.mat']);
    % find the contrast for each trial
    yc = [trials.contrast]'-1;
    yc(not(results.y_all>0)) = []; % remove the absent trial from these contrasts
    % find accuracy for each trial
    yacc = [trials.correct]';
    yacc(not(results.y_all>0)) = [];
    % Loop across factor
    for vis = 1:4 % visibility
        % mean prediction for each visibility
        mean_vis(s,:,vis) = mean(results.predict(1, results.y==vis, :, 1),2);
        % gather mean results
        for c = 1:3 % contrast
            mean_vis_c(s,:,vis, c) = nanmean(results.predict(1, results.y==vis & yc==c, :, 1),2);
            for acc = 0:1 % accuracy
                % Actually, looking at the accuracy factor does not add
                % anything, so we wont keep it in the final analyses and
                % figures, as it is directly predicted by the visibility
                % factor.
                mean_vis_c_acc(s,:, vis, c, acc+1) =  nanmean(results.predict(1, results.y==vis & yc==c & yacc==acc, :, 1),2);
            end
        end
    end
end

% Plot visibility prediction as a function of time, independently of
% contrast: note than we can decode visibility early on.
figure();
colors = colorGradient([1, 0, 0], [0, 1, 0], 4);
for vis=1:4
    plot_eb(time(toi), mean_vis(:,:,vis), colors(vis, :));
    hold on;
end

% However, when plotting visibility prediction for each contrast
% separately, we notice that the decoding seems only significant from 200ms
figure();
for c = 1:3
    subplot(3,1,c);
    for vis=1:4
        plot_eb(time(toi), mean_vis_c(:,:,vis,c), colors(vis, :));
        hold on;
    end
end

%% This is confirmed by an ANOVA: the main effect of contrast starts around 100 ms
% and last for 200 ms, whereas the main effect of visibility only starts around 300 ms
ntime = length(toi);
nfactor = 3; % subjects x visibility x contrast
P = zeros(nfactor,ntime);
for t = 1:ntime
    [Y, GROUP] = prepare_anovan(squeeze(mean_vis_c(:,t,:,:)));
    P(:,t) = anovan(Y, GROUP, 'random', 1, 'model', 'linear', 'display', 'off');
end
plot(time(toi),-log10(P(2:end,:)));
legend({'visibility', 'contrast'});