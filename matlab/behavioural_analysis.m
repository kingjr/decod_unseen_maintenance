%% behavioural analysis

clear all
close all
clc

%% libraries & toolboxes
setup_paths

%% All subjects
setup_subjects;

%% concatenate variables
for s = 1 : length(SubjectsList)
    sbj_initials = SubjectsList{s};
    data_path = [path 'data/' sbj_initials '/'] ;
    load([data_path 'behavior/' sbj_initials '.mat'],'trials')
    
    %% accuracy 
    accuracy(s,:) = [trials.correct];
    %% visibility
    visibility(s,:) = [trials.response_visibilityCode];
    %% contrast
    contrast(s,:) = [trials.contrast];
    %% orientation
    orientation(s,:) = [trials.orientation];
    %% present
    present(s,:) = [trials.present];
    %% rt
    rt(s,:) = [trials.response_RT];
    %% rt2
    rt2(s,:) = [trials.response_vis_RT];
end

%% proportion of invisible stimuli
for v =1:4
    vis_prop(:,v) = sum(visibility==v,2) ./ sum(~isnan(visibility),2);
end
% plot
sem_errorbarbar(vis_prop);
set(gca,'xticklabel',{'invisible' 'brief glimpse' 'almost clear' 'clear'});
ylabel('frequency (%)')

% save
plot2svg
close gcf

%% 2 way (visibility * contrast) ANOVA on accuracy
% (not enough cells)
i = 0;clear y_acc y_rt cont vis subj
exclude_subject = [];
for s = 1:20
    for v = 1:4
        for c = 1:3
            % update counter
            i=i+1;
            
            % define trials of interest
            toi = contrast(s,:) == c+1 & visibility(s,:)== v ;
            
            if sum(toi)<2,
                warning(['not enough trials in (s,v,c): ' num2str([s v c])]);
                exclude_subject = [exclude_subject s];
            end
            
            % dependent variable
            y_acc(i)    = nanmean(accuracy(s,toi));
            y_rt(i)     = nanmean(rt(s,toi));
            
            % plotting variable
            P_acc(s,v,c) = nanmean(accuracy(s,toi));
            P_rt(s,v,c)  = nanmean(rt(s,toi));
            
            % predictors
            cont(i)     = c;
            vis(i)      = v;
            subj(i)     = s;
        end
    end
end
% -------- ACCURACY -------------------------------------------------------
% plot accuracy * visibility and contrast
sem_errorbarbar(P_acc);
set(gca,'xticklabel',{'invisible' 'brief glimpse' 'almost clear' 'clear'});
legend('contrast .5','contrast .75','contrast 1','location','southeast')
ylabel('accuracy')

% save 
plot2svg

% anova on accuracy
toi = ~ismember(subj,exclude_subject);
[p table stats] = anovan(y_acc(toi),{cont(toi) vis(toi) subj(toi)},'varnames', {'contrast', 'visibility' 'subjects'}, ...
    'model','full','random',3)

%------------------- planned comparisons ----------------------------------
soi = ones(1,20);
soi(unique(exclude_subject)) = 0;soi=logical(soi);

% 1way anova (visibility) on accuracy
i = 0;clear y_acc y_rt
for s = 1:20
    for v = 1:4
        i=i+1;
        
        toi = visibility(s,:) == v;
        
        % dependent variable
        y_acc(s,v) = nanmean(accuracy(s,toi));
    end
end
% 1 way anova
anova1(y_acc(soi,:))

% planned ttests on averages across visibilities (no contrast because not significant)
[h p ci stats] = ttest(y_acc(soi,1),.5,'tail','right')
[h p ci stats] = ttest(y_acc(soi,1),y_acc(soi,2))
[h p ci stats] = ttest(y_acc(soi,1),y_acc(soi,3))
[h p ci stats] = ttest(y_acc(soi,1),y_acc(soi,4))
[h p ci stats] = ttest(y_acc(soi,2),y_acc(soi,3))
[h p ci stats] = ttest(y_acc(soi,2),y_acc(soi,4))
[h p ci stats] = ttest(y_acc(soi,3),y_acc(soi,4))


% -------- RT -------------------------------------------------------------
% plot RT * visibility and contrast
sem_errorbarbar(P_rt);
set(gca,'xticklabel',{'invisible' 'brief glimpse' 'almost clear' 'clear'});
legend('contrast .5','contrast .75','contrast 1','location','southeast')
ylabel('RT (sec)')

% save 
plot2svg

% anova on RT
toi = ~ismember(subj,exclude_subject);
[p table stats] = anovan(y_rt(toi),{cont(toi) vis(toi) subj(toi)},'varnames', {'contrast', 'visibility' 'subjects'}, ...
    'model','full','random',3)

%------------------- planned comparisons ----------------------------------
soi = zeros(1,20);
soi(unique(exclude_subject)) = 1;soi=logical(soi);

% 1way anova (visibility) on RT
i = 0;clear y_acc y_rt
for s = 1:20
    for v = 1:4
        i=i+1;
        
        toi = visibility(s,:) == v;
        
        % dependent variable
        y_rt(s,v) = nanmean(rt(s,toi));
    end
end
% 1-WAY VISIBILITY ON RT
% plot
sem_errorbarbar(y_rt)
set(gca,'xticklabel',{'invisible' 'brief glimpse' 'almost clear' 'clear'});
ylabel('RT (sec)')
% save figure
plot2svg

% compute anova
anova1(y_rt)

% planned comparisons
[h p ci stats] = ttest(y_rt(soi,1),y_rt(soi,2))
[h p ci stats] = ttest(y_rt(soi,1),y_rt(soi,3))
[h p ci stats] = ttest(y_rt(soi,1),y_rt(soi,4))
[h p ci stats] = ttest(y_rt(soi,2),y_rt(soi,3))
[h p ci stats] = ttest(y_rt(soi,2),y_rt(soi,4))
[h p ci stats] = ttest(y_rt(soi,3),y_rt(soi,4))

% 1way ANOVA CONTRAST ON RT
i = 0;clear y_acc y_rt
for s = 1:20
    for c = 1:3
        i=i+1;
        
        toi = contrast(s,:) == c + 1;
        
        % dependent variable
        y_rt(s,c) = nanmean(rt(s,toi));
    end
end
% plot
sem_errorbarbar(y_rt)
set(gca,'xticklabel',{'contrast .5' 'contrast .75' 'contrast 1'});
ylabel('RT (sec)')
% save figure
plot2svg

% 1-wy anova (contrasts) on RTs
anova1(y_rt(soi,:))

% planned ttests on averages across contrasts 
[h p ci stats] = ttest(y_rt(soi,1),y_rt(soi,2))
[h p ci stats] = ttest(y_rt(soi,1),y_rt(soi,3))
[h p ci stats] = ttest(y_rt(soi,2),y_rt(soi,3))


% explore interaction term
[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,1,2))
[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,1,3))

[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,2,2))
[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,2,3))

[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,3,2))
[h p ci stats] = ttest(P_rt(soi,1,1),P_rt(soi,3,3))


%% present/absent visibility
clear y
for s = 1: 20
    toi = visibility(s,:)==1;
    y(s,1) = nanmean(present(s,toi));
    
    toi = visibility(s,:)>1;
    y(s,2) = nanmean(present(s,toi));
end
errorbarbar(nanmean(y),sem(y))


%% VISIBILITY as a function of CONTRAST and ORIENTATION
i = 0;clear y_vis cont or subj P_vis
for s = 1:20
    for c = 1:4
        for o = 1:6
            % update counter
            i=i+1;
            
            % define trials of interest
            toi = contrast(s,:) == c & orientation(s,:)==o;
            
            % warning in case of empty cells
            if sum(toi)<2,
                warning(['not enough trials in (s,c,o): ' num2str([s c o])]);
                exclude_subject = [exclude_subject s];
            end
            
            % dependent variable
            y_vis(i)    = nanmean(visibility(s,toi));
            
            % plotting variable
            P_vis(s,c,o) = y_vis(i);
            
            %predictors
            cont(i)     = c;
            or(i)       = o;
            subj(i)     = s;
        end
    end
end
% plot and save 
sem_errorbarbar(P_vis);
set(gca,'xticklabel',{'absent' 'contrast .5' 'contrast .75' 'contrast 1'});
ylabel('visibility')
legend('15\circ','45\circ','75\circ','105\circ','135\circ','165\circ','location','southeast')
plot2svg

% 2-way ANOVA (contrast and orientation) on visibility
[p table stats] = anovan(y_vis,{cont or subj},'varnames', {'contrast', 'orientation' 'subjects'}, ...
    'model','full','random',3)

% --------------planned comparisons ---------------------------------------
% 1way ANOVA CONTRAST ON VISIBILITY
i = 0;clear y_vis
for s = 1:20
    for c = 1:4
        i=i+1;
        
        toi = contrast(s,:) == c;
        
        % dependent variable
        y_vis(s,c) = nanmean(visibility(s,toi));
    end
end
% plot
sem_errorbarbar(y_vis)
set(gca,'xticklabel',{'absent' 'contrast .5' 'contrast .75' 'contrast 1'});
ylabel('visibility')
% save figure
plot2svg

% 1-wy anova (contrasts) on RTs
anova1(y_vis)

% planned ttests on averages across contrasts 
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,2))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,3))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,4))

[h p ci stats] = ttest(y_vis(:,2),y_vis(:,3))
[h p ci stats] = ttest(y_vis(:,2),y_vis(:,4))

[h p ci stats] = ttest(y_vis(:,3),y_vis(:,4))


% 1way ANOVA ORIENTATION ON VISIBILITY
i = 0;clear y_vis
for s = 1:20
    for o = 1:6
        i=i+1;
        
        toi = orientation(s,:) == o;
        
        % dependent variable
        y_vis(s,o) = nanmean(visibility(s,toi));
    end
end
% plot
sem_errorbarbar(y_vis)
set(gca,'xticklabel',{'15','45','75','105','135','165'});
ylabel('visibility')
% save figure
plot2svg

% 1-wy anova (contrasts) on RTs
anova1(y_vis)

% planned ttests on averages across contrasts 
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,2))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,3))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,4))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,5))
[h p ci stats] = ttest(y_vis(:,1),y_vis(:,6))

[h p ci stats] = ttest(y_vis(:,2),y_vis(:,3))
[h p ci stats] = ttest(y_vis(:,2),y_vis(:,4))
[h p ci stats] = ttest(y_vis(:,2),y_vis(:,5))
[h p ci stats] = ttest(y_vis(:,2),y_vis(:,6))

[h p ci stats] = ttest(y_vis(:,3),y_vis(:,4))
[h p ci stats] = ttest(y_vis(:,3),y_vis(:,5))
[h p ci stats] = ttest(y_vis(:,3),y_vis(:,6))

[h p ci stats] = ttest(y_vis(:,4),y_vis(:,5))
[h p ci stats] = ttest(y_vis(:,4),y_vis(:,6))

[h p ci stats] = ttest(y_vis(:,5),y_vis(:,6))
