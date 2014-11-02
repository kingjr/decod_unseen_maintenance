%% subjects_demographics
cd('F:\Paris\scripts');
clear all
close all
clc

%% libraries & toolboxes
setup_paths

%% All subjects
setup_subjects;


%% demographics 
for s = 1 : length(SubjectsList)
    sbj_initials = SubjectsList{s};
    data_path = [path 'data/' sbj_initials '/'] ;
    load([data_path 'behavior/' sbj_initials '.mat'],'subject')
    
    age(s) = subject.age;
    rhand(s) = subject.right_handed;
    male(s) = subject.male;
    
end

mean(age);std(age);sum(male);sum(rhand);

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

%% accuracy
%2 way on contrast : not enough cells
i = 0;clear y_acc y_rt cont vis subj
for s = 1:20
    for v = 1:4
        for c = 1:3
            
            toi = contrast(s,:) == c+1 & visibility(s,:)== v ;
            
            i=i+1;
            % dependent variable
            y_acc(i) = nanmean(accuracy(s,toi));
            y_rt(i) = nanmean(rt(s,toi));
            
            %preds
            cont(i) = c;
            vis(i) = v;
            subj(i) = s;
        end
    end
end
[p table stats] = anovan(y_acc,{cont vis subj},'varnames', {'contrast', 'visibility' 'subjects'}, ...
    'model','full','random',3)


%1 way on contrast
i = 0;clear y
for s = 1:20
    for c = 1:3
        i=i+1;
        
        toi = contrast(s,:) == c+1;
        
        % dependent variable
        y(s,c) = nanmean(accuracy(s,toi));
    end
end
% 1 way anova
z = zscore(y,0,2);
anova1(z)
my_errorbar(y)


% 1way anova on visibility
i = 0;clear y_acc y_rt
for s = 1:20
    for v = 1:4
        i=i+1;
        
        toi = visibility(s,:) == v;
        
        % dependent variable
        y_acc(s,v) = nanmean(accuracy(s,toi));
        y_rt(s,v)  = nanmean(rt(s,toi));
    end
end
% 1 way anova
z = zscore(y_acc,0,2);
anova1(z)
my_errorbar(y_acc)

anova1(y_rt)

% present/absent visibility
clear y
for s = 1: 20
    toi = visibility(s,:)==1;
    y(s,1) = nanmean(present(s,toi));
    
    toi = visibility(s,:)>1;
    y(s,2) = nanmean(present(s,toi));
end
errorbarbar(nanmean(y),sem(y))


% visibility predicted by contrast and orientation
i = 0;clear y_vis cont or subj
for s = 1:20
    for c = 1:4
        for o = 1:6
            i=i+1;
            
            toi = contrast(s,:) == c & orientation(s,:)==o;
            
            % dependent variable
            y_vis(i) = nanmean(visibility(s,toi));
            
            %preds
            cont(i) = c;
            or(i) = o;
            subj(i) = s;
        end
    end
end
[p table stats] = anovan(y_vis,{cont or subj},'varnames', {'contrast', 'orientation' 'subjects'}, ...
    'model','full','random',3)

