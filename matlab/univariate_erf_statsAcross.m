

save_univariate = @(var_name) save([path 'data/across_subjects/all_univariateAcross_' var_name '.mat'],'var_name');

%% ------ DEFINE UNIVARIATE STATS ACROSS SUBJECTS (first order) HERE ------

%% angles circular linear : don't know yet!

%% Angles anova
% load data across subjects
var_name = 'angles';
clear all_angles
for s = length(SubjectsList):-1:1
    clear angles
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_angles(s,:,:,:) = angles(1:306,:,:); % subjects x channels x time x angle_category
end

% apply ANOVA
%----- a bit long... (15 min)
%----- to use matrices instead of loop have a look at:
%----- http://www.mathworks.com/matlabcentral/fileexchange/27960-resampling-statistical-toolkit 
clear p stats X y;
for chan = 306:-1:1
    for t = size(all_angles,3):-1:1
        % transform selected data into unidimensional vector (subjects * angle)
        X = reshape(all_angles(:,chan,t,:),[],1);
        % create the regressor: 
        angles = repmat((1:6)',20,1);               % all angles, for each subject
        subjects= reshape(repmat(1:20,6,1),[],1);   % subjects id, for each subject
        % apply ANOVA: random factor /!\ across subjects only /!\
        [p(chan,t,:),~,stats(chan,t)] = anovan(X,{angles,subjects},'display','off', 'random',2, 'model', 'linear');
    end
end
anglesAnova.p = p;
anglesAnova.stats = stats;
save_univariate('anglesAnova');

%% Contrast anova
% load data across subjects
var_name = 'contrast';
clear all_contrast
for s = length(SubjectsList):-1:1
    clear contrast
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_contrast(s,:,:,:) = contrast(1:306,:,:); % subjects x channels x time x contrast
end

% apply ANOVA
%----- a bit long... (15 min)
%----- to use matrices instead of loop have a look at:
%----- http://www.mathworks.com/matlabcentral/fileexchange/27960-resampling-statistical-toolkit 
clear p stats X y;
for chan = 306:-1:1
    for t = size(all_contrast,3):-1:1
        % transform selected data into unidimensional vector (subjects * contrast)
        X = reshape(all_contrast(:,chan,t,:),[],1);
        % create the regressor: 
        contrast = repmat((1:4)',20,1);               % all contrasts, for each subject
        subjects= reshape(repmat(1:20,4,1),[],1);   % subjects id, for each subject
        % apply ANOVA: random factor /!\ across subjects only /!\
        [p(chan,t,:),~,stats(chan,t)] = anovan(X,{contrast,subjects},'display','off', 'random',2, 'model', 'linear');
    end
end
contrastAnova.p = p;
contrastAnova.stats = stats;
save_univariate('contrastAnova');

%% Visibility anova
% load data across subjects
var_name = 'visibility';
clear all_visibility
for s = length(SubjectsList):-1:1
    clear visibility
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_visibility(s,:,:,:) = visibility(1:306,:,:); % subjects x channels x time x visibility
end

% apply ANOVA
%----- a bit long... (15 min)
%----- to use matrices instead of loop have a look at:
%----- http://www.mathworks.com/matlabcentral/fileexchange/27960-resampling-statistical-toolkit 
clear p stats X y;
for chan = 306:-1:1
    for t = size(all_visibility,3):-1:1
        % transform selected data into unidimensional vector (subjects * contrast)
        X = reshape(all_visibility(:,chan,t,:),[],1);
        % create the regressor: 
        visibility = repmat((1:4)',20,1);               % all visibilities, for each subject
        subjects= reshape(repmat(1:20,4,1),[],1);   % subjects id, for each subject
        % apply ANOVA: random factor /!\ across subjects only /!\
        [p(chan,t,:),~,stats(chan,t)] = anovan(X,{visibility,subjects},'display','off', 'random',2, 'model', 'linear');
    end
end
visibilityAnova.p = p;
visibilityAnova.stats = stats;
save_univariate('visibilityAnova');

%% Present versus absent
% load data across subjects
var_name = 'presentAbsent';
clear all_presentAbsent
for s = length(SubjectsList):-1:1
    clear presentAbsent
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_presentAbsent(s,:,:,:) = presentAbsent(1:306,:,:); % subjects x channels x time x present_absent
end
% apply Wilcoxon test 
%---- (non-parametric) paired test (first order univariate stats across subject)
[p h stats] = signrank_matrix(all_presentAbsent(:,:,:,1),all_presentAbsent(:,:,:,2));
presentVSabsent.p = p;
presentVSabsent.stats = stats;
save_univariate('presentVSabsent');


%% Seen versus Useen (with only present trials)
% load data across subjects
var_name = 'seenUnseen';
clear all_seenUnseen
for s = length(SubjectsList):-1:1
    clear seenUnseen
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_seenUnseen(s,:,:,:) = seenUnseen(1:306,:,:); % subjects x channels x time x present_absent
end
% apply Wilcoxon test 
[p h stats] = signrank_matrix(all_seenUnseen(:,:,:,1),all_seenUnseen(:,:,:,2));
seenVSunseen.p = p;
seenVSunseen.stats = stats;
save_univariate('seenVSunseen');

%% correct versus incorrect (with only present trials)
% load data across subjects
var_name = 'correctIncorrect';
clear all_correctIncorrect
for s = length(SubjectsList):-1:1
    clear correctIncorrect
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_correctIncorrect(s,:,:,:) = correctIncorrect(1:306,:,:); % subjects x channels x time x correctIncorrect
end
% apply Wilcoxon test 
[p h stats] = signrank_matrix(all_correctIncorrect(:,:,:,1),all_correctIncorrect(:,:,:,2));
correctVSincorrect.p = p;
correctVSincorrect.stats = stats;
save_univariate('correctVSincorrect');

%% Response button
% load data across subjects
var_name = 'responseTilt';
clear all_responseTilt
for s = length(SubjectsList):-1:1
    clear responseTilt
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_responseTilt(s,:,:,:) = responseTilt(1:306,:,:); % subjects x channels x time x responseTilt
end
% apply Wilcoxon test 
[p h stats] = signrank_matrix(all_responseTilt(:,:,:,1),all_responseTilt(:,:,:,2));
responseTilt.p = p;
responseTilt.stats = stats;
save_univariate('responseTilt');

%% Tilt
% load data across subjects
var_name = 'tilt';
clear all_tilt
for s = length(SubjectsList):-1:1
    clear tilt
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_tilt(s,:,:,:) = tilt(1:306,:,:); % subjects x channels x time x tilt
end
% apply Wilcoxon test 
[p h stats] = signrank_matrix(all_tilt(:,:,:,1),all_tilt(:,:,:,2));
Tilt.p = p;
Tilt.stats = stats;
save_univariate('Tilt');

%% Lambda
% load data across subjects
var_name = 'lambda';
clear all_lambda
for s = length(SubjectsList):-1:1
    clear lambda
    load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'],var_name);
    all_lambda(s,:,:,:) = lambda(1:306,:,:); % subjects x channels x time x responseTilt
end
% apply Wilcoxon test 
[p h stats] = signrank_matrix(all_lambda(:,:,:,1),all_lambda(:,:,:,2));
lambda.p = p;
lambda.stats = stats;
save_univariate('lambda');

%% Interactions
%% Visibility X Present (interaction ANOVA)

%% same for the rest, I'll let you do it
