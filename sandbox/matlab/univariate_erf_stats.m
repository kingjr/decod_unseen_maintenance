%% load data
clear data MEEG trials
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
load([path 'data/' subject '/behavior/' subject '_fixed.mat'], 'trials');
MEEG = binload([path 'data/' subject '/preprocessed/' subject '_preprocessed.dat'], data.Xdim);

%% prepare saving
if ~exist([path 'data/' subject '/univariate/'],'dir')
    mkdir([path 'data/' subject '/univariate/']);
end
data_path = [path 'data/' subject '/univariate/' subject '_univariate'];
labels = data.label;
time = data.time{1};


% %% ================== DEFINE UNIVARIATE STATS HERE ========================
% 
% %% Angles: ANOVA
% clear p stats X y;
% X = MEEG([trials.present]==1,1:306,:);
% y = [trials([trials.present]==1).orientation]';
% 
% for chan = 306:-1:1
%     for t = length(data.time{1}):-1:1
%         [p(chan,t),~,stats(chan,t)] = anovan(X(:,chan,t),{y},'display','off');
%     end
% end
% angleAnova.p = p;
% angleAnova.stats = stats;
% save([data_path '_angleAnova.mat'],'angleAnova', 'time', 'labels');
% 
% %% Angles: circular linear (see Berens : http://www.jstatsoft.org/v31/i10)
% clear p rho X y;
% X = MEEG([trials.present]==1,1:306,:);
% y = [trials([trials.present]==1).orientation]';
% 
% [rho p] = circ_corrcl(...
%     repmat(2*deg2rad(y*30-15),[1,size(X,2),size(X,3)]),...
%     X);
% 
% angleCirc.rho = rho;
% angleCirc.p = p;
% save([data_path '_angleCirc.mat'],'angleCirc', 'time', 'labels');
% 
% %% ----------------- BINARY CONTRASTS -----------------------------------%%
% 
% %% Seen versus Unseen (binary contrast)
% clear p h stats
% [p h stats] = ranksum_fast(...
%     MEEG([trials.response_visibilityCode]>1 & [trials.present]==1,1:306,:),...
%     MEEG([trials.response_visibilityCode]==1 & [trials.present]==1,1:306,:));
% seenUnseen.p = p;
% seenUnseen.stats = stats;
% save([data_path '_seenUnseen.mat'],'seenUnseen', 'time', 'labels');
% 
% %% Present versus Absent (binary contrast)
% clear p h stats
% [p h stats] = ranksum_fast(...
%     MEEG([trials.present]==1,1:306,:),...
%     MEEG([trials.present]==0,1:306,:));
% presentAbsent.p = p;
% presentAbsent.stats = stats;
% save([data_path '_presentAbsent.mat'],'presentAbsent', 'time', 'labels');
% 
% %% Seen versus Unseen x Present versus Absent (interaction ANOVA)
% clear p stats X y;
% X = MEEG(:,1:306,:);
% y_present = [trials.present]';
% y_seen = [trials.response_visibilityCode]'>1;
% for chan = 306:-1:1
%     for t = length(data.time{1}):-1:1
%         [p(chan,t,:),~,stats(chan,t)] = anovan(X(:,chan,t),{y_present y_seen},'display','off', 'model', 'interaction');
%     end
% end
% presentXseen.p = p;
% presentXseen.stats = stats;
% save([data_path '_presentXseen.mat'],'presentXseen', 'time', 'labels');
% 
% %% Left vs Right finger
% clear p h stats
% [p h stats] = ranksum_fast(...
%     MEEG([trials.response_tilt]==-1,1:306,:),...
%     MEEG([trials.response_tilt]==1,1:306,:));
% leftRightResponse.p = p;
% leftRightResponse.stats = stats;
% save([data_path '_leftRightResponse.mat'],'leftRightResponse', 'time', 'labels');

%% define here all binary contrasts 

%% Tilt
clear p h stats
[p h stats] = ranksum_fast(...
    MEEG([trials.tilt]==-1,1:306,:),...
    MEEG([trials.tilt]==1,1:306,:));
tilt.p = p;
tilt.stats = stats;
save([data_path '_tilt.mat'],'tilt', 'time', 'labels');
clear tilt

%% Correct vs Incorrect
clear p h stats
[p h stats] = ranksum_fast(...
    MEEG([trials.correct]==0,1:306,:),...
    MEEG([trials.correct]==1,1:306,:));
correctIncorrect.p = p;
correctIncorrect.stats = stats;
save([data_path '_correctIncorrect.mat'],'correctIncorrect', 'time', 'labels');
clear correctIncorrect

%% Lambda
clear p h stats
[p h stats] = ranksum_fast(...
    MEEG([trials.lambda]==1,1:306,:),...
    MEEG([trials.lambda]==2,1:306,:));
lambda.p = p;
lambda.stats = stats;
save([data_path '_lambda.mat'],'lambda', 'time', 'labels');
clear lambda

%% Orientation X Lambda (interaction ANOVA)
clear p stats X y;
X = MEEG([trials.present]==1,1:306,:);
y_orientation = [trials.orientation]';
y_orientation([trials.present]==0) =[];

y_lambda = [trials.lambda]';
y_lambda([trials.present]==0) =[];

for chan = 306:-1:1
    for t = length(data.time{1}):-1:1
        [p(chan,t,:),~,stats(chan,t)] = anovan(X(:,chan,t),{y_orientation y_lambda},'display','off', 'model', 'interaction');
    end
end
orientationXlambda.p = p;
orientationXlambda.stats = stats;
save([data_path '_orientationXlambda.mat'],'orientationXlambda', 'time', 'labels');
clear orientationXlambda

%% Orientation X Visibility (interaction ANOVA)
clear p stats X y;
X = MEEG([trials.present]==1,1:306,:);
y_orientation = [trials.orientation]';
y_orientation([trials.present]==0) =[];

y_visibility = [trials.response_visibilityCode]';
y_visibility([trials.present]==0) =[];

for chan = 306:-1:1
    for t = length(data.time{1}):-1:1
        [p(chan,t,:),~,stats(chan,t)] = anovan(X(:,chan,t),{y_orientation y_visibility},'display','off', 'model', 'interaction');
    end
end
orientationXvisibility.p = p;
orientationXvisibility.stats = stats;
save([data_path '_orientationXvisibility.mat'],'orientationXvisibility', 'time', 'labels');
clear orientationXvisibility

%% Visibility X Present (interaction ANOVA)
clear p stats X y;
X = MEEG(:,1:306,:);
y_visibility = [trials.response_visibilityCode]';
y_present    = [trials.present]';

for chan = 306:-1:1
    for t = length(data.time{1}):-1:1
        [p(chan,t,:),~,stats(chan,t)] = anovan(X(:,chan,t),{y_visibility y_present},'display','off', 'model', 'interaction');
    end
end
visibilityXpresent.p = p;
visibilityXpresent.stats = stats;
save([data_path '_visibilityXpresent.mat'],'visibilityXpresent', 'time', 'labels');
clear visibilityXpresent
