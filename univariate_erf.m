%% load data
clear data MEEG trials
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
load([path 'data/' subject '/behavior/' subject '_fixed.mat'], 'trials');
MEEG = binload([path 'data/' subject '/preprocessed/' subject '_preprocessed.dat'], data.Xdim);

%% prepare saving
if ~exist([path 'data/' subject '/erf/'],'dir')
    mkdir([path 'data/' subject '/erf/']);
end
labels = data.label;
time = data.time{1};
data_path = [path 'data/' subject '/erf/' subject '_erf'];

%===================== DEFINE ERF OF INTEREST HERE ========================
%% orientations
clear ERF
for a = 6:-1:1
    sel  = [trials.orientation]==a;
    sel([trials.present]~=1) = 0;
    ERF(:,:,a) = my_mean(MEEG(sel,:,:));
end
angles = ERF;
save([data_path '_angles.mat'],'angles', 'time', 'labels');



%% present absent
clear ERF
ERF(:,:,1) = my_mean(MEEG([trials.present]==1,:,:));
ERF(:,:,2) = my_mean(MEEG([trials.present]==0,:,:));
presentAbsent = ERF;
save([data_path '_presentAbsent.mat'],'presentAbsent', 'time', 'labels');

%% visibility
clear ERF
for v = 4:-1:1
    ERF(:,:,v) = my_mean(MEEG([trials.response_visibilityCode]==v,:,:));
end
visibility = ERF;
save([data_path '_visibility.mat'],'visibility', 'time', 'labels');

%% seen / unseen
clear ERF
ERF(:,:,1) = my_mean(MEEG([trials.response_visibilityCode]>1,:,:));
ERF(:,:,2) = my_mean(MEEG([trials.response_visibilityCode]==1,:,:));
seenUnseen = ERF;
save([data_path '_seenUnseen.mat'],'seenUnseen', 'time', 'labels');


%% visibility x present
clear ERF
for v = 4:-1:1
    for p = 1:2
        sel = [trials.response_visibilityCode]==v & [trials.present]==mod(p,2);
        ERF(:,:,v,p) = my_mean(MEEG(sel,:,:));
    end
end
visibilityXpresent = ERF;
save([data_path '_visibilityXpresent.mat'],'visibilityXpresent', 'time', 'labels');

%% seen / unseen x present
clear ERF
for p = 1:2
    ERF(:,:,1,p) = my_mean(MEEG([trials.response_visibilityCode]>1 & [trials.present]==mod(p,2),:,:));
    ERF(:,:,2,p) = my_mean(MEEG([trials.response_visibilityCode]==1 & [trials.present]==mod(p,2),:,:));
end
seenXpresent = ERF;
save([data_path '_seenXpresent.mat'],'seenXpresent', 'time', 'labels');

%% accuracy
clear ERF
ERF(:,:,1) = my_mean(MEEG([trials.correct]==1,:,:));
ERF(:,:,2) = my_mean(MEEG([trials.correct]==0,:,:));
correctIncorrect = ERF;
save([data_path '_correctIncorrect.mat'],'correctIncorrect', 'time', 'labels');

%% response button
clear ERF
ERF(:,:,1) = my_mean(MEEG([trials.response_tilt]==-1,:,:));
ERF(:,:,2) = my_mean(MEEG([trials.response_tilt]==1,:,:));
responseTilt = ERF;
save([data_path '_responseTilt.mat'],'responseTilt', 'time', 'labels');

%% tilt
clear ERF
ERF(:,:,1) = my_mean(MEEG([trials.tilt]==-1,:,:));
ERF(:,:,2) = my_mean(MEEG([trials.tilt]==1,:,:));
tilt = ERF;
save([data_path '_tilt.mat'],'tilt', 'time', 'labels');

%% contrast
clear ERF
for c = 1:4
    ERF(:,:,c) = my_mean(MEEG([trials.contrast]==c,:,:));
end
contrast = ERF;
save([data_path '_contrast.mat'],'contrast','time','labels');

%% lambda
clear ERF
for l = 1:2
    sel = [trials.present]==1 & [trials.lambda]==l;
    ERF(:,:,l) = my_mean(MEEG(sel,:,:));
end
lambda = ERF;
save([data_path '_lambda.mat'],'lambda','time','labels');

%% orientationXvisibility
clear ERF
for o = 1:6
    for v = 1:4
        sel = [trials.orientation]==o & [trials.response_visibilityCode]==v & [trials.present] == 1;
        ERF(:,:,o,v) = my_mean(MEEG(sel,:,:));
    end
end
orientationXvisibility = ERF;
save([data_path '_orientationXvisibility.mat'],'orientationXvisibility','time','labels');

%% orientation x lambda
clear ERF
for o = 1:6
    for l = 1:2
        sel = [trials.present]==1 & [trials.orientation]==o & [trials.lambda]==l;
        ERF(:,:,o,l) = my_mean(MEEG(sel,:,:));
    end
end
orientationXlambda = ERF;
save([data_path '_orientationXlambda.mat'],'orientationXlambda','time','labels');

%% to be completed (interaction etc)
