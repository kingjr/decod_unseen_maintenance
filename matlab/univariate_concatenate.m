SubjectsList ={'ak130184' 'el130086' 'ga130053' 'gm130176' 'hn120493'...
    'ia130315' 'jd110235' 'jm120476' 'ma130185' 'mc130295'...
    'mj130216' 'mr080072' 'oa130317' 'rg110386' 'sb120316'...
    'tc120199' 'ts130283' 'yp130276' 'av130322' 'ps120458'};


%% concatenate Event Related Fields
% ErfList = {'angles', 'presentAbsent','visibility', 'seenUnseen', 'visibilityXpresent', ...
%     'seenXpresent', 'correctIncorrect', 'responseTilt', 'tilt'};
% clear erf
% for s = length(SubjectsList):-1:1
%     s
%     for e = length(ErfList):-1:1
%         var_name = ErfList{e};
%         load([path 'data/' SubjectsList{s} '/erf/' SubjectsList{s} '_erf_' var_name '.mat'], var_name);
%         eval(['erf(s).' var_name '=' var_name  ';']);
%         clear var_name
%     end   
% end
% save([path 'data/across_subjects/withinSubject_erf.mat'], 'erf');
% clear erf

%% concatenate within-subjects univariate stats on Event related fields
UnivariateList = {'angleAnova', 'angleCirc','presentAbsent', 'seenUnseen', ...
    'visibilityXpresent', 'presentXseen', 'correctIncorrect', 'leftRightResponse','tilt',...
    'lambda','orientationXlambda', 'orientationXvisibility'};
clear univariate
for s = length(SubjectsList):-1:1
    s
    for uni = length(UnivariateList):-1:1
        var_name = UnivariateList{uni};
        load([path 'data/' SubjectsList{s} '/univariate/' SubjectsList{s} '_univariate_' var_name '.mat'], var_name);
        load([path 'data/across_subjects/withinSubject_univariate.mat'], 'univariate');
        eval(['univariate(s).' var_name '=' var_name  ';']);
        save([path 'data/across_subjects/withinSubject_univariate.mat'], 'univariate');
        clear var_name univariate
    end    
end

%% concatenate mean time frequency

%% concatenate within-subjects univariate stats on time frequency

%% concatenate mean gamma power

%% concatenate within-subjects univariate stats on gamma power
