SubjectsList ={'ak130184' 'el130086' 'ga130053' 'gm130176' 'hn120493'...
    'ia130315' 'jd110235' 'jm120476' 'ma130185' 'mc130295'...
    'mj130216' 'mr080072' 'oa130317' 'rg110386' 'sb120316'...
    'tc120199' 'ts130283' 'yp130276' 'av130322' 'ps120458'};

SelList={...
    [1:700 700:840];...                 % ak_130184
    [1:765 765:840];...                 % el_130086
    [1:405 405:581 581:840];...         % ga_130053
    [1:60 60:219 219:244 244:840];...   % gm130176
    [1:89 91:460 460:840];...           % hn120493
    [1:230 232:840];...                 % ia130315
    [1:755 757:840];...                 % jd_110235
    [1:538 538:840];...                 % jm120476
    [1:840];...                         % ma130185
    [1:596 598:840];...                 % mc_130295
    [1:298 298:805 805:840];...         % mj_130216
    [1:535 535:770 770:840];...         % mr080072
    [1:310 310:398 398:840];...         % oa130317
    [1:481 481:776 776:840];...         % rg110386
    [1:840];...                         % sb_120316
    [2:840]; ...                        % tc120199
    [3:840];...                         % ts_130283
    [1:71 71:268 268:840];...           % yp130276
    [1:147 147:314 314:840];...         % av130322: sick guy
    [1:693 695:840]};%                   % ps120458: always correct
SelFileOrder = {...
    [2:6],...       % ak_130184
    [2:6],...       % el_130086
    [2:6],...       % ga_130053
    [3:7],...       % gm130176
    [2:6],...       % hn120493
    [2:6],...       % ia130315
    [2:6],...       % jd_110235
    [2:6],...       % jm120476
    [2:6],...       % ma130185
    [2:6],...       % mc_130295
    [2:4 6:7],...   % mj_130216
    [6 2:5],...     % mr080072
    [2:6],...       % oa130317
    [2:6],...       % rg110386
    [3:4 6 8 11],...% sb_120316
    [2:6],...       % tc120199
    [3:7],...       % ts_130283
    [2:6],...       % yp130276;
    [2:6],...       % av130322: sick guy
    [2:6]};         % ps120458: always correct
numberOfSubjects=length(SubjectsList);

% check folders
for sbj_number = 1:length(SelList)
    sbj_initials = SubjectsList{sbj_number};
    %dir([path 'data/' sbj_initials])
    %dir([path 'data/' sbj_initials '/behavior/'])
    %dir([path 'data/' sbj_initials '/fif/'])
    %dir([path 'data/' sbj_initials '/preprocessed/'])
%     try
%     movefile(...
%         [path 'data/' sbj_initials '/preprocessed/' sbj_initials '_preprocessed.mat'],...
%         [path 'data/' sbj_initials '/preprocessed/old_' sbj_initials '_preprocessed.mat'])
%     end
% dir([path 'data/' sbj_initials '/mvpas/'])
% mkdir([path 'data/' sbj_initials '/figures/']);
end
