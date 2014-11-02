%% fix triggers
%---- load behavior
load([data_path 'behavior/' sbj_initials '.mat'], 'trials');
sel = SelList{sbj_number};
if 0
    %---- plot
    clf;hold on;plot(data.trialinfo);plot([trials.ttl_value],'r');
    figure;
    clf;hold on;plot(data.trialinfo);plot([trials(sel).ttl_value],'r');
end
%---- force consistency
trials = trials(sel);
%---- save
save([data_path 'behavior/' sbj_initials '_fixed.mat'], 'trials');

