%% add toolboxes

%% load data
% behavioral
load('ma13018573538241054_ma130185final.mat');
% meg decoding probe
load('results_probe_ma130185.mat');

%% reorder orentations
probe = 1+mod([trials.orientation] + [trials.tilt],6)';
target = [trials.orientation]';
clear probas;
for a = 1:6
    sel = probe==a;
    probas(sel,:,:) = results_probe.probas(1,sel,:,:,[a:end 1:(a-1)]);
end
imagesc(squeeze(mean(probas))')


%% reorder according to target orientation
tilt=[trials.tilt]';
clear probas;
for a = 1:6
    sel = probe==a & tilt==-1;
    probas(sel,:,:) = results_probe.probas(1,sel,:,:,[a:end 1:(a-1)]);
end
imagesc(squeeze(mean(probas))');
