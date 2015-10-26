%% plot SVC t_all

%% load time and label data from a standard subject
subject = SubjectsList{1};
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
labels = data.label;
time = data.time{1};

%% define images path
im_path = 'F:/Paris/images_acrossSubjs/';

%% concatenate subjects
close all
contrasts = {'tilt' 'lambda' 'responseButton'};
for c = 1:3
    clear probas
    for s = length(SubjectsList):-1:1
        subject = SubjectsList{s};
        s
        %% load data
        load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
        results = load([path 'data/' subject '/mvpas/' subject '_preprocessed_' contrasts{c} '_SVC_tAll_results.mat'],'probas', 'y','dims');
        toi = time(results.dims);
        
        %% plot difference between classes probas
        prob = results.probas;
        probas(s,:,:) = squeeze(nanmean(prob(1,:,:,:,1),2)-nanmean(prob(1,:,:,:,2),2));
    end
    % plot
    figure
    imagesc(toi,toi,squeeze(nanmean(probas)))
    colorbar;
    set(gca,'fontsize',24,'ydir','normal')
    ylabel('train time');xlabel('test time');
    title(contrasts{c})
end

