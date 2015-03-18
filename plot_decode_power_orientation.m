%% plot gamma decoding of orientation
for foi = FOIs
    for s=1:20
        % load subject data
        subject = SubjectsList{s};
        data_path ='/media/niccolo/Yupi/Paris/data/';
        results=load([data_path subject '/mvpas/timeFreq/' subject '_preprocessed_' num2str(round(foi)) 'Hz_targetAngle_SVC_results.mat'],'probas','y');
        
        % concatenate
        probas = [];
        rads = unique(results.y); %angles in radiants
        for a = 1:6
            sel = results.y==rads(a);
            probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
        end
        all_probas(s,:,:) = squeeze(trimmean(probas(:,:,[4:6 1:4]),80))';
    end
    figure
    imagesc(time(toi),[],squeeze(nanmean(all_probas)));
    title(num2str(foi))
end