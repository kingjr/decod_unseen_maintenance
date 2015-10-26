FOIs            = [6.5 9.5 12 17.5 29 70 105];
%% plot gamma decoding of orientation
sp=0;
for foi = FOIs
    sp=sp+1;
    subplot(2,4,sp),
    set(gca,'fontsize',15)
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
    imagesc(time(toi),[],squeeze(nanmean(all_probas)));
    title(num2str(foi))
end
saveas(gcf,['/home/niccolo/vboxshared/DOCUP/timeFrequency/orientationDecoding_powspctr'],'jpg')

%% plot probas decoding (applies to all binary classifications)
sp=0;
contrasts = {'visibilityPresent_SVC'};
for c=1:length(contrasts)
    for foi = FOIs([2 6 7]) % these are the current fois that I computed
        sp=sp+1;
        subplot(2,4,sp),
        set(gca,'fontsize',15)
        for s=1:20
            % load subject data
            subject = SubjectsList{s};
            data_path ='/media/niccolo/Yupi/Paris/data/';
            results=load([data_path subject '/mvpas/timeFreq/' subject '_preprocessed_' num2str(round(foi)) 'Hz_' contrasts{c} '_results.mat'],'probas','y');
            
            % concatenate
            for class = 1:2
                across_subjects(s,:,class) = mean(results.probas(1, results.y==class, :, 1, class),2);
                across_subjects(s,:,class) = across_subjects(s,:,class) - mean(across_subjects(s,:,class),2);
            end
        end
        hold on
        plot_eb(time(toi),across_subjects(:,:,1),'color',[1 0 0])
        hold on
        plot_eb(time(toi),across_subjects(:,:,2),'color',[0 1 0])
        title(num2str(foi))
    end 
end