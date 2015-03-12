%% plot visibility present SVR tAll

%% load time and label data from a standard subject
subject = SubjectsList{1};
load([path 'data/' subject '/preprocessed/' subject '_preprocessed.mat'], 'data');
labels = data.label;
time = data.time{1};

%% define images path
im_path = 'F:/Paris/images_acrossSubjs/';

%% concatenate subjects
close all
contrast = '4visibilitiesPresent';
for s = 18:-1:1
    tic
    subject = SubjectsList{s};
    s
    %% load data
    load([path 'data/' subject '/behavior/' subject '_fixed.mat'],'trials');
    results = load([path 'data/' subject '/mvpas/' subject '_preprocessed_' contrast '_SVR_tAll_results.mat'],'predict', 'y','dims');
    toi = time(results.dims);
    
    %% compute several measures of decoding performance
    %-----------------squared error ---------------------------------------
    try
        sqerror = (squeeze(results.predict) - repmat(results.y', [1 sz(results.predict, [3,4])])).^2;
        MSE(s,:,:) = squeeze(nanmean(sqerror));
    catch me
    end
    %-------------------correlation----------------------------------------
    for t=1:length(toi)
        for tt=1:length(toi)
            X = squeeze(results.predict);
            Y = repmat(results.y', [1 sz(results.predict, [3,4])]);
            [rho(t,tt) pval(t,tt)] = corr(X(:,t,tt),Y(:,t,tt));
        end
    end
    RHOS(s,:,:) = rho;
    PVAL(s,:,:) = pval;
    toc
end

measures = {'MSE' 'RHOS' 'PVAL'};
for i=1:3
    figure
    eval(['imagesc(toi,toi,squeeze(nanmean(' measures{i} ')));']);
    colorbar;
    set(gca,'fontsize',24,'ydir','normal')
    title(measures{i})
    
    plot2svg([path_images '/4visibilitiesPresent_SVR_tAll_' measures{i}])
end


