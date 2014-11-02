auc = [];
probas = [];
% loop across subjects
for s = 1:20
    % load data
    subject = SubjectsList{s};
    data_path = [path 'data/' subject '/'] ;
    file_behavior   = [data_path 'behavior/' subject '_fixed.mat'];
    load(file_behavior, 'trials');
    results=load([path 'data/' subject '/mvpas/' subject '_preprocessed_accuracy_SVC_results.mat']);
    
    auc(s,:) = colAUC(squeeze(results.probas(1,:,:,1,1)), results.y);
    probas(s,:,1) = squeeze(mean(results.probas(1,results.y==1,:,1,1),2));
    probas(s,:,2) = squeeze(mean(results.probas(1,results.y==2,:,1,1),2));
    
end

plot_eb(time(toi), auc);

figure;
plot_eb(time(toi), probas(:,:,1), [1 0 0]);
hold on
plot_eb(time(toi), probas(:,:,2), [0 0 1]);

clf
plot_eb(time(toi), probas(:,:,1)-probas(:,:,2));

imagesc(time(toi), [], probas(:,:,1)-probas(:,:,2));
imagesc(time(toi), [], auc);
