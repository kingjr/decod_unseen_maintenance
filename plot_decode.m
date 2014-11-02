%% Plotting
% figure of targetAngle
% load results

switch cfg.clf_type
    case 'SVR'
        results_x = load([data_path 'mvpas/' subject '_preprocessed_' contrast '_' cfg.clf_type cfg.gentime '_x_results.mat']); % classifier (and file) name
        results_y = load([data_path 'mvpas/' subject '_preprocessed_' contrast '_' cfg.clf_type cfg.gentime '_y_results.mat']); % classifier (and file) name
    case 'SVC'
        results = load([data_path 'mvpas/' subject '_preprocessed_' contrast '_' cfg.clf_type cfg.gentime '_results.mat']); % classifier (and file) name
end
% plot
switch cfg.clf_type
    case 'SVR'
        % relign predictions
        switch contrast
            case 'targetAngle';
                [trial_prop predict] = decode_reg2angle(...
                    results_x,....
                    results_y,...
                    [trials([trials.present]==1).orientation]);
            case 'probeAngle'
                [trial_prop predict] = decode_reg2angle(...
                    results_x,....
                    results_y,...
                    mod([trials.orientation]+[trials.tilt]-1,6)+1);
            case '4visibilitiesPresent'
                % to be done
        end
        figure;
        %                 % separate visible versus invisible
        %                 % invisible
        %                 visible = [trials([trials.present]==1).response_visibilityCode];
        %                 train_angles = [trials([trials.present]==1).orientation];
        %                 train_angles(visible>1) = NaN;
        %                 [trial_prop predict] = decode_reg2angle(results_x,results_y,train_angles);
        %                 subplot(1,2,1);
        %                 imagesc(squeeze(trial_prop(round(end/2),:,:))), [.05 .17];
        %                 set(gca,'ydir', 'normal');axis image;
        %                 % visible
        %                 train_angles = [trials([trials.present]==1).orientation];
        %                 train_angles(visible<=1) = NaN;
        %                 [trial_prop predict] = decode_reg2angle(results_x,results_y,train_angles);
        %                 subplot(1,2,2);
        %                 imagesc(squeeze(trial_prop(round(end/2),:,:)), [.05 .17]);
        %                 set(gca,'ydir', 'normal');axis image;
        
        
        switch cfg.gentime
            case '_tAll'
                
                % plot gentime on closest angle distance
                imagesc(squeeze(trial_prop(round(end/2),:,:)));
                set(gca,'ydir', 'normal');
                
                % plot p value
                [p z] = circ_rtest(2*predict-pi);
                imagesc(time(cfg.dims),time(cfg.dims),squeeze(-log10(p)));
                set(gca,'ydir', 'normal');
                
            otherwise
                imagesc(trial_prop);%,[0 .5]);
                imagesc(time(cfg.dims),[],trial_prop);%,[0 .5]);
                
        end
        
        
    case 'SVC'
        % plot
        % realign angles to the 4th line
        switch contrast
            case 'targetAngle'
                probas = [];
                rads = unique(results.y); %angles in radiants
                for a = 1:6
                    sel = results.y == rads(a);
                    probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
                end
                figure();
                imagesc(time(cfg.dims),[],squeeze(trimmean(probas(:,:,[4:6 1:4]),80))');
                xlabel('time (ms)');ylabel('angle distance');
                set(gca,'FontSize',24,'YTickLabel',90:-30:-90);colorbar;
            
            case 'probeAngle'
                probas = [];
                
                % realign angles to the 4th line
                angles = [15:30:165];
                for a = 1:6
                    sel = results.y==angles(a);
                    probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
                end
                figure();
                imagesc(time(cfg.dims),[],squeeze(trimmean(probas(:,:,[4:6 1:4]),80))');
                xlabel('time (ms)');ylabel('angle distance');
                set(gca,'FontSize',24,'YTickLabel',90:-30:-90);colorbar;
            
            otherwise
                [~,~,class] = unique(results.y);
                clf;hold on;
                plot_eb(time(cfg.dims),squeeze(results.probas(1,class==1,:,1,1)),[0 0 1]);
                plot_eb(time(cfg.dims),squeeze(results.probas(1,class==2,:,1,1)),[1 0 0]);
                title(contrast)
        end
end