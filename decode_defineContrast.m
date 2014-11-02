%% define contrast SVR
switch cfg.clf_type
    case 'SVC'
        switch contrast
            case 'targetAngle';
                angles                  = deg2rad([trials.orientation]*30-15); % from degrees to radians
                angles([trials.present]==0)  = 0; % remove absent trials
                class = angles;
            case 'probeAngle'
                angles                  = [trials.orientation]*30-15;
                angles                  = angles+[trials.tilt]*30; % add or remove 30°
                class = angles;
            case 'responseButton'
                class = [trials.response_tilt]+2;
                class([trials.response_tilt]==0) = 0;
            case 'tilt'
                class = [trials.tilt]+2;
                class([trials.tilt]==0) = 0;
            case 'visibility'
                class = 1+([trials.response_visibilityCode]>1);
            case 'visibilityPresent'
                class = 1+([trials.response_visibilityCode]>1);
                class([trials.present]==0) = 0;
            case 'presentAbsent'
                class = [trials.present]+1;
            case 'accuracy'
                class = [trials.correct]+1;
                class(isnan([trials.correct])) = 0;
            case 'lambda'
                class = [trials.lambda];
            case '4visibilitiesPresent'
                class = [trials.response_visibilityCode];
                class([trials.present]==0) = 0;
            case 'reported'
                % to be done
        end
    case 'SVR'
        switch contrast
            case '4visibilitiesPresent'
                class = [trials.response_visibilityCode];
                class([trials.present]==0) = 0;
            case 'targetAngle';
                angles                  = deg2rad([trials.orientation]*30-15); % from degrees to radians
                
                % x coordinate
                x                       = 2+cos(2*angles); % get cos (add 2 to get only predictor >0, otherwised used as generalized... I know I know)
                x([trials.present]==0)  = 0; % remove absent trials
                % y coordinate
                y                       = 2+sin(2*angles); % get sin (y coordinate)
                y([trials.present]==0)  = 0;% remove absent trials
                
            case 'probeAngle'
                angles                  = [trials.orientation]*30-15;
                angles                  = angles+[trials.tilt]*30; % add or remove 30°
                angles                  = mod(angles, 180);
                angles                  = deg2rad(angles);
                
                x                       = 2+cos(2*angles); % get cos (add 2 to get only predictor >0, otherwise used as generalized... I know I know)
                y                       = 2+sin(2*angles); % get sin (y coordinate)
            case 'reported'
        end
end
