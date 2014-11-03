function [class1 class2] = decode_defineContrast(cfg,trials)
class1=[];class2=[];

%% define contrast SVC
switch cfg.clf_type
    case 'SVC'
        switch cfg.contrast
            case 'targetAngle';
                angles                  = deg2rad([trials.orientation]*30-15); % from degrees to radians
                angles([trials.present]==0)  = 0; % remove absent trials
                class1 = angles;
            case 'probeAngle'
                angles                  = [trials.orientation]*30-15;
                angles                  = angles+[trials.tilt]*30; % add or remove 30°
                class1 = angles;
            case 'responseButton'
                class1 = [trials.response_tilt]+2;
                class1([trials.response_tilt]==0) = 0;
            case 'tilt'
                class1 = [trials.tilt]+2;
                class1([trials.tilt]==0) = 0;
            case 'visibility'
                class1 = 1+([trials.response_visibilityCode]>1);
            case 'visibilityPresent'
                class1 = 1+([trials.response_visibilityCode]>1);
                class1([trials.present]==0) = 0;
            case 'presentAbsent'
                class1 = [trials.present]+1;
            case 'accuracy'
                class1 = [trials.correct]+1;
                class1(isnan([trials.correct])) = 0;
            case 'lambda'
                class1 = [trials.lambda];
            case '4visibilitiesPresent'
                class1 = [trials.response_visibilityCode];
                class1([trials.present]==0) = 0;
            case 'reported'
                % to be done
        end
    case 'SVR'
        switch cfg.contrast
            case '4visibilitiesPresent'
                class1 = [trials.response_visibilityCode];
                class1([trials.present]==0) = 0;
            case 'targetAngle';
                angles                  = deg2rad([trials.orientation]*30-15); % from degrees to radians
                
                % x coordinate
                class1                       = 2+cos(2*angles); % get cos (add 2 to get only predictor >0, otherwised used as generalized... I know I know)
                class1([trials.present]==0)  = 0; % remove absent trials
                % y coordinate
                class2                       = 2+sin(2*angles); % get sin (y coordinate)
                class2([trials.present]==0)  = 0;% remove absent trials
                
            case 'probeAngle'
                angles                  = [trials.orientation]*30-15;
                angles                  = angles+[trials.tilt]*30; % add or remove 30°
                angles                  = mod(angles, 180);
                angles                  = deg2rad(angles);
                
                class1                      = 2+cos(2*angles); % get cos (add 2 to get only predictor >0, otherwise used as generalized... I know I know)
                class1([trials.present]==0) = 0; % remove absent trials (not necessary for probe decoding but good for script consistency)
                class2                      = 2+sin(2*angles); % get sin (y coordinate)
                class2([trials.present]==0) = 0; % remove absent trials
            case 'reported'
        end
end
