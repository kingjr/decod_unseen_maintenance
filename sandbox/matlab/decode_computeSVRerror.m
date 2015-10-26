function [trial_proportion predict_angle radius] = decode_computeSVRerror(predicted_x,predicted_y,angles,res)
% This function was first called decode_reg2angle.m
%[trial_proportion predict_angle radius] = decode_computeSVRerror(predicted_x,predicted_y,angles,res)
% this function takes the two predictions of an SVR to transform them into
% predicted angles and predicted radi;
% predicted_x ans predicted_y are vectors of predictions
% angle is the true angle.
% the prediction error is estimated by the difference between the
% recombined x-y prediction and the true angle.
% the resolution (res) specify the bin width for the 2D histogram


if nargin == 3, res = 6;end

% combine two predictions
px = squeeze(predicted_x-2); % remove 2 to get back to normal
py = squeeze(predicted_y-2);
[theta radius] = cart2pol(px,py); % gives the predicted angle theta and the predicted radius

% realign to get single tuning curve across angles
predict_angle = [];
for a = 6:-1:1
    % select trials with angles a
    sel = angles==a;
    % relign across angle categories
    predict_angle(sel,:,:) = mod((pi+mod(theta(sel,:,:),2*pi)-2*deg2rad(-15+30*a))/2,pi);
end
sel = isnan(angles);
predict_angle(sel,:,:) = NaN;


% tuning function
borns =@(m,M,n) m:(M-m)/n:M;
borns =@(m,M,n) (m+(M-m)/n/2):(M-m)/n:(M+(M-m)/n/2);

% compute proportion of trials correctly predicted
trial_proportion = hist(predict_angle-pi/2,borns(-pi/2,pi/2,res));
trial_proportion = reshape(trial_proportion,[size(trial_proportion,1) size(predict_angle,2) size(predict_angle,3)]);
trial_proportion(1,:,:) = trial_proportion(1,:,:)+trial_proportion(end,:,:);
trial_proportion(end,:,:) = trial_proportion(1,:,:);
% get proportion of trials
trial_proportion = trial_proportion./repmat(sum(trial_proportion),[size(trial_proportion,1),1,1]);