function predict_angle = decode_reg2angle_predict(predict_x,predict_y,angles)% combine two predictions
px = squeeze(predict_x-2); % remove 2 to get back to normal
py = squeeze(predict_y-2);
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
